import torch
import torch.nn as nn
from timm.layers import ClassifierHead


class ClassificationHead(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_classes,
        dropout=0.1,
        patch_aggregation_method="avg",
        cls_token_available=True,
    ):
        """
        Args:
            embed_dim (int): size of the embedding.
            num_classes (int): Number of output classes.
            dropout (float): Dropout rate applied before the output layer.
            patch_aggregation_method (string): "cls_token" for taking the class token, "avg" or "sum"
                                                for aggregating the individual token vectors
        """
        super(ClassificationHead, self).__init__()

        self.fc = ClassifierHead(embed_dim, num_classes, "", dropout)

        self.patch_aggregation_method = patch_aggregation_method
        self.cls_token_available = cls_token_available

    def forward(self, x):

        if self.patch_aggregation_method is not None:
            if self.patch_aggregation_method == "cls_token":
                assert self.cls_token_available
                x = x[:, 0]
            elif self.patch_aggregation_method == "avg":
                x = x[:, 1:].mean(dim=1) if self.cls_token_available else x.mean(dim=1)
            elif self.patch_aggregation_method == "sum":
                x = x[:, 1:].sum(dim=1) if self.cls_token_available else x.sum(dim=1)

        x = self.fc(x)

        return x


import torch
import torch.nn as nn
from typing import Optional

class ClassifierHeadCFE2d(nn.Module):
    """
    Drop-in alternative to the existing 2D ClassifierHead.
    - API matches: __init__(in_features, num_classes, pool_type='avg', drop_rate=0., use_conv=False, input_fmt='NCHW')
    - Forward contract matches: forward(x, pre_logits: bool = False)
    - Expects NCHW (B, C, H, W) like the original.
    - Difference: applies a lightweight Conv-BN-ReLU "CFE" stack BEFORE global pooling.
      (By default it preserves channel count so pre_logits shape stays consistent.)
    """
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        pool_type: str = 'avg',
        drop_rate: float = 0.,
        use_conv: bool = True,
        input_fmt: str = 'NCHW',
        # extra knobs (optional; safe defaults preserve shapes)
        cfe_depth: int = 2,              # how many Conv-BN-ReLU blocks before pooling
        cfe_mid_channels: Optional[int] = None,  # if None, stays at in_features
        cfe_kernel_size: int = 3,
        cfe_padding: Optional[int] = None,       # default = kernel_size // 2 (same-spatial)
        cfe_stride: int = 1,
        cfe_bias: bool = False,
        cfe_use_bn: bool = True,
    ):
        super().__init__()
        assert input_fmt == 'NCHW', "ClassifierHeadCFE2d expects NCHW input"
        self.in_features = in_features
        self.use_conv = use_conv
        self.input_fmt = input_fmt

        # --- CFE stack that preserves channels by default ---
        out_ch = in_features if cfe_mid_channels is None else cfe_mid_channels
        pad = cfe_kernel_size // 2 if cfe_padding is None else cfe_padding

        layers = []
        c_in = in_features
        for _ in range(max(0, cfe_depth)):
            layers.append(nn.Conv2d(c_in, out_ch, kernel_size=cfe_kernel_size, stride=cfe_stride,
                                    padding=pad, bias=cfe_bias))
            if cfe_use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            c_in = out_ch
        self.cfe = nn.Sequential(*layers) if layers else nn.Identity()

        # --- Global pool & classifier identical semantics to the original head ---
        from .classifier import create_classifier  # same helpers as the existing ClassifierHead
        global_pool, fc = create_classifier(
            in_features=out_ch,
            num_classes=num_classes,
            pool_type=pool_type,
            use_conv=use_conv,
            input_fmt=input_fmt,
        )
        self.global_pool = global_pool
        self.drop = nn.Dropout(drop_rate)
        self.fc = fc
        self.flatten = nn.Flatten(1) if use_conv and pool_type else nn.Identity()

    def reset(self, num_classes: int, pool_type: Optional[str] = None):
        # Match the original behavior so external callers still work
        from .classifier import _create_fc  # same helper as original
        if pool_type is not None and pool_type != self.global_pool.pool_type:
            from .classifier import create_classifier
            self.global_pool, self.fc = create_classifier(
                self.in_features,
                num_classes,
                pool_type=pool_type,
                use_conv=self.use_conv,
                input_fmt=self.input_fmt,
            )
            self.flatten = nn.Flatten(1) if self.use_conv and pool_type else nn.Identity()
        else:
            num_pooled_features = self.in_features * self.global_pool.feat_mult()
            self.fc = _create_fc(
                num_pooled_features,
                num_classes,
                use_conv=self.use_conv,
            )

    def forward(self, x, pre_logits: bool = False):
        # x: (B, C, H, W), same as original
        x = self.cfe(x)             # <-- extra feature mixing (new behavior)
        x = self.global_pool(x)     # identical semantics to original
        x = self.drop(x)
        if pre_logits:
            return self.flatten(x)  # identical contract
        x = self.fc(x)
        return self.flatten(x)



# ---- Alternative MLP-style head (drop-in, same constructor & forward signature) ----
import torch
import torch.nn as nn
from typing import Optional, Tuple, List

class ClassificationHeadAlt(nn.Module):
    """
    Drop-in alternative to the existing ClassificationHead.
    - Same API:
        __init__(embed_dim, num_classes, dropout=0.1,
                 patch_aggregation_method='avg', cls_token_available=True)
        forward(x, pre_logits: bool=False)
    - Accepts either tokens (B, N, C) with the same aggregation logic,
      or a feature vector (B, C) (what ResEncoder returns).
    - Implements a small MLP head (Linear->BN->ReLU->Dropout)*k -> Linear [-> optional BN on logits]
    """
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        patch_aggregation_method: Optional[str] = "avg",
        cls_token_available: bool = True,
        # extra knobs (optional; defaults are safe)
        mlp_dims: Tuple[int, ...] = (256, 128, 64, 32),
        mlp_dropouts: Tuple[float, ...] = (0.3, 0.3, 0.3, 0.2),
        use_logits_bn: bool = True,
    ):
        super().__init__()
        self.patch_aggregation_method = patch_aggregation_method
        self.cls_token_available = cls_token_available

        # Pre-classifier dropout (mirror timm ClassifierHead behaviour)
        self.pre_drop = nn.Dropout(dropout)

        # Build MLP stack
        blocks: List[nn.Module] = []
        in_dim = embed_dim
        for i, out_dim in enumerate(mlp_dims):
            p = mlp_dropouts[i] if i < len(mlp_dropouts) else mlp_dropouts[-1]
            blocks += [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True), nn.Dropout(p)]
            in_dim = out_dim
        self.mlp = nn.Sequential(*blocks) if blocks else nn.Identity()

        # Final classifier
        self.fc_out = nn.Linear(in_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.bn_out = nn.BatchNorm1d(num_classes) if (use_logits_bn and num_classes > 0) else nn.Identity()

    # keep the same token-aggregation semantics as the original
    def _aggregate_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_aggregation_method is None:
            return x
        if self.patch_aggregation_method == "cls_token":
            assert self.cls_token_available, "cls_token not available"
            return x[:, 0]
        if self.patch_aggregation_method == "avg":
            return x[:, 1:].mean(dim=1) if self.cls_token_available else x.mean(dim=1)
        if self.patch_aggregation_method == "sum":
            return x[:, 1:].sum(dim=1) if self.cls_token_available else x.sum(dim=1)
        raise ValueError(f"Unknown patch_aggregation_method={self.patch_aggregation_method}")

    def forward(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        # Accept either (B, N, C) tokens or (B, C) vector
        if x.ndim == 3:
            x = self._aggregate_tokens(x)
        elif x.ndim != 2:
            raise ValueError(f"ClassificationHeadAlt expects (B,N,C) or (B,C), got shape {tuple(x.shape)}")

        x = self.pre_drop(x)
        feats = self.mlp(x)
        if pre_logits:
            return feats
        logits = self.fc_out(feats)
        return self.bn_out(logits)

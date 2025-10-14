import random
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler   # <-- ADDED


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        data_root_dir,
        name,
        batch_size,
        train_transforms,
        test_transforms,
        random_batches,
        num_workers,
        prepare_data_per_node,
        fold,
        use_weighted_sampling,
        *args,
        **kwargs
    ):
        super(BaseDataModule, self).__init__()

        self.data_path = Path(data_root_dir)  # / name
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.random_batches = random_batches
        self.num_workers = num_workers
        self.prepare_data_per_node = prepare_data_per_node
        self.fold = fold
        self.use_weighted_sampling = use_weighted_sampling

        # --- NEW: default scaling range for weights so your existing code works
        # allow override via kwargs, e.g., weight_scaling_range=(1.0, 5.0)
        self.weight_scaling_range = kwargs.get("weight_scaling_range", (1.0, 10.0))

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: str) -> None:
        pass

    # --- NEW: helper to get labels from the train dataset robustly
    def _extract_labels(self, dataset):
        # Common attributes used by many datasets
        if hasattr(dataset, "labels"):
            return np.asarray(dataset.labels)
        if hasattr(dataset, "y"):
            return np.asarray(dataset.y)
        if hasattr(dataset, "targets"):
            return np.asarray(dataset.targets)
        if hasattr(dataset, "df") and hasattr(dataset.df, "columns") and ("label" in dataset.df.columns):
            return dataset.df["label"].to_numpy()
        if hasattr(dataset, "samples"):  # e.g., torchvision ImageFolder-style [(path, label), ...]
            return np.asarray([lbl for _, lbl in dataset.samples])

        # Fallback: try indexing label as the 2nd item of each sample
        try:
            return np.asarray([dataset[i][1] for i in range(len(dataset))])
        except Exception as e:
            raise RuntimeError(
                "Could not infer labels from the training dataset. "
                "Implement compute_sample_weights in your DataModule or ensure the dataset exposes labels."
            ) from e

    # --- NEW: generic balanced weights using inverse class frequency
    def compute_sample_weights(self):
        labels = self._extract_labels(self.train_dataset).astype(int)
        if labels.size == 0:
            raise RuntimeError("Empty label array when computing sample weights.")

        # handle missing classes robustly
        num_classes = int(labels.max()) + 1
        class_counts = np.bincount(labels, minlength=num_classes)
        class_counts[class_counts == 0] = 1  # avoid div by zero
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]

        # return as torch tensor (float)
        return torch.as_tensor(sample_weights, dtype=torch.float)

    def train_dataloader(self):
        if self.use_weighted_sampling:
            print("Initializing weighted random sampler...")

            # Compute weights for each sample
            sample_weights = self.compute_sample_weights()

            # Log weight statistics before scaling
            print(
                f"Weight statistics - Before scaling: "
                f"Max: {sample_weights.max().item():.4f}, "
                f"Min: {sample_weights.min().item():.4f}, "
                f"Mean: {sample_weights.mean().item():.4f}"
            )

            # Handle case where all weights are identical (or zero range)
            if torch.isclose(sample_weights.max(), sample_weights.min()):
                print("Warning: All samples have the same weight. Using uniform sampling instead.")
                sampler = RandomSampler(
                    self.train_dataset,
                    replacement=True,
                    num_samples=len(self.train_dataset),
                )
            else:
                # Scale weights to specified range
                min_weight = sample_weights.min()
                max_weight = sample_weights.max()
                min_scale, max_scale = self.weight_scaling_range

                scaled_sample_weights = min_scale + (max_scale - min_scale) * (
                    (sample_weights - min_weight) / (max_weight - min_weight)
                )

                # Log weight statistics after scaling
                print(
                    f"Weight statistics - After scaling: "
                    f"Max: {scaled_sample_weights.max().item():.4f}, "
                    f"Min: {scaled_sample_weights.min().item():.4f}"
                )

                # Create weighted sampler
                sampler = WeightedRandomSampler(
                    weights=scaled_sample_weights,
                    num_samples=len(self.train_dataset),
                    replacement=True,
                )

            trainloader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                persistent_workers=True,
                sampler=sampler,         # NOTE: when sampler is set, do NOT also set shuffle
            )

        elif not self.random_batches:
            trainloader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                persistent_workers=True,
            )

        else:
            print("RandomSampler with replacement is used!")
            random_sampler = RandomSampler(
                self.train_dataset,
                replacement=True,
                num_samples=len(self.train_dataset),
            )
            trainloader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                persistent_workers=True,
                sampler=random_sampler,
            )

        return trainloader


    def val_dataloader(self):
        valloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            persistent_workers=True,
        )

        return valloader

    def test_dataloader(self):
        testloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            persistent_workers=True,
        )

        return testloader

    def predict_dataloader(self):
        print(f"[INFO] Predicting on {len(self.test_dataset)} test samples.")
        predictloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            persistent_workers=True,
        )

        return predictloader


def seed_worker(worker_id):
    """
    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    to fix https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
    ensures different random numbers each batch with each worker every epoch while keeping reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

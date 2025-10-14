import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load predictions and metadata
base_dir = "/bdm-das/ADSP_v1/nnssl_dataset/nnssl_results/ADNI_ResEnc_VoCo_pretrained_frozenEnc_ord_reg_no_Warmup_lr_1e-4_BalancedSampler/AgeReg/checkpoints"
pred_df = pd.read_excel(f"{base_dir}/predictions.xlsx")
meta_df = pd.read_csv("/bdm-das/ADSP_v1/data/df_downstream_checked.csv")

# Extract basename from SS_IMG_PATH for matching
meta_df["basename"] = meta_df["SS_IMG_PATH"].apply(lambda x: x.split("/")[-1].replace(".nii.gz", ""))
basename_to_dx = dict(zip(meta_df["basename"], meta_df["DX"]))
pred_df["DX"] = pred_df["PatientID"].map(basename_to_dx)

# --------- Group by Disease (DX) ----------
dx_grouped = pred_df.groupby("DX")[["MAE", "Error"]].mean()
dx_grouped.loc["Overall"] = pred_df[["MAE", "Error"]].mean()
dx_grouped.columns = ["Mean_Absolute_Error", "Mean_Error"]
dx_grouped.to_excel(f"{base_dir}/mae_me_by_disease.xlsx")

# --------- Group by Age Group ----------
bins = [0, 54, 59, 64, 69, 74, 79, 84, 89, 94, 100]
labels = ["<55", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85-89", "90-94", "95-100"]
pred_df["AgeGroup"] = pd.cut(pred_df["GroundTruth"], bins=bins, labels=labels, right=True)

age_grouped = pred_df.groupby("AgeGroup")[["MAE", "Error"]].mean()
age_grouped.loc["Overall"] = pred_df[["MAE", "Error"]].mean()
age_grouped.columns = ["Mean_Absolute_Error", "Mean_Error"]
age_grouped.to_excel(f"{base_dir}/mae_me_by_age_group.xlsx")

# --------- Save full predictions (optional) ----------
pred_df.to_excel(f"{base_dir}/predictions_with_dx.xlsx", index=False)

# --------- Plot ME by Age Group with Trendline and Labels ----------
age_grouped_no_total = age_grouped.drop(index="Overall")
grouped_me = age_grouped_no_total["Mean_Error"]
bar_positions = np.arange(len(grouped_me))

# Midpoints for curve fitting
midpoints = {
    "<55": 50,
    "55-59": 57,
    "60-64": 62,
    "65-69": 67,
    "70-74": 72,
    "75-79": 77,
    "80-84": 82,
    "85-89": 87,
    "90-94": 92,
    "95-100": 97
}
x = np.array([midpoints[label] for label in grouped_me.index])
y = grouped_me.values

# Polynomial fit
coeffs = np.polyfit(x, y, deg=2)
poly = np.poly1d(coeffs)
x_fit = np.linspace(x.min(), x.max(), 100)
y_fit = poly(x_fit)
x_pos_fit = np.interp(x_fit, x, bar_positions)

# Use numeric x (age midpoints) for both bars and trendline
plt.figure(figsize=(10, 5))
bars = plt.bar(x, y, width=4, color="orange", edgecolor="black", label="Mean Error (ME)")

# Add value labels on each bar
for xi, yi in zip(x, y):
    plt.text(xi, yi + 0.1 * np.sign(yi), f"{yi:.2f}",
             ha='center', va='bottom' if yi >= 0 else 'top', fontsize=9)

# Plot trendline
plt.plot(x_fit, y_fit, linestyle="--", color="blue", label="Trendline (2nd deg poly)")
# plt.axhline(0, color='red', linestyle='--', label="Zero Error")

# Set x-ticks to match age bins
plt.xticks(x, grouped_me.index)

plt.title("Mean Error (ME) by Age Group")
plt.ylabel("Mean Error")
plt.xlabel("Age Group")
plt.grid(axis='y')
plt.legend()
plt.tight_layout()
plt.savefig(f"{base_dir}/mean_error_by_age_group_with_trend.png")
plt.show()
import matplotlib
matplotlib.use("Agg")  # HPC-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------
# Translate + Noise modes
# -----------------------------
translate_modes = [
    "NotTranslated",
    "FullyTranslated",
    "PromptTranslate",
    "PostProcessDifferent",
    "PostProcessSame"
]

noise_modes = ["NO_NOISE", "PARAPHRASE", "SYNONYM"]

# -----------------------------
# Random placeholder data (5×3)
# -----------------------------
data = np.random.rand(len(translate_modes), len(noise_modes))
df = pd.DataFrame(data, index=translate_modes, columns=noise_modes)

# -----------------------------
# Plot heatmap
# -----------------------------
plt.figure(figsize=(8, 5))

# Use a lighter, pleasant colormap
plt.imshow(df, cmap="RdYlGn", interpolation="nearest", vmin=-1.0, vmax=1.0)

# Colorbar
plt.colorbar(label="Metric Value")

# Ticks
plt.xticks(np.arange(len(noise_modes)), noise_modes, rotation=45)
plt.yticks(np.arange(len(translate_modes)), translate_modes)

# Annotate values in each grid cell
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        value = df.iloc[i, j]
        plt.text(
            j, i,
            f"{value:.3f}",             # round to 3 decimals
            ha="center", va="center",
            color="black", fontsize=9   # black text = readable on light colormap
        )

plt.title("Heatmap: Translate Mode × Noise Mode")
plt.tight_layout()
plt.savefig("translate_noise_heatmap_annotated.png")

print("Saved heatmap to translate_noise_heatmap_annotated.png")
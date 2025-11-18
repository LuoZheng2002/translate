import matplotlib
matplotlib.use("Agg")  # HPC-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path

# -----------------------------
# Translate + Noise modes
# -----------------------------
translate_modes = [
    "NT", # Not Translated
    "FT", # Fully Translated
    "PT", # Fully Translated + Prompt Translate
    "PPD", # Fully translated + Post-Process Different
    "PPS", # Fully translated + Post-Process Same
    "PTPS" # Fully Translated + Prompt Translate + Post-Process Same
]

noise_modes = ["NO_NOISE", "PARAPHRASE", "SYNONYM"]

# Mapping from file naming conventions to display names
translate_mode_mapping = {
    "": "NT",      # Not Translated (no postfix)
    "_f": "FT",    # Fully Translated
    "_pt": "PT",   # Prompt Translate
    "_ppd": "PPD", # Post-Process Different
    "_pps": "PPS", # Post-Process Same
    "_ptps": "PTPS" # Prompt Translate + Post-Process Same
}

noise_mode_mapping = {
    "": "NO_NOISE",  # No noise (no postfix)
    "_para": "PARAPHRASE",  # Paraphrase
    "_syno": "SYNONYM"      # Synonym
}


def generate_heatmap(model_name: str, output_dir: str = ".", result_dir: str = "result/score") -> None:
    """
    Generate a heatmap for a given model showing accuracy across translate and noise modes.

    Args:
        model_name: The model name (e.g., "llama3_1_8b", "gpt4o_mini", "qwen2_5_7b")
        output_dir: Directory to save the heatmap image (default: current directory)
        result_dir: Directory containing the score files (default: "result/score")
    """

    # Initialize data structure: dict[translate_mode][noise_mode] = accuracy
    data_dict = {}
    for tm in translate_modes:
        data_dict[tm] = {}
        for nm in noise_modes:
            data_dict[tm][nm] = None

    # Find all score files matching this model
    score_dir = Path(result_dir)

    # Construct the pattern to match: BFCL_v4_multiple{model_name}*
    # The file naming convention from main.py is:
    # BFCL_v4_multiple{model_postfix}{language_postfix}{translate_mode_postfix}{noise_postfix}.json
    # We look for files containing the model_name postfix

    for score_file in score_dir.glob(f"BFCL_v4_multiple*{model_name}*.json"):
        try:
            with open(score_file, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                data = json.loads(first_line)
                accuracy = data.get("accuracy")

                if accuracy is None:
                    print(f"Warning: No 'accuracy' field in {score_file.name}")
                    continue

                # Extract translate and noise modes from filename
                # Remove prefix and suffix
                filename = score_file.stem  # Remove .json
                filename = filename.replace("BFCL_v4_multiple", "").replace(model_name, "")

                # Parse the remaining postfixes
                # Format: {language_postfix}{translate_mode_postfix}{noise_postfix}
                # Language postfix: _zh, _hi (we'll ignore for now since we're focusing on model)
                # Translate mode postfix: _f, _pt, _ppd, _pps, _ptps, or empty (NT)
                # Noise postfix: _para, _syno, or empty (NO_NOISE)

                translate_mode_str = ""
                noise_mode_str = ""

                # Extract noise mode (check for _para or _syno at the end)
                if filename.endswith("_para"):
                    noise_mode_str = "_para"
                    filename = filename[:-5]  # Remove _para
                elif filename.endswith("_syno"):
                    noise_mode_str = "_syno"
                    filename = filename[:-5]  # Remove _syno

                # Extract translate mode from what remains
                # Check for translate mode postfixes in what's left
                if "_ptps" in filename:
                    translate_mode_str = "_ptps"
                elif "_ppd" in filename:
                    translate_mode_str = "_ppd"
                elif "_pps" in filename:
                    translate_mode_str = "_pps"
                elif "_pt" in filename:
                    translate_mode_str = "_pt"
                elif "_f" in filename:
                    translate_mode_str = "_f"
                # else: empty string for NT

                # Convert to display names
                translate_mode = translate_mode_mapping.get(translate_mode_str, "UNKNOWN")
                noise_mode = noise_mode_mapping.get(noise_mode_str, "UNKNOWN")

                if translate_mode != "UNKNOWN" and noise_mode != "UNKNOWN":
                    data_dict[translate_mode][noise_mode] = accuracy
                    print(f"Loaded {score_file.name}: {translate_mode} + {noise_mode} = {accuracy:.3f}")
                else:
                    print(f"Warning: Could not parse {score_file.name} (translate: {translate_mode}, noise: {noise_mode})")

        except Exception as e:
            print(f"Error reading {score_file.name}: {e}")

    # Convert to DataFrame
    data = []
    for tm in translate_modes:
        row = []
        for nm in noise_modes:
            value = data_dict[tm][nm]
            row.append(value if value is not None else np.nan)
        data.append(row)

    df = pd.DataFrame(data, index=translate_modes, columns=noise_modes)

    # Check if we have any data
    if df.isna().all().all():
        print(f"Error: No valid data found for model '{model_name}'")
        return

    # Print summary
    print(f"\nData for model '{model_name}':")
    print(df)

    # Plot heatmap
    plt.figure(figsize=(8, 5))

    # Use a lighter, pleasant colormap
    plt.imshow(df, cmap="RdYlGn", interpolation="nearest", vmin=0.0, vmax=1.0)

    # Colorbar
    plt.colorbar(label="Accuracy")

    # Ticks
    plt.xticks(np.arange(len(noise_modes)), noise_modes, rotation=45)
    plt.yticks(np.arange(len(translate_modes)), translate_modes)

    # Annotate values in each grid cell
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            value = df.iloc[i, j]
            # Only annotate if we have data
            if not pd.isna(value):
                plt.text(
                    j, i,
                    f"{value:.3f}",             # round to 3 decimals
                    ha="center", va="center",
                    color="black", fontsize=9   # black text = readable on light colormap
                )

    plt.title(f"Heatmap: {model_name} - Translate Mode × Noise Mode")
    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"heatmap_{model_name}.png")
    plt.savefig(output_path)
    print(f"\nSaved heatmap to {output_path}")
    plt.close()


# Example usage
if __name__ == "__main__":
    # Generate heatmaps for different models
    models = ["llama3_1_8b", "llama3_1_70b", "qwen2_5_7b", "qwen2_5_14b", "gpt4o_mini"]

    for model in models:
        print(f"\n{'='*60}")
        print(f"Generating heatmap for {model}")
        print(f"{'='*60}")
        generate_heatmap(model)
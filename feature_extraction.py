import pandas as pd
import numpy as np
import os

INPUT_FILE = "data/processed/harth_4class.csv"
OUTPUT_FILE = "data/processed/harth_features.csv"

# 50 Hz = 50 readings per second
SAMPLE_RATE = 50

# 2-second window = 100 readings
WINDOW_SIZE = 100

# 50% overlap
STEP_SIZE = 50

FEATURE_COLUMNS = [
    "back_x", "back_y", "back_z",
    "thigh_x", "thigh_y", "thigh_z"
]

all_features = []

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df):,} samples.")
print("Starting feature extraction...\n")


def extract_features(window):

    features = {}

    for column in FEATURE_COLUMNS:

        values = window[column].values

        features[f"{column}_mean"] = np.mean(values)
        features[f"{column}_std"] = np.std(values)
        features[f"{column}_min"] = np.min(values)
        features[f"{column}_max"] = np.max(values)
        features[f"{column}_range"] = np.max(values) - np.min(values)
        features[f"{column}_var"] = np.var(values)

    # Overall acceleration magnitude
    back_magnitude = np.sqrt(
        window["back_x"] ** 2 +
        window["back_y"] ** 2 +
        window["back_z"] ** 2
    )

    thigh_magnitude = np.sqrt(
        window["thigh_x"] ** 2 +
        window["thigh_y"] ** 2 +
        window["thigh_z"] ** 2
    )

    features["back_magnitude_mean"] = back_magnitude.mean()
    features["back_magnitude_std"] = back_magnitude.std()

    features["thigh_magnitude_mean"] = thigh_magnitude.mean()
    features["thigh_magnitude_std"] = thigh_magnitude.std()

    return features


# Process each subject separately
for subject in df["subject"].unique():

    print(f"Processing {subject}...")

    subject_data = df[df["subject"] == subject].reset_index(drop=True)

    start = 0

    while start + WINDOW_SIZE <= len(subject_data):

        window = subject_data.iloc[start:start + WINDOW_SIZE]

        # Check that the whole window belongs to one activity
        labels = window["activity"].unique()

        if len(labels) == 1:

            feature_row = extract_features(window)

            feature_row["activity"] = labels[0]
            feature_row["subject"] = subject

            all_features.append(feature_row)

        start += STEP_SIZE


# Convert extracted features into DataFrame
features_df = pd.DataFrame(all_features)

# Create output directory
os.makedirs("data/processed", exist_ok=True)

# Save
features_df.to_csv(OUTPUT_FILE, index=False)

print("\n===================================")
print("FEATURE EXTRACTION COMPLETE")
print("===================================")

print(f"\nTotal feature windows: {len(features_df):,}")

print("\nClass distribution:")
print(features_df["activity"].value_counts())

print(f"\nNumber of features: {len(features_df.columns) - 2}")

print(f"\nSaved to:")
print(OUTPUT_FILE)
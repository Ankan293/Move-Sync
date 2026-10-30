import pandas as pd
import glob
import os

# ==============================
# MOVE SYNC - HARTH PREPROCESSING
# ==============================

INPUT_FOLDER = "data/activity/harth"
OUTPUT_FOLDER = "data/processed"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# HARTH labels we need
LABEL_MAP = {
    1: "WALKING",
    2: "RUNNING",
    6: "STANDING",
    7: "SITTING"
}

all_data = []

files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))

print(f"Found {len(files)} HARTH files.\n")

for file in files:
    filename = os.path.basename(file)

    print(f"Processing: {filename}")

    df = pd.read_csv(file)

    # Keep only our four activities
    df = df[df["label"].isin(LABEL_MAP.keys())].copy()

    # Convert numeric labels to activity names
    df["activity"] = df["label"].map(LABEL_MAP)

    # Add subject ID
    df["subject"] = filename.replace(".csv", "")

    # Keep only required columns
    df = df[
        [
            "timestamp",
            "back_x",
            "back_y",
            "back_z",
            "thigh_x",
            "thigh_y",
            "thigh_z",
            "label",
            "activity",
            "subject"
        ]
    ]

    all_data.append(df)

# Combine all subjects
combined = pd.concat(all_data, ignore_index=True)

# Save processed dataset
output_file = os.path.join(
    OUTPUT_FOLDER,
    "harth_4class.csv"
)

combined.to_csv(output_file, index=False)

print("\n===================================")
print("PREPROCESSING COMPLETE")
print("===================================")

print(f"\nTotal samples: {len(combined):,}")

print("\nActivity distribution:")
print(combined["activity"].value_counts())

print("\nSubjects:")
print(combined["subject"].nunique())

print(f"\nSaved to:")
print(output_file)
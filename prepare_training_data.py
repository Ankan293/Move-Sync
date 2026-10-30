import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==========================================
# MOVE SYNC - TRAINING DATA PREPARATION
# ==========================================

INPUT_FILE = "data/processed/harth_features.csv"
OUTPUT_FOLDER = "data/processed"

# Our four activity classes
CLASSES = [
    "WALKING",
    "RUNNING",
    "STANDING",
    "SITTING"
]

# ==========================================
# 1. LOAD DATA
# ==========================================

print("Loading feature dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Total windows loaded: {len(df):,}")


# ==========================================
# 2. KEEP ONLY OUR FOUR CLASSES
# ==========================================

df = df[df["activity"].isin(CLASSES)].copy()

print("\nOriginal class distribution:")
print(df["activity"].value_counts())


# ==========================================
# 3. BALANCE THE CLASSES
# ==========================================

# Find the smallest class
minimum_samples = df["activity"].value_counts().min()

print(f"\nBalancing each class to {minimum_samples:,} samples...")

balanced_parts = []

for activity in CLASSES:

    activity_data = df[df["activity"] == activity]

    # Randomly select the same number from every class
    activity_data = activity_data.sample(
        n=minimum_samples,
        random_state=42
    )

    balanced_parts.append(activity_data)


balanced_df = pd.concat(
    balanced_parts,
    ignore_index=True
)

# Shuffle the complete dataset
balanced_df = balanced_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


print("\nBalanced distribution:")
print(balanced_df["activity"].value_counts())


# ==========================================
# 4. SPLIT BY SUBJECT
# ==========================================

print("\nSplitting subjects into training and testing...")


subjects = balanced_df["subject"].unique()

print(f"Total subjects: {len(subjects)}")


train_subjects, test_subjects = train_test_split(
    subjects,
    test_size=0.20,
    random_state=42
)


train_df = balanced_df[
    balanced_df["subject"].isin(train_subjects)
].copy()

test_df = balanced_df[
    balanced_df["subject"].isin(test_subjects)
].copy()


print(f"Training subjects: {len(train_subjects)}")
print(f"Testing subjects: {len(test_subjects)}")

print(f"Training windows: {len(train_df):,}")
print(f"Testing windows: {len(test_df):,}")


# ==========================================
# 5. SELECT FEATURES
# ==========================================

# Remove activity and subject because they aren't input features
FEATURE_COLUMNS = [
    column
    for column in balanced_df.columns
    if column not in ["activity", "subject"]
]

print(f"\nNumber of input features: {len(FEATURE_COLUMNS)}")


X_train = train_df[FEATURE_COLUMNS].values
X_test = test_df[FEATURE_COLUMNS].values

y_train = train_df["activity"].values
y_test = test_df["activity"].values


# ==========================================
# 6. FEATURE SCALING
# ==========================================

print("\nScaling features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

# IMPORTANT:
# Test data is transformed using the scaler
# fitted only on training data.
X_test = scaler.transform(X_test)


# ==========================================
# 7. CONVERT LABELS TO NUMBERS
# ==========================================

label_mapping = {
    "SITTING": 0,
    "STANDING": 1,
    "WALKING": 2,
    "RUNNING": 3
}

y_train = np.array([
    label_mapping[label]
    for label in y_train
])

y_test = np.array([
    label_mapping[label]
    for label in y_test
])


# ==========================================
# 8. CREATE OUTPUT DIRECTORY
# ==========================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==========================================
# 9. SAVE DATA
# ==========================================

np.save(
    f"{OUTPUT_FOLDER}/X_train.npy",
    X_train
)

np.save(
    f"{OUTPUT_FOLDER}/X_test.npy",
    X_test
)

np.save(
    f"{OUTPUT_FOLDER}/y_train.npy",
    y_train
)

np.save(
    f"{OUTPUT_FOLDER}/y_test.npy",
    y_test
)

# Save scaler parameters
np.save(
    f"{OUTPUT_FOLDER}/scaler_mean.npy",
    scaler.mean_
)

np.save(
    f"{OUTPUT_FOLDER}/scaler_scale.npy",
    scaler.scale_
)


# ==========================================
# 10. FINAL INFORMATION
# ==========================================

print("\n==========================================")
print("TRAINING DATA PREPARATION COMPLETE")
print("==========================================")

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

print("\nLabel mapping:")
print(label_mapping)

print("\nTraining class distribution:")

for label, number in label_mapping.items():
    print(
        f"{label}: "
        f"{np.sum(y_train == number):,}"
    )

print("\nTesting class distribution:")

for label, number in label_mapping.items():
    print(
        f"{label}: "
        f"{np.sum(y_test == number):,}"
    )

print("\nFiles saved in:")
print(OUTPUT_FOLDER)
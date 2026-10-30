import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from pathlib import Path

# ==========================================
# MOVE SYNC - MODULE 1
# ANN / MLP ACTIVITY RECOGNITION
# ==========================================

print("Loading training data...")

# ------------------------------------------
# 1. LOAD DATA
# ------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

required_files = [
    DATA_DIR / "X_train.npy",
    DATA_DIR / "X_test.npy",
    DATA_DIR / "y_train.npy",
    DATA_DIR / "y_test.npy",
]

missing_files = [str(path) for path in required_files if not path.exists()]

if missing_files:
    raise FileNotFoundError(
        "Missing processed training data files. Expected files in "
        f"{DATA_DIR}. Missing: {', '.join(missing_files)}"
    )

X_train = np.load(DATA_DIR / "X_train.npy")
X_test = np.load(DATA_DIR / "X_test.npy")

y_train = np.load(DATA_DIR / "y_train.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

print(f"Training data: {X_train.shape}")
print(f"Testing data : {X_test.shape}")


# ------------------------------------------
# 2. CONVERT TO PYTORCH TENSORS
# ------------------------------------------

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.long
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.long
)


# ------------------------------------------
# 3. CREATE DATA LOADERS
# ------------------------------------------

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# ------------------------------------------
# 4. DEFINE ANN / MLP MODEL
# ------------------------------------------

class ActivityANN(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            # Input: 40 features
            nn.Linear(40, 128),

            nn.ReLU(),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Linear(64, 32),

            nn.ReLU(),

            # Output: 4 activities
            nn.Linear(32, 4)
        )

    def forward(self, x):

        return self.network(x)


model = ActivityANN()


print("\nModel architecture:")
print(model)


# ------------------------------------------
# 5. LOSS FUNCTION
# ------------------------------------------

criterion = nn.CrossEntropyLoss()


# ------------------------------------------
# 6. OPTIMIZER
# ------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ------------------------------------------
# 7. TRAINING
# ------------------------------------------

EPOCHS = 30

train_losses = []
train_accuracies = []

print("\nStarting training...\n")


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:

        # Forward pass
        outputs = model(X_batch)

        loss = criterion(
            outputs,
            y_batch
        )

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # Statistics
        total_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == y_batch
        ).sum().item()

        total += y_batch.size(0)


    epoch_loss = (
        total_loss / len(train_loader)
    )

    epoch_accuracy = (
        correct / total
    ) * 100

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)


    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Accuracy: {epoch_accuracy:.2f}%"
    )


# ------------------------------------------
# 8. TEST MODEL
# ------------------------------------------

print("\nEvaluating model...")

model.eval()

all_predictions = []
all_actual = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        outputs = model(X_batch)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.numpy()
        )

        all_actual.extend(
            y_batch.numpy()
        )


# ------------------------------------------
# 9. CLASSIFICATION REPORT
# ------------------------------------------

class_names = [
    "SITTING",
    "STANDING",
    "WALKING",
    "RUNNING"
]

print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================\n")

print(
    classification_report(
        all_actual,
        all_predictions,
        target_names=class_names
    )
)


# ------------------------------------------
# 10. CONFUSION MATRIX
# ------------------------------------------

cm = confusion_matrix(
    all_actual,
    all_predictions
)

print("\nConfusion Matrix:")
print(cm)


# ------------------------------------------
# 11. TEST ACCURACY
# ------------------------------------------

test_accuracy = (
    np.array(all_predictions)
    ==
    np.array(all_actual)
).mean() * 100

print(
    f"\nFinal Test Accuracy: "
    f"{test_accuracy:.2f}%"
)


# ------------------------------------------
# 12. SAVE MODEL
# ------------------------------------------

torch.save(
    model.state_dict(),
    "activity_ann_module1.pth"
)

print(
    "\nModel saved as: "
    "activity_ann_module1.pth"
)


# ------------------------------------------
# 13. PLOT TRAINING LOSS
# ------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "MOVE SYNC - Module 1 Training Loss"
)

plt.legend()

plt.grid(True)

plt.show()


# ------------------------------------------
# 14. PLOT TRAINING ACCURACY
# ------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    train_accuracies,
    label="Training Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.title(
    "MOVE SYNC - Module 1 Training Accuracy"
)

plt.legend()

plt.grid(True)

plt.show()
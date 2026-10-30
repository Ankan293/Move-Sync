import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================
# MOVE SYNC - MODULE 2
# BACKPROPAGATION + HYPERPARAMETER TUNING
# ============================================================

print("=" * 60)
print("MOVE SYNC - MODULE 2")
print("BACKPROPAGATION + HYPERPARAMETER TUNING")
print("=" * 60)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

print("\nLoading training data...")


# ============================================================
# 2. LOAD DATA
# ============================================================

X_train = np.load(DATA_DIR / "X_train.npy")
X_test = np.load(DATA_DIR / "X_test.npy")

y_train = np.load(DATA_DIR / "y_train.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

print(f"Training data: {X_train.shape}")
print(f"Testing data : {X_test.shape}")


# ============================================================
# 3. CREATE VALIDATION SET
# ============================================================
#
# IMPORTANT:
# The test set is kept completely separate.
# Hyperparameters are selected using validation data.
#

X_train_main, X_val, y_train_main, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

print("\nData split:")
print(f"Training   : {X_train_main.shape}")
print(f"Validation : {X_val.shape}")
print(f"Testing    : {X_test.shape}")


# ============================================================
# 4. CONVERT TO PYTORCH TENSORS
# ============================================================

X_train_tensor = torch.tensor(
    X_train_main,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train_main,
    dtype=torch.long
)

X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_val_tensor = torch.tensor(
    y_val,
    dtype=torch.long
)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.long
)


# ============================================================
# 5. ANN MODEL
# ============================================================

class ActivityANN(nn.Module):

    def __init__(
        self,
        hidden1=128,
        hidden2=64,
        hidden3=32,
        dropout=0.0,
        initialization="xavier"
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(40, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden3, 4)
        )

        self.initialize_weights(initialization)


    # --------------------------------------------------------
    # Weight Initialization
    # --------------------------------------------------------

    def initialize_weights(self, method):

        for layer in self.network:

            if isinstance(layer, nn.Linear):

                if method == "xavier":

                    nn.init.xavier_uniform_(
                        layer.weight
                    )

                elif method == "kaiming":

                    nn.init.kaiming_uniform_(
                        layer.weight,
                        nonlinearity="relu"
                    )

                elif method == "normal":

                    nn.init.normal_(
                        layer.weight,
                        mean=0.0,
                        std=0.02
                    )

                else:

                    nn.init.xavier_uniform_(
                        layer.weight
                    )

                nn.init.zeros_(layer.bias)


    def forward(self, x):

        return self.network(x)


# ============================================================
# 6. TRAINING FUNCTION
# ============================================================

def train_model(config):

    model = ActivityANN(
        hidden1=config["hidden1"],
        hidden2=config["hidden2"],
        hidden3=config["hidden3"],
        dropout=config["dropout"],
        initialization=config["initialization"]
    )

    # --------------------------------------------------------
    # Loss Function
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    #
    # weight_decay = L2 regularization
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    val_dataset = TensorDataset(
        X_val_tensor,
        y_val_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False
    )

    best_val_accuracy = 0.0
    best_state = None

    train_losses = []
    val_accuracies = []


    # ========================================================
    # TRAINING / BACKPROPAGATION
    # ========================================================

    for epoch in range(config["epochs"]):

        model.train()

        total_loss = 0.0

        for X_batch, y_batch in train_loader:

            # ------------------------------------------------
            # FORWARD PROPAGATION
            # ------------------------------------------------

            outputs = model(X_batch)

            # ------------------------------------------------
            # LOSS CALCULATION
            # ------------------------------------------------

            loss = criterion(
                outputs,
                y_batch
            )

            # ------------------------------------------------
            # BACKPROPAGATION
            # ------------------------------------------------
            #
            # 1. Clear old gradients
            # 2. Calculate gradients
            # 3. Update weights
            #

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()


        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                outputs = model(X_batch)

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                correct += (
                    predictions == y_batch
                ).sum().item()

                total += y_batch.size(0)


        val_accuracy = (
            correct / total
        ) * 100

        epoch_loss = (
            total_loss / len(train_loader)
        )

        train_losses.append(epoch_loss)
        val_accuracies.append(val_accuracy)


        # Save best validation model

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_state = {
                key: value.cpu().clone()
                for key, value in model.state_dict().items()
            }


    # Restore best model

    model.load_state_dict(best_state)

    return (
        model,
        best_val_accuracy,
        train_losses,
        val_accuracies
    )


# ============================================================
# 7. HYPERPARAMETER EXPERIMENTS
# ============================================================

experiments = [

    {
        "name": "Baseline",
        "hidden1": 128,
        "hidden2": 64,
        "hidden3": 32,
        "learning_rate": 0.001,
        "batch_size": 64,
        "dropout": 0.0,
        "weight_decay": 0.0,
        "initialization": "xavier",
        "epochs": 20
    },

    {
        "name": "Lower Learning Rate",
        "hidden1": 128,
        "hidden2": 64,
        "hidden3": 32,
        "learning_rate": 0.0005,
        "batch_size": 64,
        "dropout": 0.0,
        "weight_decay": 0.0,
        "initialization": "xavier",
        "epochs": 20
    },

    {
        "name": "Higher Learning Rate",
        "hidden1": 128,
        "hidden2": 64,
        "hidden3": 32,
        "learning_rate": 0.002,
        "batch_size": 64,
        "dropout": 0.0,
        "weight_decay": 0.0,
        "initialization": "xavier",
        "epochs": 20
    },

    {
        "name": "Dropout",
        "hidden1": 128,
        "hidden2": 64,
        "hidden3": 32,
        "learning_rate": 0.001,
        "batch_size": 64,
        "dropout": 0.20,
        "weight_decay": 0.0,
        "initialization": "xavier",
        "epochs": 20
    },

    {
        "name": "L2 Regularization",
        "hidden1": 128,
        "hidden2": 64,
        "hidden3": 32,
        "learning_rate": 0.001,
        "batch_size": 64,
        "dropout": 0.0,
        "weight_decay": 0.0001,
        "initialization": "xavier",
        "epochs": 20
    },

    {
        "name": "Kaiming Initialization",
        "hidden1": 128,
        "hidden2": 64,
        "hidden3": 32,
        "learning_rate": 0.001,
        "batch_size": 64,
        "dropout": 0.0,
        "weight_decay": 0.0,
        "initialization": "kaiming",
        "epochs": 20
    },

    {
        "name": "Larger Network",
        "hidden1": 256,
        "hidden2": 128,
        "hidden3": 64,
        "learning_rate": 0.001,
        "batch_size": 64,
        "dropout": 0.20,
        "weight_decay": 0.0001,
        "initialization": "kaiming",
        "epochs": 20
    },

    {
        "name": "Optimized Candidate",
        "hidden1": 128,
        "hidden2": 64,
        "hidden3": 32,
        "learning_rate": 0.0005,
        "batch_size": 128,
        "dropout": 0.20,
        "weight_decay": 0.0001,
        "initialization": "kaiming",
        "epochs": 25
    }
]


# ============================================================
# 8. RUN EXPERIMENTS
# ============================================================

results = []

best_model = None
best_config = None
best_val_accuracy = -1

best_losses = []
best_val_curve = []


print("\n")
print("=" * 60)
print("STARTING HYPERPARAMETER EXPERIMENTS")
print("=" * 60)


for index, config in enumerate(experiments):

    print(
        f"\nExperiment {index + 1}/"
        f"{len(experiments)}: "
        f"{config['name']}"
    )

    print(
        f"Learning Rate : {config['learning_rate']}"
    )

    print(
        f"Batch Size    : {config['batch_size']}"
    )

    print(
        f"Dropout       : {config['dropout']}"
    )

    print(
        f"L2            : {config['weight_decay']}"
    )

    print(
        f"Initialization: {config['initialization']}"
    )

    model, val_accuracy, losses, val_curve = train_model(
        config
    )

    results.append({
        "experiment": config["name"],
        "validation_accuracy": val_accuracy,
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "dropout": config["dropout"],
        "weight_decay": config["weight_decay"],
        "initialization": config["initialization"]
    })

    print(
        f"Best Validation Accuracy: "
        f"{val_accuracy:.2f}%"
    )


    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_model = model

        best_config = config

        best_losses = losses

        best_val_curve = val_curve


# ============================================================
# 9. DISPLAY EXPERIMENT RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("HYPERPARAMETER RESULTS")
print("=" * 60)

for result in results:

    print(
        f"{result['experiment']:25s} "
        f"{result['validation_accuracy']:.2f}%"
    )


print("\n")
print("BEST CONFIGURATION")
print("=" * 60)

print(
    f"Experiment      : {best_config['name']}"
)

print(
    f"Hidden Layers   : "
    f"{best_config['hidden1']} → "
    f"{best_config['hidden2']} → "
    f"{best_config['hidden3']}"
)

print(
    f"Learning Rate   : "
    f"{best_config['learning_rate']}"
)

print(
    f"Batch Size      : "
    f"{best_config['batch_size']}"
)

print(
    f"Dropout         : "
    f"{best_config['dropout']}"
)

print(
    f"L2              : "
    f"{best_config['weight_decay']}"
)

print(
    f"Initialization  : "
    f"{best_config['initialization']}"
)

print(
    f"Validation Acc. : "
    f"{best_val_accuracy:.2f}%"
)


# ============================================================
# 10. FINAL TEST EVALUATION
# ============================================================

print("\n")
print("=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)


test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False
)

best_model.eval()

all_predictions = []
all_actual = []


with torch.no_grad():

    for X_batch, y_batch in test_loader:

        outputs = best_model(X_batch)

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


test_accuracy = accuracy_score(
    all_actual,
    all_predictions
) * 100


print(
    f"\nOptimized Test Accuracy: "
    f"{test_accuracy:.2f}%"
)


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

class_names = [
    "SITTING",
    "STANDING",
    "WALKING",
    "RUNNING"
]

print("\nClassification Report:\n")

print(
    classification_report(
        all_actual,
        all_predictions,
        target_names=class_names
    )
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_actual,
    all_predictions
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# 13. SAVE OPTIMIZED MODEL
# ============================================================

model_path = (
    MODEL_DIR /
    "activity_ann_module2_optimized.pth"
)

torch.save(
    best_model.state_dict(),
    model_path
)

print(
    f"\nOptimized model saved to:\n"
    f"{model_path}"
)


# ============================================================
# 14. PLOT BEST MODEL LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    best_losses,
    label="Training Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "MOVE SYNC - Module 2 Training Loss"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 15. PLOT VALIDATION ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    best_val_curve,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.title(
    "MOVE SYNC - Module 2 Validation Accuracy"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("MODULE 2 COMPLETE")
print("=" * 60)

print(
    f"Module 1 Baseline Test Accuracy : 97.35%"
)

print(
    f"Module 2 Optimized Test Accuracy: "
    f"{test_accuracy:.2f}%"
)

print(
    f"Improvement: "
    f"{test_accuracy - 97.35:+.2f} percentage points"
)

print("=" * 60)
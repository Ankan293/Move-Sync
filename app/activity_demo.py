import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


# ============================================================
# MOVE SYNC
# MODULE 1 + MODULE 2 DEMONSTRATION
# ============================================================

st.set_page_config(
    page_title="MOVE SYNC",
    page_icon="🏃",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"

MODEL1_PATH = MODEL_DIR / "activity_ann_module1.pth"
MODEL2_PATH = MODEL_DIR / "activity_ann_module2_optimized.pth"


# ============================================================
# ACTIVITY CLASSES
# ============================================================

CLASS_NAMES = [
    "SITTING",
    "STANDING",
    "WALKING",
    "RUNNING"
]


ACTIVITY_ICONS = {
    "SITTING": "🪑",
    "STANDING": "🧍",
    "WALKING": "🚶",
    "RUNNING": "🏃"
}


# ============================================================
# MODULE 1
# BASELINE ANN
#
# Architecture:
# 40 → 128 → 64 → 32 → 4
# ============================================================

class ActivityANN_Module1(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(40, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 4)
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# MODULE 2
# OPTIMIZED MODEL
#
# Checkpoint architecture:
# 40 → 128 → 64 → 32 → 4
#
# Includes Dropout layers because the saved checkpoint
# uses network.0, network.3, network.6 and network.9
# ============================================================

class ActivityANN_Module2(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(40, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 4)
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    model1 = ActivityANN_Module1()
    model2 = ActivityANN_Module2()

    model1.load_state_dict(
        torch.load(
            MODEL1_PATH,
            map_location="cpu"
        )
    )

    model2.load_state_dict(
        torch.load(
            MODEL2_PATH,
            map_location="cpu"
        )
    )

    model1.eval()
    model2.eval()

    return model1, model2


# ============================================================
# LOAD TEST DATA
# ============================================================

@st.cache_data
def load_test_data():

    X_test = np.load(
        DATA_DIR / "X_test.npy"
    )

    y_test = np.load(
        DATA_DIR / "y_test.npy"
    )

    return X_test, y_test


# ============================================================
# HEADER
# ============================================================

st.title("🏃 MOVE SYNC")

st.subheader(
    "AI-Powered Human Activity Recognition"
)

st.write(
    "Module 1: ANN/MLP Activity Recognition  |  "
    "Module 2: Backpropagation & Hyperparameter Tuning"
)


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    model1, model2 = load_models()

    X_test, y_test = load_test_data()

except Exception as e:

    st.error("Could not load project files.")

    st.code(str(e))

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🎛️ Demo Controls")

st.sidebar.write(
    f"Total test samples: {len(X_test):,}"
)

sample_number = st.sidebar.number_input(
    "Select Test Sample",
    min_value=0,
    max_value=len(X_test) - 1,
    value=0,
    step=1
)


# ============================================================
# SELECT SAMPLE
# ============================================================

sample = X_test[sample_number]

actual_label = int(
    y_test[sample_number]
)

sample_tensor = torch.tensor(
    sample,
    dtype=torch.float32
).unsqueeze(0)


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    output1 = model1(
        sample_tensor
    )

    output2 = model2(
        sample_tensor
    )

    probabilities1 = torch.softmax(
        output1,
        dim=1
    )[0]

    probabilities2 = torch.softmax(
        output2,
        dim=1
    )[0]


# ============================================================
# PREDICTED LABELS
# ============================================================

prediction1 = torch.argmax(
    probabilities1
).item()

prediction2 = torch.argmax(
    probabilities2
).item()


activity1 = CLASS_NAMES[prediction1]
activity2 = CLASS_NAMES[prediction2]
actual_activity = CLASS_NAMES[actual_label]

confidence1 = (
    probabilities1[prediction1].item() * 100
)

confidence2 = (
    probabilities2[prediction2].item() * 100
)


# ============================================================
# CURRENT ACTIVITY
# MODULE 2 IS THE OPTIMIZED MODEL
# ============================================================

st.markdown("---")

st.subheader("🎯 Current Activity")

st.markdown(
    f"""
    <div style="
        text-align: center;
        padding: 35px;
        border-radius: 20px;
        border: 2px solid rgba(128,128,128,0.35);
        margin-top: 10px;
        margin-bottom: 25px;
    ">

        <div style="
            font-size: 85px;
            line-height: 1.2;
        ">
            {ACTIVITY_ICONS[activity2]}
        </div>

        <div style="
            font-size: 42px;
            font-weight: 700;
            margin-top: 12px;
        ">
            {activity2}
        </div>

        <div style="
            font-size: 20px;
            margin-top: 10px;
        ">
            Module 2 Optimized Prediction
        </div>

        <div style="
            font-size: 28px;
            font-weight: 600;
            margin-top: 15px;
        ">
            {confidence2:.2f}% Confidence
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PREDICTION SUMMARY
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Actual Activity",
        f"{ACTIVITY_ICONS[actual_activity]} "
        f"{actual_activity}"
    )


with col2:

    st.metric(
        "Module 1",
        f"{ACTIVITY_ICONS[activity1]} "
        f"{activity1}",
        f"{confidence1:.2f}% confidence"
    )


with col3:

    st.metric(
        "Module 2",
        f"{ACTIVITY_ICONS[activity2]} "
        f"{activity2}",
        f"{confidence2:.2f}% confidence"
    )


# ============================================================
# CORRECTNESS
# ============================================================

st.markdown("---")

col1, col2 = st.columns(2)


with col1:

    if prediction1 == actual_label:

        st.success(
            "✅ Module 1 correctly classified this sample."
        )

    else:

        st.warning(
            "⚠️ Module 1 misclassified this sample."
        )


with col2:

    if prediction2 == actual_label:

        st.success(
            "✅ Module 2 correctly classified this sample."
        )

    else:

        st.warning(
            "⚠️ Module 2 misclassified this sample."
        )


# ============================================================
# CONFIDENCE DISPLAY
# ============================================================

st.markdown("---")

st.subheader("📊 Activity Confidence")


col1, col2 = st.columns(2)


# ============================================================
# MODULE 1 CONFIDENCE
# ============================================================

with col1:

    st.write("### Module 1 — Baseline")

    for i, name in enumerate(CLASS_NAMES):

        confidence = (
            probabilities1[i].item() * 100
        )

        st.write(
            f"**{ACTIVITY_ICONS[name]} {name}** — "
            f"{confidence:.2f}%"
        )

        st.progress(
            min(int(confidence), 100)
        )


# ============================================================
# MODULE 2 CONFIDENCE
# ============================================================

with col2:

    st.write("### Module 2 — Optimized")

    for i, name in enumerate(CLASS_NAMES):

        confidence = (
            probabilities2[i].item() * 100
        )

        st.write(
            f"**{ACTIVITY_ICONS[name]} {name}** — "
            f"{confidence:.2f}%"
        )

        st.progress(
            min(int(confidence), 100)
        )


# ============================================================
# MODEL COMPARISON
# ============================================================

st.markdown("---")

st.subheader("⚖️ Model Comparison")


col1, col2 = st.columns(2)


# ============================================================
# MODULE 1 INFORMATION
# ============================================================

with col1:

    st.write("### Module 1 — Baseline ANN")

    st.write(
        "**Architecture:** "
        "40 → 128 → 64 → 32 → 4"
    )

    st.write(
        "**Test Accuracy:** 97.35%"
    )

    st.write(
        "**Initialization:** Xavier"
    )

    st.write(
        "**Learning Rate:** 0.001"
    )

    st.write(
        "**Batch Size:** 64"
    )


# ============================================================
# MODULE 2 INFORMATION
# ============================================================

with col2:

    st.write("### Module 2 — Optimized ANN")

    st.write(
        "**Architecture:** "
        "40 → 128 → 64 → 32 → 4"
    )

    st.write(
        "**Test Accuracy:** 97.06%"
    )

    st.write(
        "**Best Validation Accuracy:** 99.06%"
    )

    st.write(
        "**Initialization:** Kaiming"
    )

    st.write(
        "**Learning Rate:** 0.001"
    )

    st.write(
        "**Batch Size:** 64"
    )


# ============================================================
# HYPERPARAMETER EXPERIMENTS
# ============================================================

st.markdown("---")

st.subheader(
    "🔬 Module 2 — Hyperparameter Experiments"
)


experiments = [
    ("Baseline", "98.83%"),
    ("Lower Learning Rate", "98.95%"),
    ("Higher Learning Rate", "98.95%"),
    ("Dropout", "98.78%"),
    ("L2 Regularization", "98.89%"),
    ("Kaiming Initialization", "99.06%"),
    ("Larger Network", "98.80%"),
    ("Optimized Candidate", "98.63%")
]


for experiment, accuracy in experiments:

    col1, col2 = st.columns([4, 1])

    with col1:

        st.write(
            f"**{experiment}**"
        )

    with col2:

        st.write(
            f"**{accuracy}**"
        )


# ============================================================
# INPUT INFORMATION
# ============================================================

st.markdown("---")

st.subheader("📥 Input Information")


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Sample Index",
        sample_number
    )


with col2:

    st.metric(
        "Input Features",
        len(sample)
    )


with col3:

    st.metric(
        "Actual Class",
        actual_activity
    )


# ============================================================
# FEATURE INFORMATION
# ============================================================

with st.expander(
    "🔍 View 40 Extracted Features"
):

    for i, value in enumerate(sample):

        st.write(
            f"Feature {i + 1}: "
            f"{value:.6f}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "MOVE SYNC | Human Activity Recognition | "
    "Module 1 + Module 2"
)
# MOVE SYNC

MOVE SYNC is a Python human activity recognition project. It includes a Streamlit demo that compares two trained activity classification models and a real-time webcam dashboard that detects pose-based movement.

## Requirements

- Windows (the commands below use PowerShell)
- Python 3.12
- A webcam for the real-time camera dashboard

### Activity demo

The sample-based Streamlit app uses:

```text
numpy==1.26.4
streamlit==1.62.0
torch==2.13.0
```

### Live camera dashboard

The webcam server uses:

```text
opencv-python==4.11.0
mediapipe==0.10.21
```

The Streamlit dashboard also uses `streamlit==1.62.0` and `requests==2.34.2`.

These versions match the project's current Python environments. CPU-only PyTorch is sufficient for the demo.

## Setup

In Command Prompt, open the project folder and run this one-line command to create a virtual environment and install all requirements:

```cmd
py -3.12 -m venv .venv && .venv\Scripts\python.exe -m pip install numpy==1.26.4 streamlit==1.62.0 torch==2.13.0 requests==2.34.2 opencv-python==4.11.0 mediapipe==0.10.21
```

This installs the camera and dashboard packages in the same environment. Use `.venv\Scripts\python.exe` to start the camera server and `.venv\Scripts\streamlit.exe` to start either dashboard.

## Run the activity demo

From the project folder:

```cmd
.venv\Scripts\streamlit.exe run app\activity_demo.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Run the live camera dashboard

Start the camera server in one PowerShell window:

```cmd
.venv\Scripts\python.exe camera\camera_server.py
```

Then start the dashboard in a second window:

```cmd
.venv\Scripts\streamlit.exe run app_camera.py
```

Open the local URL printed by Streamlit. Keep the camera server running while using the dashboard.

## Project data

The sample demo expects the trained model files in `models/` and the test arrays in `data/processed/`. The large HARTH source dataset and generated CSV files are excluded from this repository; use `prepare_harth.py` and `prepare_training_data.py` to regenerate processed data when the source dataset is available.

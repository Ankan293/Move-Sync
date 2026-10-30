import streamlit as st
import requests
import time


# ============================================================
# MOVE SYNC - REAL TIME CAMERA DASHBOARD
# ============================================================

st.set_page_config(
    page_title="MOVE SYNC",
    page_icon="🏃",
    layout="wide"
)


SERVER_URL = "http://127.0.0.1:8765"


# ============================================================
# TITLE
# ============================================================

st.title("🏃 MOVE SYNC")

st.subheader(
    "AI-Powered Real-Time Human Activity Recognition"
)

st.write(
    "Real-time pose-based activity detection using "
    "MediaPipe and OpenCV."
)


# ============================================================
# SERVER CHECK
# ============================================================

def get_status():

    try:

        response = requests.get(
            f"{SERVER_URL}/status",
            timeout=1
        )

        if response.status_code == 200:
            return response.json()

    except Exception:
        pass

    return None


def get_frame():

    try:

        response = requests.get(
            f"{SERVER_URL}/frame",
            timeout=2
        )

        if response.status_code == 200:
            return response.content

    except Exception:
        pass

    return None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Camera Controls")

st.sidebar.write(
    "Camera Server:"
)

status = get_status()

if status is None:

    st.sidebar.error(
        "Camera server is OFFLINE"
    )

    st.info(
        "Start camera\\camera_server.py "
        "inside camera_venv first."
    )

    st.stop()

else:

    st.sidebar.success(
        "Camera server is ONLINE"
    )


# ============================================================
# LIVE DASHBOARD
# ============================================================

st.markdown("---")

left, right = st.columns([1.4, 1])


# ============================================================
# CAMERA
# ============================================================

with left:

    st.subheader("📷 Live Camera")

    frame = get_frame()

    if frame:

        st.image(
            frame,
            use_container_width=True
        )

    else:

        st.warning(
            "Waiting for camera frame..."
        )


# ============================================================
# ACTIVITY
# ============================================================

with right:

    st.subheader("🎯 Current Activity")

    activity = status["activity"]

    movement = status["movement"]


    # --------------------------------------------------------
    # ACTIVITY
    # --------------------------------------------------------

    if activity == "SITTING":

        icon = "🪑"

    elif activity == "STANDING":

        icon = "🧍"

    elif activity == "WALKING":

        icon = "🚶"

    elif activity == "RUNNING":

        icon = "🏃"

    else:

        icon = "❓"


    st.markdown(
        f"""
        <div style="
            padding:30px;
            border-radius:15px;
            text-align:center;
            border:1px solid #cccccc;
            margin-bottom:20px;
        ">
            <div style="font-size:55px;">
                {icon}
            </div>

            <div style="
                font-size:32px;
                font-weight:bold;
                margin-top:10px;
            ">
                {activity}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # MOVEMENT
    # --------------------------------------------------------

    st.metric(
        "Movement Score",
        f"{movement:.2f}"
    )


# ============================================================
# ACTIVITY STATUS
# ============================================================

st.markdown("---")

st.subheader("📊 Activity Status")


activities = [
    "SITTING",
    "STANDING",
    "WALKING",
    "RUNNING"
]


cols = st.columns(4)


for i, name in enumerate(activities):

    with cols[i]:

        if activity == name:

            st.success(
                f"● {name}"
            )

        else:

            st.write(
                f"○ {name}"
            )


# ============================================================
# SYSTEM INFORMATION
# ============================================================

st.markdown("---")

st.subheader("⚙️ System Information")


info1, info2, info3 = st.columns(3)


with info1:

    st.metric(
        "Detection",
        "MediaPipe Pose"
    )


with info2:

    st.metric(
        "Camera",
        "OpenCV"
    )


with info3:

    st.metric(
        "Classes",
        "4"
    )


st.caption(
    "MOVE SYNC — Real-Time Human Activity Recognition"
)


# ============================================================
# AUTO REFRESH
# ============================================================

time.sleep(0.15)

st.rerun()
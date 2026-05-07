import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
import time
import os


SAVE_DIR = "saved_frames"
os.makedirs(SAVE_DIR, exist_ok=True)


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Live Object Detection & Tracing")
st.write("Real-time AI detection with counting, alerts, and image saving.")


st.sidebar.header("⚙️ Settings")

target_object = st.sidebar.selectbox(
    "Select object for alert:",
    ["person", "cell phone", "bottle"]
)

enable_save = st.sidebar.checkbox("💾 Save detected frames")


last_saved_time = 0


def video_frame_callback(frame):
    global last_saved_time

    img = frame.to_ndarray(format="bgr24")

    results = model.track(
        img,
        persist=True,
        conf=0.5,
        verbose=False
    )

    result = results[0]


    if result.boxes is not None:
        boxes = result.boxes
        class_ids = boxes.cls.cpu().numpy().astype(int)
        names = model.names

        detected_labels = [names[i] for i in class_ids]

    
        person_count = detected_labels.count("person")


        cv2.putText(
            img,
            f"People Count: {person_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

 
        if target_object in detected_labels:
            cv2.putText(
                img,
                f"ALERT: {target_object} detected!",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

 
        if enable_save and len(detected_labels) > 0:
            current_time = time.time()
            if current_time - last_saved_time > 3:
                filename = f"{SAVE_DIR}/frame_{int(current_time)}.jpg"
                cv2.imwrite(filename, img)
                last_saved_time = current_time


    annotated_frame = result.plot()

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)

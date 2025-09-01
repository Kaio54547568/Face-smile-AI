
import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from detection import detect_smiles_bgr

st.set_page_config(page_title="Smile Detector", page_icon="😀", layout="wide")
st.title("😀 Smile Detector")

st.markdown("Detect smiles from your webcam in **Realtime (WebRTC)** or via **Snapshot/Upload**.")

with st.sidebar:
    st.header("Settings")
    face_scale = st.slider("Face scaleFactor", 1.05, 1.5, 1.10, 0.01)
    face_neighbors = st.slider("Face minNeighbors", 3, 12, 5, 1)
    smile_scale = st.slider("Smile scaleFactor", 1.3, 2.0, 1.7, 0.1)
    smile_neighbors = st.slider("Smile minNeighbors", 10, 40, 22, 1)
    min_face_size = st.slider("Min face size (px)", 40, 200, 80, 5)
    draw_fps = st.checkbox("Show FPS (realtime)", True)

tabs = st.tabs(["Realtime (WebRTC)", "Snapshot / Upload"])

with tabs[0]:
    st.subheader("Realtime Webcam")
    rtc_config = RTCConfiguration(
        {
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )

    class VideoProcessor:
        def __init__(self):
            self._last_ts = None
            self._fps = 0.0

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            annotated, smiling = detect_smiles_bgr(img,face_scale=face_scale,face_neighbors=face_neighbors,smile_scale=smile_scale,smile_neighbors=smile_neighbors,min_face_size=min_face_size,)

            if draw_fps:
                # Ước lượng FPS
                import time
                now = time.time()
                if self._last_ts is not None:
                    dt = max(now - self._last_ts, 1e-6)
                    self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
                self._last_ts = now
                cv2.putText(annotated, f"FPS: {self._fps:.1f}", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(key="smile-webrtc",mode=WebRtcMode.SENDRECV,rtc_configuration=rtc_config,media_stream_constraints={"video": True, "audio": False},video_processor_factory=VideoProcessor,)

# Ảnh / Tải lên tab
with tabs[1]:
    st.subheader("Snapshot or Upload")
    c1, c2 = st.columns(2)

    with c1:
        snap = st.camera_input("Take a snapshot")
    with c2:
        uploaded = st.file_uploader("...or upload an image", type=["jpg", "jpeg", "png"])

    img_source = None
    if snap is not None:
        img_source = snap
    elif uploaded is not None:
        img_source = uploaded

    if img_source is not None:
        image = Image.open(img_source).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        annotated, smiling = detect_smiles_bgr(img_bgr,face_scale=face_scale,face_neighbors=face_neighbors,smile_scale=smile_scale,smile_neighbors=smile_neighbors,min_face_size=min_face_size,
        )
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"{'Smiling' if smiling else 'Not smiling'}")
    else:
        st.info("Take a snapshot or upload an image to run detection.")

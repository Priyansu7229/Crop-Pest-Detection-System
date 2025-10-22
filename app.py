import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load model
model = YOLO("best.pt")
model.fuse()
model.to("cpu")

st.title("Crop Pest Detection using YOLOv8")
st.write("Upload an image, video, or use webcam to detect pests in crops.")

option = st.radio("Select Input Source", ("Image", "Video"))

if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        results = model.predict(source=np.array(image), conf=0.5)
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="Detected Results", use_container_width=True)

elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        results = model.predict(source=tfile.name, conf=0.5, save=True)
        st.video(tfile.name)

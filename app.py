import streamlit as st
import cv2
import numpy as np
from PIL import Image

from utilities import largest_bounding_box, crop_single_person

st.title("📸 Smart Photo Cropper")

st.write(
    "Upload an image and automatically crop it around the person."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

ratio = st.selectbox(
    "Aspect Ratio (Ratio of photo width to height)",
    [
        "1:1",
        "2:3",
        "3:2",
        "4:3",
        "3:4",
        "4:5",
        "5:4",
        "9:16",
        "16:9"
    ]
)

input_w, input_h = map(int, ratio.split(":"))

padding = st.slider(
    "Padding",
    0.0,
    1.0,
    0.2,
    0.05
)

if uploaded_file is not None:
    img = np.array(Image.open(uploaded_file))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    bbox = largest_bounding_box(img)
    cropped = crop_single_person(
        img,
        bbox,
        input_w,
        input_h,
        padding,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            caption="Original"
        )

    with col2:
        st.image(
            cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB),
            caption="Cropped"
        )

    _, buffer = cv2.imencode(".jpg", cropped)

    st.download_button(
        "Download",
        buffer.tobytes(),
        file_name="cropped.jpg",
        mime="image/jpeg"
    )
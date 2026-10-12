import streamlit as st
import cv2
import numpy as np
from PIL import Image

from utilities import (
    detect_bounding_boxes,
    crop_single_object,
    draw_numbered_bounding_boxes
)

st.title("📸 Smart Photo Cropper")

st.write(
    "Upload an image and automatically crop it around the object."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

ratio = st.selectbox(
    "Aspect Ratio (Ratio of photo width to height)",
    [
        "1:1", "2:3", "3:2", "4:3", "3:4",
        "4:5", "5:4", "9:16", "16:9"
    ]
)

input_w, input_h = map(int, ratio.split(":"))

padding = st.slider(
    "Padding (the amount of space around the subject)",
    0.0,
    1.0,
    0.2,
    0.05
)

max_objects = st.selectbox(
    "Maximum objects to display",
    options=[3, 5, 10],
    index=1,
    help="Limits the number of detected objects available "
         "for selection. Larger objects are prioritised."
)

if uploaded_file is not None:
    img = np.array(Image.open(uploaded_file).convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Detect subjects
    bbox_list = detect_bounding_boxes(img, max_number_of_objects=max_objects)

    # Handle functions that return None when nothing is detected
    if bbox_list is None:
        bbox_list = []

    # Always display the original image
    st.image(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        caption="Original",
        use_container_width=True
    )

    # CASE 1: No subject detected
    if len(bbox_list) == 0:
        st.warning(
            "No detectable subject found. "
            "Please try another image. No crop has been generated."
        )

    else:
        # CASE 2: Exactly one subject — crop immediately
        if len(bbox_list) == 1:
            bbox = bbox_list[0]["bbox"]

        # CASE 3: Multiple subjects — wait for user selection
        else:
            annotated_img, label_dict = draw_numbered_bounding_boxes(
                img, bbox_list
            )

            st.image(
                cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                caption="Detected subjects — choose the main subject",
                use_container_width=True
            )

            selected_index = st.selectbox(
                "Select the main subject",
                options=range(len(bbox_list)),
                format_func=lambda i: f"Object {i + 1}: {label_dict[i]}",
                key="selected_subject"
            )

            bbox = bbox_list[selected_index]["bbox"]

        # Crop only when a valid subject has been identified
        cropped, applied_padding = crop_single_object(
            img,
            bbox,
            input_w,
            input_h,
            padding
        )

        # Report padding limitations
        if applied_padding is not None and applied_padding != -1:
            if applied_padding < padding:
                st.info(
                    f"⚠️ Requested padding ({padding:.2f}) could not "
                    f"be achieved. Maximum possible padding "
                    f"({applied_padding:.2f}) was applied."
                )

        # Display the result or report a cropping failure
        if applied_padding == -1 or cropped is None:
            st.warning(
                "Could not find a valid crop with the selected "
                "aspect ratio and padding. Try different settings."
            )

        else:
            st.image(
                cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB),
                caption="Cropped",
                use_container_width=True
            )

            # Download button
            success, buffer = cv2.imencode(".jpg", cropped)

            if success:
                st.download_button(
                    "Download cropped image",
                    buffer.tobytes(),
                    file_name="cropped.jpg",
                    mime="image/jpeg"
                )
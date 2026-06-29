# 📸 Smart Photo Cropper

Automatically crop photos around a single person while preserving a user-defined aspect ratio.

The application uses **YOLOv8** to detect the person in the image and then computes the **largest possible crop** that:

* Maintains the requested aspect ratio
* Keeps the subject centred
* Ensures the entire subject remains visible
* Applies as much padding as possible without exceeding image boundaries

A simple **Streamlit** interface allows users to upload an image, choose an aspect ratio and padding, preview the result, and download the cropped image.

---

## Features

* 🎯 Automatic person detection using YOLOv8
* 📐 Supports any aspect ratio (e.g. 1:1, 4:5, 9:16, 16:9, custom)
* 👤 Keeps the detected subject centred
* 🖼️ Preserves the requested aspect ratio exactly
* ➕ User-adjustable padding around the subject
* 🔄 Automatically reduces padding when the requested value is impossible while keeping the subject visible
* 📥 Download the cropped image directly from the web app

---

## Workflow

```
Original Image
        ↓
YOLOv8 detects the person
        ↓
Compute the largest valid crop
        ↓
Maintain aspect ratio
        ↓
Return the cropped image
```

---

## Example

Original:

<img src="https://github.com/sumantchopdata/photo_cropper/blob/main/photos/man_on_beach.png" alt="Original" width="400">

Cropped to 16:9 ratio:

<img src="https://github.com/sumantchopdata/photo_cropper/blob/main/photos/beach_16_9_0.66.jpg" alt="Cropped to 16:9 ratio" width="400">

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/photo_cropper.git
cd photo_cropper
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Running the App

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open automatically in your browser at:

```
http://localhost:8501\
```

Alternatively, the browser version can be accessed at:

```
https://smart-photo-cropper.streamlit.app/
```


## Project Structure

```
photo_cropper/
│
├── app.py                 # Streamlit interface
├── utilities.py           # Person detection and cropping algorithm
├── requirements.txt       # the requirements to run locally
├── yolov8n.pt             # YOLO V8 model
└── README.md              # README file
└── photos/                # sample stock photos to get started
```

---

## How It Works

1. Upload an image.
2. YOLOv8 detects the person and returns a bounding box.
3. The algorithm expands the bounding box according to the requested padding.
4. If the requested padding is not feasible, it automatically finds the maximum possible padding.
5. A crop with the requested aspect ratio is computed while ensuring:

   * the subject remains fully visible,
   * the subject stays centred,
   * the crop lies entirely within the image.
6. The cropped image is displayed and can be downloaded.

---

## Supported Aspect Ratios

All positive integer ratios are supported. Examples include:

* 1:1
* 3:2
* 4:5
* 9:16
* 16:9

Equivalent ratios are simplified automatically (e.g. `100:100` becomes `1:1`, `6:4` becomes `3:2`).

---

## Limitations

* Designed for images containing **one primary person**.
* Currently optimised for centred subject composition.
* Multiple-person support is not yet implemented.

---

## Future Improvements

* Rule-of-thirds composition
* Face-aware cropping
* Multi-person support
* Presets for Instagram, LinkedIn, YouTube, and other platforms
* Batch image processing
* API endpoint using FastAPI

---

## Technologies Used

* Python
* Streamlit
* YOLOv8 (Ultralytics)
* OpenCV
* Pillow

---

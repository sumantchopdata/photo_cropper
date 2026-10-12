# 📸 Smart Photo Cropper

Automatically crop photos around a detected subject while preserving a user-defined aspect ratio.

The application uses **YOLOv8** to detect objects in an image, then computes a crop that:

- Maintains the requested aspect ratio
- Keeps the selected subject centred
- Keeps the detected subject fully visible
- Applies as much padding as possible without exceeding the image boundaries

A **Streamlit** interface lets users upload an image, choose an aspect ratio and padding, select a main subject when multiple objects are detected, preview the result, and download the cropped image.

## Features

- **Multi-class object detection** using a pretrained YOLOv8 model
- **Multiple-object selection:** when several objects are detected, the app displays numbered bounding boxes and class labels on the original image
- **Configurable object limit:** choose how many of the largest detected objects to display for selection
- **Automatic cropping:** crops immediately when exactly one object is detected
- **User-selected subject:** when multiple objects are detected, crops around the selected object
- **No-detection handling:** warns the user and displays the original image without generating a crop when no objects are detected
- Supports the aspect ratios offered in the app, including 1:1, 2:3, 3:2, 4:3, 3:4, 4:5, 5:4, 9:16, and 16:9
- User-adjustable padding around the subject
- Automatically reduces padding when the requested amount is not feasible while keeping the subject visible
- Download the cropped image directly from the web app

## Workflow

```text
Upload original image
        ↓
YOLOv8 detects objects and their classes
        ↓
No objects? ── Yes ──> Display original image and show a warning
        │
        No
        ↓
One object? ── Yes ──> Crop automatically
        │
        No
        ↓
Display numbered bounding boxes and class labels
        ↓
User selects the main subject
        ↓
Compute a valid crop for the selected aspect ratio and padding
        ↓
Preview and download the cropped image
```

## Example

**Original:**

<img src="https://github.com/sumantchopdata/photo_cropper/blob/main/photos/man_on_beach.png" alt="Original" width="400">

**Cropped to 16:9 ratio:**

<img src="https://github.com/sumantchopdata/photo_cropper/blob/main/photos/beach_16_9_0.66.jpg" alt="Cropped to 16:9 ratio" width="400">

## Installation

Clone the repository:

```bash
git clone https://github.com/sumantchopdata/photo_cropper.git
cd photo_cropper
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the App

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the local app in your browser at:

```text
http://localhost:8501
```

Alternatively, try the deployed version:

https://smart-photo-cropper.streamlit.app/

## Project Structure

```text
photo_cropper/
├── app.py                 # Streamlit interface and user interaction
├── utilities.py           # Object detection, bounding-box annotation, and cropping logic
├── requirements.txt       # Python dependencies
├── yolov8n.pt             # YOLOv8 model weights
├── photos/                # Sample images
└── README.md              # Project documentation
```

## How It Works

1. Upload an image.
2. YOLOv8 detects objects supported by the model and returns their bounding boxes, class names, and confidence scores.
3. The app sorts detections by bounding-box area and limits the list to the user-selected maximum number of objects.
4. If one object remains, the app crops around it automatically.
5. If multiple objects remain, the app displays numbered bounding boxes and class labels on the original image. The user selects the intended main subject.
6. The algorithm expands the selected bounding box according to the requested padding.
7. If the requested padding is not feasible, the algorithm uses the maximum possible padding.
8. The crop is computed with the requested aspect ratio while keeping the subject visible and the crop within the image boundaries.
9. The cropped image is displayed and can be downloaded.

If no objects are detected, the app displays the original image and a warning instead of treating the entire image as a detected subject.

## Supported Aspect Ratios

The current Streamlit interface offers these aspect ratios:

- 1:1
- 2:3
- 3:2
- 4:3
- 3:4
- 4:5
- 5:4
- 9:16
- 16:9

## Limitations

- The app relies on the classes supported by the selected pretrained YOLOv8 model. It cannot reliably detect every possible subject or understand the photographer's intent.
- The app prioritises detections with larger bounding-box areas, which does not always mean they are the most relevant subjects.
- The maximum-object setting limits the available choices; a desired subject may be excluded if it is not among the largest detections.
- Cropping is centred on the selected object's bounding box. Scene-level composition, such as preserving a sunset or balancing several subjects, is not yet considered.
- A detection can be correct while the resulting crop may still be visually undesirable.

## Future Improvements

- Rule-of-thirds composition
- Face-aware cropping
- Better subject prioritisation beyond bounding-box area
- Scene-aware fallback cropping when no relevant object is detected
- Presets for Instagram, LinkedIn, YouTube, and other platforms
- Batch image processing
- API endpoint using FastAPI
- A labelled evaluation set to measure crop acceptability and analyse failure cases

## Technologies Used

- Python
- Streamlit
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- Pillow
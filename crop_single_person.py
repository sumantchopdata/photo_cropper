#%%
# detect a person in an image with one person
# and crop a square image around them with user defined padding

import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
#%%
def square_crop(img, padding=0):
    '''
    Detect a person in an image with one person and
    crop a square image around them with user defined padding.
    
    Inputs:
    img: a CV2 object read directly from the image
        e.g. img = cv2.imread("person.jpg")
    padding: a variable to expand the image beyond the bounding box of the
    detected person, as a fraction of the bounding box size. Default value is 0.

    Output: array of the square cropped image with the selected padding.
    Saved to the same folder in jpg format.
    '''

    # Run YOLO detection and filter for person class (cls == 0)
    result = model(img)[0]
    persons = [box for box in result.boxes if int(box.cls) == 0]

    # Pick the largest detected person by bounding box area
    largest = max(
        persons,
        key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1])
    )
    x1, y1, x2, y2 = map(int, largest.xyxy[0])

    # Image dimensions for boundary checking
    h, w = img.shape[:2]

    # Center of the bounding box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # Square side is the longer bounding box dimension, expanded by padding
    side = max(x2 - x1, y2 - y1)
    pad_px = int(side * padding)
    side = side + 2 * pad_px
    # If the square is larger than the image, clamp to image size
    side = min(side, w, h)

    # Initial square crop coordinates centered on the person
    crop_x1 = cx - side // 2
    crop_y1 = cy - side // 2
    crop_x2 = crop_x1 + side
    crop_y2 = crop_y1 + side

    # Shift crop window if it extends beyond image boundaries
    if crop_x1 < 0:
        crop_x2 -= crop_x1  # shift right
        crop_x1 = 0
    if crop_y1 < 0:
        crop_y2 -= crop_y1  # shift down
        crop_y1 = 0
    if crop_x2 > w:
        crop_x1 -= (crop_x2 - w)  # shift left
        crop_x2 = w
    if crop_y2 > h:
        crop_y1 -= (crop_y2 - h)  # shift up
        crop_y2 = h

    # Final clamp in case the square is larger than the image
    crop_x1 = max(crop_x1, 0)
    crop_y1 = max(crop_y1, 0)

    # Crop, save, and return
    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
    cv2.imwrite("cropped_person.jpg", cropped)
    return cropped
#%%
img = cv2.imread("person.jpg")
print(img.shape)
cropped = square_crop(img, padding=0.7)
print(cropped.shape)
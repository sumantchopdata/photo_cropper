#%%
# detect a person in an image with one person
# and crop a square image around them with user defined padding

import numpy as np
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
#%%
def largest_bounding_box(img):
    # Run YOLO detection and filter for person class (cls == 0)
    result = model(img)[0]
    persons = [box for box in result.boxes if int(box.cls) == 0]

    # Pick the largest detected person by bounding box area
    largest = max(
        persons,
        key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1])
    )
    x1, y1, x2, y2 = map(int, largest.xyxy[0])
    return x1, y1, x2, y2

# calculate the largest aspect ratio possible for an image
def largest_centered_aspect_ratio(bbox, img_width, img_height):
    """
    Parameters
    ----------
    bbox : tuple (x1, y1, x2, y2) coordinates of the person's bounding box
    img_width and img_height : int

    Returns
    -------
    aspect_ratio : float
        Width / Height of the largest possible crop centered on the subject.
    crop_dims : tuple
        (crop_width, crop_height)
    """

    x1, y1, x2, y2 = bbox

    # Subject center
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Maximum symmetric expansion possible
    half_crop_width = min(cx, img_width - cx)
    half_crop_height = min(cy, img_height - cy)

    crop_width = 2 * half_crop_width
    crop_height = 2 * half_crop_height

    aspect_ratio = crop_width / crop_height

    return aspect_ratio

#%%
def square_crop(x1, y1, x2, y2, padding=0):
    '''
    crop a square image around them with user defined padding.
    
    Inputs:
    img: a CV2 object read directly from the image
        e.g. img = cv2.imread("person.jpg")
    padding: a variable to expand the image beyond the bounding box of the
    detected person, as a fraction of the bounding box size. Default value is 0.

    Output: array of the square cropped image with the selected padding.
    Saved to the same folder in jpg format.
    '''

    # Image dimensions for boundary checking
    h, w = img.shape[:2]

    # Center of the bounding box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # Square side is the longer bounding box dimension, expanded by padding
    side = max(x2 - x1, y2 - y1)
    pad_px = int(side * padding)
    side += 2 * pad_px

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
img = cv2.imread("man.jpg")
print(img.shape)
h, w = img.shape[:2]

bbox = largest_bounding_box(img)
(x1, y1, x2, y2) = bbox
print('x1, y1, x2, y2 = ', bbox)

# crop the photo according to user defined aspect ratio r = w:h
input_w, input_h = 2, 1
padding = 0

largest_ratio = largest_centered_aspect_ratio(bbox, w, h)
print(largest_ratio)

if input_w/input_h > largest_ratio:
    print(f'The image cannot be cropped in the aspect ratio {input_w}:{input_h}, the largest aspect ratio possible with the subject in the center is {round(largest_ratio, 2)}:1.')

else:
    # if input_w == 1 or input_h == 1:
    #     w = input_w*2
    #     h = input_h*2

    # Center of the bounding box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    print('cx, cy = ', cx, cy)

    # First we make a square around the person
    side = max(x2 - x1, y2 - y1)
    pad_px = int(side * padding)
    side += 2 * pad_px

    # If the square is larger than the image, clamp to image size
    side = min(side, w, h)
    print('side = ', side)

    # Initial crop coordinates centered on the person

    crop_x1 = cx - side // (h/2)
    crop_y1 = cy - side // (w/2)
    crop_x2 = cx + side // (h/2)
    crop_y2 = cy + side // (w/2)

    print('Initial: crop_x1, crop_x2, crop_y1, crop_y2 = ', crop_x1, crop_x2, crop_y1, crop_y2)

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

    # turn into integers
    crop_x1, crop_x2 = int(crop_x1), int(crop_x2) 
    crop_y1, crop_y2 = int(crop_y1), int(crop_y2)
    print('crop_x1, crop_x2, crop_y1, crop_y2 = ', crop_x1, crop_x2, crop_y1, crop_y2)

    # debug
    final_width = crop_x2 - crop_x1
    final_height = crop_y2 - crop_y1
    # aspect ratio is width by height
    print(final_width/final_height, input_w/input_h)
    print('Is close:', np.isclose(final_width/final_height, input_w/input_h,
                                  rtol=1e-02, atol=1e-02))

    # Crop, save, and return
    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
    print(cropped.shape)
    # cv2.imwrite(f"cropped_person_{input_w}_{input_h}.jpg", cropped)
# %%

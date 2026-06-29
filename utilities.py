#%%
# detect a person in an image with one person
# and crop a image around them with user defined padding

import cv2
from math import gcd
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def largest_bounding_box(img):
    if img is None:
        raise("Error: Image not found or unable to load.")
    
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

def largest_centered_aspect_ratio(bbox, img_width, img_height):
    """
    Calculate the largest possible aspect ratio possible for an image.

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

def crop_single_person(img, bbox, input_w, input_h, padding=0, filename='cropped_person'):
    """
    Crop an image around a single person while maintaining a specified
    aspect ratio and keeping the subject centered.
    A user defined but maximum possible padding is applied around the person.

    Parameters
    ----------
    img : np.ndarray
        Image read by cv2.imread().
    bbox : tuple
        (x1, y1, x2, y2) coordinates of the person.
    input_w : int
        Desired width ratio.
    input_h : int
        Desired height ratio.
    padding : float
        Padding as a fraction of the larger bbox dimension.

    Returns
    -------
    cropped : np.ndarray
        Cropped image.
    """
    if img is None:
        raise("Error: Image not found or unable to load.")
    
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    g = gcd(input_w, input_h)
    input_w //= g
    input_h //= g

    target_ratio = input_w / input_h

    current_padding = padding

    while current_padding >= 0:

        # Padded bounding box dimensions
        person_w = x2 - x1
        person_h = y2 - y1

        pad_x = current_padding * person_w
        pad_y = current_padding * person_h

        px1 = x1 - pad_x
        px2 = x2 + pad_x
        py1 = y1 - pad_y
        py2 = y2 + pad_y
        
        box_w = px2 - px1
        box_h = py2 - py1

        if box_w / box_h < target_ratio:
            # extra width is required
            crop_h = box_h
            crop_w = crop_h * target_ratio
        else:
            # extra height is required
            crop_w = box_w
            crop_h = crop_w / target_ratio

        # Expand to desired aspect ratio
        cx = (px1 + px2)/2
        cy = (py1 + py2)/2

        crop_x1 = cx - crop_w/2
        crop_x2 = cx + crop_w/2
        crop_y1 = cy - crop_h/2
        crop_y2 = cy + crop_h/2

        # # Shift crop window inside image boundaries
        if crop_x1 < 0:
            crop_x2 -= crop_x1
            crop_x1 = 0

        if crop_x2 > w:
            crop_x1 -= (crop_x2 - w)
            crop_x2 = w

        if crop_y1 < 0:
            crop_y2 -= crop_y1
            crop_y1 = 0

        if crop_y2 > h:
            crop_y1 -= (crop_y2 - h)
            crop_y2 = h

        # Check whether crop fits inside image
        if (crop_x1 >= 0 and crop_y1 >= 0 and crop_x2 <= w and crop_y2 <= h):
            break

        # Reduce padding by 1%
        current_padding -= 0.01

    else:
        raise ValueError("Could not find a valid crop.")

    # Fix off-by-one errors caused by rounding
    final_w = crop_x2 - crop_x1
    final_h = crop_y2 - crop_y1

    desired_w = round(final_h * target_ratio)

    if desired_w <= w:
        crop_x2 = crop_x1 + desired_w
    else:
        desired_h = round(final_w / target_ratio)
        crop_y2 = crop_y1 + desired_h

    crop_x1, crop_y1 = int(crop_x1), int(crop_y1)
    crop_x2, crop_y2 = int(crop_x2), int(crop_y2)
    print(crop_x1, crop_y1, crop_x2, crop_y2)

    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]

    print(f"Crop size = {crop_x2-crop_x1} * {crop_y2-crop_y1}")

    if g > 1:
        print(f"Using equivalent aspect ratio {input_w}:{input_h}.")
    
    print(f"Aspect ratio = {(crop_x2-crop_x1)/(crop_y2-crop_y1):.3f}")
    print(f"Requested padding : {padding:.2f}")

    if current_padding != padding:
        print(f"Applied padding   : {current_padding:.2f} (maximum possible)")
    else:
        print(f"Applied padding   : {current_padding:.2f}")

    cv2.imwrite(f"{filename}_{input_w}_{input_h}_{current_padding:.2f}.jpg",
                cropped)
    return cropped
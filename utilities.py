#%%
# detect a object in an image and crop a image around them
# with user defined padding

import cv2
from math import gcd
from ultralytics import YOLO

model = YOLO("yolov8n.pt")


def detect_bounding_boxes(img, max_number_of_objects=5):
    """
    Detect all objects and return their bounding boxes, class names,
    and confidence scores, sorted by area.
    """
    if img is None:
        print("Error: Image not found or unable to load.")
        return []

    result = model(img)[0]
    objects = []

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])

        objects.append({
            "bbox": (x1, y1, x2, y2),
            "class_name": model.names[class_id],
            "confidence": float(box.conf[0])
        })

    if len(objects) > max_number_of_objects:
        objects = objects[:max_number_of_objects]

    # Sort by bounding-box area, largest first
    objects.sort(
        key=lambda obj: (
            (obj["bbox"][2] - obj["bbox"][0]) *
            (obj["bbox"][3] - obj["bbox"][1])
        ),
        reverse=True
    )

    return objects


def largest_centered_aspect_ratio(bbox, img_width, img_height):
    """
    Calculate the largest possible aspect ratio possible for a given bounding box in an image.

    Parameters
    ----------
    bbox : tuple (x1, y1, x2, y2) coordinates of the object's bounding box
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


def crop_single_object(img, bbox, input_w, input_h, padding=0):
    """
    Crop an image around a single object while maintaining a specified
    aspect ratio and keeping the subject centered.
    A user defined but maximum possible padding is applied around the object.

    Parameters
    ----------
    img : np.ndarray
        Image read by cv2.imread().
    bbox : tuple
        (x1, y1, x2, y2) coordinates of the object.
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
        print("Error: Image not found or unable to load.")
        return None, None
    
    h, w = img.shape[:2]
    if bbox is not None:
        x1, y1, x2, y2 = bbox
    else:
        print('No object found in the image.')
        return None, None

    g = gcd(input_w, input_h)
    input_w //= g
    input_h //= g

    target_ratio = input_w / input_h

    current_padding = padding

    while current_padding >= 0:

        # Padded bounding box dimensions
        object_w = x2 - x1
        object_h = y2 - y1

        pad_x = current_padding * object_w
        pad_y = current_padding * object_h

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
        print("Error: Could not find a valid crop with the chosen aspect ratio and padding.")
        return None, -1

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

    return cropped, current_padding


def bbox_list_to_image(bbox_list, img):
    """
    Convert a list of bounding boxes to a list of images for display in a selectbox.
    To be used when user is asked to select one of multiple detected objects as the main subject.

    Parameters
    ----------
    bbox_list : list of tuples
        List of bounding boxes (x1, y1, x2, y2).
    img : np.ndarray
        Original image.

    Returns
    -------
    images : list of np.ndarray
        List of cropped images corresponding to the bounding boxes.
    """
    images = []
    for bbox in bbox_list:
        x1, y1, x2, y2 = bbox
        cropped_img = img[y1:y2, x1:x2]
        images.append(cropped_img)
    return images


def draw_numbered_bounding_boxes(img, bbox_list):
    """
    Draw numbered bounding boxes with class names.
    """
    annotated_img = img.copy()

    label_dict = {}
    for i, obj in enumerate(bbox_list):
        x1, y1, x2, y2 = obj["bbox"]
        class_name = obj["class_name"]

        # Label: object number and detected class
        label = f"{i + 1}: {class_name}"

        # Draw bounding box
        cv2.rectangle(
            annotated_img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Calculate label dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # Position label above the box when possible
        label_y = y1 - 8

        # Shift below the box if not enough space above
        if label_y - text_h - baseline < 0:
            label_y = y1 + text_h + baseline + 8 

        # Draw filled background behind label
        cv2.rectangle(
            annotated_img,
            (x1, label_y - text_h - baseline),
            (x1 + text_w, label_y + baseline),
            (0, 255, 0),
            -1
        )

        # Draw dark text
        cv2.putText(
            annotated_img,
            label,
            (x1, label_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )

        label_dict[i] = class_name.title()

    return annotated_img, label_dict
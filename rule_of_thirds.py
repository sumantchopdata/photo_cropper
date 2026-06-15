#%%
# detect a person in an image with one person
# and crop a image around them according to rule of thirds.

import cv2
from ultralytics import YOLO
from crop_single_person import largest_bounding_box
model = YOLO("yolov8n.pt")


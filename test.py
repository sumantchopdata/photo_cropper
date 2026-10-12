#%%
# detect a object in an image with one object
# and crop a image around them.

import cv2
import matplotlib
from utilities import detect_bounding_boxes, crop_single_object, draw_numbered_bounding_boxes

img = cv2.imread('photos/friends-group.jpg')
print('img.shape', img.shape)

bbox_list = detect_bounding_boxes(img)

if len(bbox_list) > 1:
    print('More than one object detected, please select one of them.')
    annotated_img, _ = draw_numbered_bounding_boxes(img, bbox_list)
    
    for i, dictionary in enumerate(bbox_list):
        print(f'object {i}: {dictionary["class_name"]}')

    selected_index = int(input('Select the main subject (0, 1, 2, ...): '))
    bbox = bbox_list[selected_index]["bbox"]

if len(bbox_list) == 1:
    print(f'object {i}: {bbox_list[0]["class_name"]}, bbox: {bbox_list[0]["bbox"]}')

if len(bbox_list) == 0:
    print('no object detected, please try another image')


crop_width = 9
crop_height = 16
padding = 0

cropped, _ = crop_single_object(img, bbox, crop_width, crop_height, padding=padding)
print('cropped.shape', cropped.shape)
#%%
print('Displaying cropped image...')
# print(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
matplotlib.pyplot.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
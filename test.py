#%%
# detect a person in an image with one person
# and crop a image around them.

import cv2
from utilities import largest_bounding_box, crop_single_person

img = cv2.imread('photos/slides.png')
print('img.shape', img.shape)
#%%
bbox = largest_bounding_box(img)
print('bbox', bbox)
#%%
cropped = crop_single_person(img, bbox, 16, 9, 0.7, 'photos/beach')
print('cropped.shape', cropped.shape)
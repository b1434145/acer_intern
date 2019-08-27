#!/usr/bin/env python

import cv2
from core.detectors import CornerNet_Squeeze_traffic_light
from core.vis_utils import draw_bboxes
from core.vis_utils_original import draw_bboxes_original

detector = CornerNet_Squeeze_traffic_light()

image    = cv2.imread("demo/test4.jpg")
#print('a',image)
image = cv2.resize(image,(800,500))
bboxes = detector(image)
# print(bboxes)

image  = draw_bboxes_original(image, bboxes,thresh=0.4)

cv2.imwrite("demo_out2/demo_out2222.jpg", image)

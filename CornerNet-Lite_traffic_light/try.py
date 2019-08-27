import cv2
import time
from core.detectors import CornerNet_Squeeze
from core.detectors import CornerNet_Saccade
from core.vis_utils import draw_bboxes

detector = CornerNet_Squeeze()
#detector = CornerNet_Saccade()

video = cv2.VideoCapture("demo/test.mp4")

fps = video.get(cv2.CAP_PROP_FPS)
print(fps)

ret, frame = video.read()


while ret:
  start = time.time()
  frame = cv2.resize(frame, (640,360))
  bboxes = detector(frame)
  frame = draw_bboxes(frame, bboxes)
  cv2.waitKey(1)
  cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
  cv2.imshow('frame', frame)
  ret, frame = video.read()
  end = time.time()
  print(1/(end-start))


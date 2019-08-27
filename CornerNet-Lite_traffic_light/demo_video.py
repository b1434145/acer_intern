import cv2
import time
import torch
from core.detectors import CornerNet_Squeeze_traffic_light
from core.vis_utils import draw_bboxes
from core.vis_utils_original import draw_bboxes_original
from core.vis_utils_demo import draw_bboxes_demo
from core.vis_utils_demo import CapsuleNet

model = CapsuleNet(input_size=[3, 28, 28], classes=6, routings=3)
model.cuda()
model.load_state_dict(torch.load('/home/arno/Downloads/epoch8.pkl'))
model.eval()

CUDA = torch.cuda.is_available()
print(CUDA)

detector = CornerNet_Squeeze_traffic_light()

video = cv2.VideoCapture("arno.avi")

# fps = video.get(cv2.CAP_PROP_FPS)
# print(fps)
count = 0

while True:
    start = time.time()
    if video.grab():
        _, frame = video.retrieve()
        # frame = cv2.resize(frame, (511,511))
        # frame = cv2.resize(frame, (640,360))
        frame = cv2.resize(frame, (960,540))
        count = count + 1
        if (count % 1 == 0):
            bboxes = detector(frame)
            # frame = draw_bboxes_original(frame, bboxes, thresh=0.4)
            frame = draw_bboxes_demo(model, frame,bboxes,thresh=0.4)
            cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
            cv2.imshow('frame', frame)

        # bboxes = detector(frame)
        # frame = draw_bboxes_demo(model, frame,bboxes,thresh=0.45)
        # cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
        # cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        continue
    end = time.time()
    print("fps:",(1/(end-start)))
        


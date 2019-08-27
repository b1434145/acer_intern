
import cv2
import os
import torch
import time
from core.detectors import CornerNet_Squeeze_traffic_light
from core.vis_utils import draw_bboxes
from core.vis_utils_original import draw_bboxes_original
from core.vis_utils_demo import draw_bboxes_demo
from core.vis_utils_demo import draw_bboxes_demo
from core.vis_utils_demo import CapsuleNet

model = CapsuleNet(input_size=[3, 28, 28], classes=6, routings=3)
model.cuda()
model.load_state_dict(torch.load('/home/arno/Downloads/epoch8.pkl'))
model.eval()

count = 0
detector = CornerNet_Squeeze_traffic_light()

# test_image = []
demo_image = []
# test_image = os.listdir("data/coco/images/temp_img")
demo_image = os.listdir("data/coco/images/demo")
demo_image.sort()
# test_image.sort()
# aaa = 0
imgs = []
print('Reading...')
for i in range(len(demo_image)):
    #image = cv2.imread('data/coco/images/temp_img/'+str(test_image[i]))
     imgs.append(cv2.imread('data/coco/images/demo/'+str(demo_image[i])))

print('Finish read images')
fps = 15.0
pathOut = 'video.avi'
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, (960,540))

for i, image in enumerate(imgs):
    #image = cv2.imread('data/coco/images/temp_img/'+str(test_image[i]))
    start = time.time()
    image = cv2.resize(image, (960,540))
    #name = str(test_image[i]).split('j',1)[0]
    #print(name)
    #image = draw_bboxes_original(image,bboxes,thresh=0.40)
    # image = draw_bboxes(image,bboxes,name,thresh=0.4)
    count = count + 1
    if (count % 1 == 0):
        bboxes = detector(image)
        image = draw_bboxes_demo(model, image,bboxes,thresh=0.45)
        cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
        cv2.imshow('frame', image)
        out.write(image)
    # image = draw_bboxes_demo(image,bboxes,thresh=0.4)

    #cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    #cv2.imshow('frame', image)
    end = time.time()
    print("fps:",(1/(end-start)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
out.release()

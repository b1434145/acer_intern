#!/usr/bin/env python

import cv2
from argparse import ArgumentParser
from time import time

from core.detectors import CornerNet_Saccade, CornerNet_Squeeze
from core.vis_utils import draw_bboxes

def main(args):

    cam = cv2.VideoCapture(args.device)

    if args.codec == 'YUY2':
        cam.set(cv2.CAP_PROP_FOURCC, 844715353.0)
    elif args.codec == 'MJPG':
        cam.set(cv2.CAP_PROP_FOURCC, 0x47504A4D)
    else:
        print('use default video codec.')

    if args.resolution:
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, args.resolution[0])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT,args.resolution[1])

    detector = CornerNet_Squeeze(model_name=args.model) if args.model else CornerNet_Squeeze()

    frame_count = 0
    init_time = time()
    tic = time()

    try:
        while True:
            # Capture frame-by-frame
            if cam.grab():
                _, frame = cam.retrieve()
                bboxes = detector(frame)
                frame  = draw_bboxes(frame, bboxes)
                toc = time()
                frame_count += 1
            else:
                continue

            # Calculate fps
            if toc - init_time > 3:
                fps = frame_count / (toc - tic)
                print('{:.2f}: {} x {} @ {:5.1f}'.format(time(), frame.shape[1], frame.shape[0], fps))

                if toc -tic > 3:
                    tic = time()
                    frame_count = 0

            # Show the resulting frame
            if args.visual:
                frame = cv2.resize(frame, (0, 0), fx=args.scale, fy=args.scale)
                cv2.imshow('/dev/video{}'.format(args.device), frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt')
        pass

    # When everything done, release the capture
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--device', help='device number: /dev/video#', type=int, default=0)
    parser.add_argument('-c', '--codec', help='video codec: MJPG/YUY2')
    parser.add_argument('-v', '--visual', action='store_true', dest='visual', help='Show image frame')
    parser.add_argument('-r', '--resolution', nargs='+', type=float, help='resolution: w, h')
    parser.add_argument('-s', '--scale', type=float, help='output frame scale: [0.25]', default=0.25)
    parser.add_argument('-m', '--model', type=str, help='model name')

    args = parser.parse_args()

    main(args)

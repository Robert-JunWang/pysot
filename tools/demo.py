from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg

from pysot.core.config import alex_cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
args = parser.parse_args()


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame

from time import sleep

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main():


    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')


    # create model
    model = ModelBuilder(cfg)

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    # siamrpn_tracker = tracker

    # # # load config
    # alex_cfg.merge_from_file('experiments/siamrpn_alex_dwxcorr/config.yaml')
    # alex_cfg.CUDA = torch.cuda.is_available()

    # # # create model
    # alex = ModelBuilder(alex_cfg)
    # # load model
    # alex.load_state_dict(torch.load('experiments/siamrpn_alex_dwxcorr/model.pth',
    #     map_location=lambda storage, loc: storage.cpu()))
    # alex.eval().to(device)

    # # build tracker
    # siamrpn_tracker = build_tracker(alex)


    kcf_tracker = cv2.TrackerKCF_create()
    csrt_tracker = cv2.TrackerCSRT_create()

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    # cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)

    default_rois={'HUMAN_005_1920_30':(604, 723, 54, 91),
                'HUMAN_003_30_2':(171, 257, 10, 29),
                'HUMAN_003_30_3':(920, 151, 32, 72)}
    if video_name in default_rois:
        init_rect = default_rois[video_name]
    else:
        init_rect = None

    colors=[(0,0,255), (255,0,0), (0,255,0),(0,0,0)]
    # kcf_color={'color':(0,0,0), 'pos':(80,40)}

    avg_fps = AverageMeter()

    for frame in get_frames(args.video_name):


        if first_frame:
            if init_rect is None:
                print('init rect')

                try:
                    init_rect = cv2.selectROI(video_name, frame, False, False)
                except:
                    exit()
            print('frame size:',frame.shape)
            print('ROI:', init_rect)
            tracker.init(frame, init_rect)
            # siamrpn_tracker.init(frame, init_rect)
            # kcf_tracker.init(frame, init_rect)
            # csrt_tracker.init(frame, init_rect)
            first_frame = False

            # cv2.imshow(video_name, frame)
            # cv2.waitKey(40)
            # sleep(10)
        else:
                    # Start timer
            timer = cv2.getTickCount()

            outputs = tracker.track(frame)
            # outputs2 = siamrpn_tracker.track(frame)

            # ok3, bbox3 = kcf_tracker.update(frame)
            # ok4, bbox4 = csrt_tracker.update(frame)

            # ok3, bbox3 = csrt_tracker.update(frame)

            fps = cv2.getTickFrequency()  / (cv2.getTickCount() - timer)
            avg_fps.update(fps)

            # cv2.rectangle(frame, (80,40), (160,55), colors[0], cv2.FILLED)
            # cv2.putText(frame, "KCF", (85,55), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255),1)

            # cv2.rectangle(frame, (80,60), (160,75), colors[1], cv2.FILLED)
            # cv2.putText(frame, "CSRT", (85,75), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255),1)

            # cv2.rectangle(frame, (80,80), (160,95), colors[2], cv2.FILLED)
            # cv2.putText(frame, "SiamMask", (82,93), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,0,0),1)

            # cv2.rectangle(frame, (80,100), (160,115), colors[3], cv2.FILLED)
            # cv2.putText(frame, "SiamRPN", (82,113), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255),1)

            # Display FPS on frame
            cv2.putText(frame, "FPS:{fps:.1f}, Average:{avg:.1f} ".format(fps=fps, avg=avg_fps.avg), (100,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
    
            # SiamMask
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)

            # # SiamRPN
            # bbox = list(map(int, outputs2['bbox']))
            # cv2.rectangle(frame, (bbox[0], bbox[1]),
            #                 (bbox[0]+bbox[2], bbox[1]+bbox[3]),
            #                 colors[3], 3)

            # # KCF
            # if ok3:
            #     bbox = bbox3
            #     # Tracking success
            #     p1 = (int(bbox[0]), int(bbox[1]))
            #     p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            #     cv2.rectangle(frame, p1, p2, colors[0], 2, 1)
            # else :
            #     # Tracking failure
            #     cv2.putText(frame, "failure detected", (170,55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,colors[0],2)

            # # CSRT
            # if ok4:
            #     # Tracking success
            #     bbox = bbox4
            #     p1 = (int(bbox[0]), int(bbox[1]))
            #     p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            #     cv2.rectangle(frame, p1, p2, colors[1], 2, 1)
            # else :
            #     # Tracking failure
            #     cv2.putText(frame, "failure detected", (170,75), cv2.FONT_HERSHEY_SIMPLEX, 0.5,colors[1],2)
      

            cv2.imshow(video_name, frame)
            cv2.waitKey(40)


if __name__ == '__main__':
    main()

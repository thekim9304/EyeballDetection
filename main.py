import cv2
import dlib
import numpy as np

from model.mobilenetv1 import MobileNetV1
from model.ssd import SSD, Predictor

import torch

from utils import *


def main():
    # face detector
    pth_path = 'Z:/pths/face_detection/ssd_mobilenetv1/ssd-mobilev1-face-2134_0.0192.pth'
    face_detector = face_detector_loader(pth_path)
    # landmark detector
    dat_path = 'Z:/dlib/shape_68.dat'
    land_detector = dlib.shape_predictor(dat_path)

    cap = cv2.VideoCapture(1)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow('annotated')
    cv2.createTrackbar('threshold', 'annotated', 0, 255, nothing)
    cv2.createTrackbar('land_height', 'annotated', 0, 20, nothing)

    # init value
    face_detect, land_detect = False, False
    prev_bbox, prev_land = [], []
    n_point, i, j = 0, 0, 0
    while True:
        ret, frame = cap.read()

        if ret:
            blackboard = np.full((face_size * 2, (face_size + frame_width), 3), 0, dtype=np.uint8)
            boxes, labels, probs, sec1 = get_face(face_detector, frame)
            sec2 = 0.0

            if boxes.size(0) and probs[0] > 0.5:
                box = boxes[0, :]
                label = f"Face: {probs[0]:.2f}"
                cur_bbox = add_face_region(box)
                cur_bbox, prev_bbox, face_detect = low_pass_filter(cur_bbox, prev_bbox, face_detect, mode='face')
                x1, x2, y1, y2 = cur_bbox

                '''detect landmark'''
                cur_land, sec2 = get_landmark(land_detector, frame, cur_bbox)
                land_add = cv2.getTrackbarPos('land_height', 'annotated')
                cur_land = cvt_shape_to_np(cur_land, land_add=land_add)
                cur_land, prev_land, land_detect = low_pass_filter(cur_land, prev_land, land_detect, mode='landmark')
                cur_rel_coord = cvt_land_rel(cur_land, cur_bbox)

                '''detect eyeball'''
                face = frame[y1:y2, x1:x2].copy()
                face = cv2.resize(face, (face_size, face_size))
                abs_land = (cur_rel_coord * face_size).astype(np.int)
                centers, thresh = get_eye_centers(face, cur_rel_coord)
                if 0 not in centers[0] and 0 not in centers[1]:
                    for land_i in range(27, 36):
                        face = cv2.line(face, centers[0], (abs_land[land_i][0], abs_land[land_i][1]), (255, 0, 0), 1)
                        face = cv2.line(face, centers[1], (abs_land[land_i][0], abs_land[land_i][1]), (255, 255, 0), 1)
                    l_center = np.array(centers[0]) / face.shape[0]
                    r_center = np.array(centers[1]) / face.shape[0]
                    frame = cv2.circle(frame, (int(l_center[0]*(x2-x1)+x1), int(l_center[1]*(y2-y1)+y1)), 2, (255, 255, 255), -1)
                    frame = cv2.circle(frame, (int(r_center[0] * (x2 - x1) + x1), int(r_center[1] * (y2 - y1) + y1)), 2, (255, 255, 255), -1)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 0, 255), 4)
                cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                frame = draw_land(frame, cur_land, (0, 0, 255))
                blackboard[:face_size, :face_size] = face
                blackboard[face_size:, :face_size] = thresh
            else:
                land_detect = False
                face_detect = False

            '''draw result'''
            notice_board = draw_speed((frame_height, frame_width), (sec1, sec2))
            blackboard[frame_height:, face_size:] = notice_board
            blackboard[:frame_height, face_size:] = frame
            cv2.imshow('annotated', blackboard)

            if cv2.waitKey(1) == 27:
                break
        else:
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()

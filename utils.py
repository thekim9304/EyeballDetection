import cv2
import time
import dlib
import numpy as np

from model.mobilenetv1 import MobileNetV1
from model.ssd import SSD, Predictor

import torch

# setting
face_size = 300
bbox_region = {'forehead': 35, 'chin': 0, 'add_face_width': 10}
filters = {'bbox': 15, 'landmark': 3}

left_eye = [36, 37, 38, 39, 40, 41]
right_eye = [42, 43, 44, 45, 46, 47]
eye_top = [37, 38,  43, 44]
eye_bottom = [40, 41, 46, 47]
eye_side_left = [36, 42]
eye_side_right = [39, 45]


def nothing(x):
    pass


def face_detector_loader(pth_path):
    f_detection_model = SSD(2, MobileNetV1(2), is_training=False)
    state = torch.load(pth_path)
    f_detection_model.load_state_dict(state['model_state_dict'])
    predictor = Predictor(f_detection_model, 300)

    return predictor


def landmark_detector_loader(dat_path):
    land_predictor = dlib.shape_predictor(dat_path)

    return land_predictor


def get_face(detector, frame):
    frame = cv2.resize(frame, (640, 480))
    prevTime = time.time()
    boxes, labels, probs = detector.predict(frame, 1, 0.5)
    sec = time.time() - prevTime

    return boxes, labels, probs, sec


def add_face_region(box):
    x1, x2, y1, y2 = int(box[0].item() - bbox_region['add_face_width']), int(
        box[2].item() + bbox_region['add_face_width']), int(box[1].item() + bbox_region['forehead']), int(
        box[3].item() + bbox_region['chin'])
    x1 = 0 if x1 < 0 else x1

    return [x1, x2, y1, y2]


def low_pass_filter(cur, prev, detect, mode=None):
    if mode == 'face':
        if detect:
            if abs(prev[0] - cur[0]) < filters['bbox']:
                cur[0] = prev[0]
            else:
                prev[0] = cur[0]
            if abs(prev[1] - cur[1]) < filters['bbox']:
                cur[1] = prev[1]
            else:
                prev[1] = cur[1]
            if abs(prev[2] - cur[2]) < filters['bbox']:
                cur[2] = prev[2]
            else:
                prev[2] = cur[2]
            if abs(prev[3] - cur[3]) < filters['bbox']:
                cur[3] = prev[3]
            else:
                prev[3] = cur[3]
        else:
            detect = True
            prev = cur
    elif mode == 'landmark':
        if detect:
            idx = 0
            for land, prev_land in zip(cur, prev):
                if abs(land[0] - prev_land[0]) < filters['landmark']:
                    cur[idx][0] = prev_land[0]
                else:
                    prev[idx][0] = land[0]
                if abs(land[1] - prev_land[1]) < filters['landmark']:
                    cur[idx][1] = prev_land[1]
                else:
                    prev[idx][1] = land[1]
                idx += 1
        else:
            detect = True
            prev = cur

    return cur, prev, detect


def get_landmark(detector, ori_img, bbox):
    if ori_img.shape[-1] == 3:
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)

    face_box = dlib.rectangle(left=bbox[0], top=bbox[2], right=bbox[1], bottom=bbox[3])
    prev_time = time.time()
    landmark = detector(ori_img, face_box)
    sec = time.time() - prev_time

    return landmark, sec


def cvt_shape_to_np(landmakrs, land_add=0, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        x = landmakrs.part(i).x
        y = landmakrs.part(i).y
        if i in eye_top:
            coords[i] = (x, y - land_add)
        elif i in eye_bottom:
            coords[i] = (x, y + land_add)
        elif i in eye_side_left:
            coords[i] = (x - (land_add-2), y)
        elif i in eye_side_right:
            coords[i] = (x + (land_add+2), y)
        else:
            coords[i] = (x, y)

    return coords


def cvt_land_rel(land, cur_box):
    rel_land = np.zeros((68, 2), dtype="float")

    rel_land[:, 0] = (land[:, 0] - cur_box[0]) / (cur_box[1] - cur_box[0])
    rel_land[:, 1] = (land[:, 1] - cur_box[2]) / (cur_box[3] - cur_box[2])

    return rel_land


def draw_land(img, land, color):
    for (x, y) in land:
        cv2.circle(img, (x, y), 2, color, -1)

    return img


def draw_abs_land(img, land, color, size=None):
    if size:
        img = cv2.resize(img, size)

    land[:, 0] = land[:, 0] * img.shape[1]
    land[:, 1] = land[:, 1] * img.shape[0]
    land = land.astype(np.int)

    for (x, y) in land:
        cv2.circle(img, (x, y), 2, color, -1)

    return img


def draw_speed(frame_size, sec):
    notice_board = np.full((600 - frame_size[0], frame_size[1], 3), 0, dtype=np.uint8)
    cv2.putText(notice_board, f'face detection : {sec[0] * 100:.2}ms', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2)
    cv2.putText(notice_board, f'landmark detection : {sec[1] * 100:.2}ms', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)

    return notice_board


def eye_on_mask(landmarks, mask, side):
    points = [landmarks[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask


def contouring(thresh, mid, img, right=False):
    cx, cy = 0, 0
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    try:
        cnt = max(cnts, key=cv2.contourArea)  # finding contour with #maximum area
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if right:
            cx += mid  # Adding value of mid to x coordinate of centre of #right eye to adjust for dividing into two parts
        cv2.circle(img, (cx, cy), 2, (0, 0, 255), -1)  # drawing over #eyeball with red
    except:
        pass

    return cx, cy


def get_eye_box_coord(rel_land, mode=None):
    abs_land = (rel_land * face_size).astype(np.int)

    min_x, min_y, max_x, max_y = face_size, face_size, 0, 0
    if mode == 'right':
        eye_idx = right_eye
    elif mode == 'left':
        eye_idx = left_eye
    else:
        print(f'Error : get_eye_box_coord function')
        eye_idx = []

    for idx in eye_idx:
        min_x = abs_land[idx][0] if min_x > abs_land[idx][0] else min_x
        min_y = abs_land[idx][1] if min_y > abs_land[idx][1] else min_y
        max_x = abs_land[idx][0] if max_x < abs_land[idx][0] else max_x
        max_y = abs_land[idx][1] if max_y < abs_land[idx][1] else max_y

    return min_x, min_y, max_x, max_y


def get_eye_centers(face, rel_land):
    abs_land = (rel_land * face_size).astype(np.int)
    r_min_x, r_min_y, r_max_x, r_max_y = get_eye_box_coord(rel_land, mode='right')
    l_min_x, l_min_y, l_max_x, l_max_y = get_eye_box_coord(rel_land, mode='left')

    mask = np.zeros(face.shape[:2], dtype=np.uint8)
    mask = eye_on_mask(abs_land, mask, left_eye)
    mask = eye_on_mask(abs_land, mask, right_eye)
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.dilate(mask, kernel, 5)

    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_gray[r_min_y:r_max_y, r_min_x:r_max_x] = cv2.equalizeHist(face_gray[r_min_y:r_max_y, r_min_x:r_max_x])
    face_gray[l_min_y:l_max_y, l_min_x:l_max_x] = cv2.equalizeHist(face_gray[l_min_y:l_max_y, l_min_x:l_max_x])
    face_gray = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2BGR)
    eyes = cv2.bitwise_and(face_gray, face_gray, mask=mask)
    mask = (eyes == [0, 0, 0]).all(axis=2)
    eyes[mask] = [255, 255, 255]
    eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
    threshold = cv2.getTrackbarPos('threshold', 'annotated')
    _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    thresh = cv2.medianBlur(thresh, 3)
    thresh = cv2.bitwise_not(thresh)

    mid = (abs_land[42][0] + abs_land[39][0]) // 2
    l_center = contouring(thresh[:, 0:mid], mid, face)
    r_center = contouring(thresh[:, mid:], mid, face, True)

    return (l_center, r_center), cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
#!/usr/bin/env python

import cv2
import numpy as np
from utils.detector import Detector
from utils.interpolator import Interpolator
from utils.imagewarp import ImageWarp
import time

src = [[70, 240], [430, 240], [0, 0], [500, 0]]
dst = [[155, 270], [165, 270], [0, 0], [320, 0]]
scan_range={'start': 0, 'stop': 240, 'steps': 10}
scan_window={'height': 8, 'max_adjust': 8}
offset = 150

DISPLAY_COLS = 1
DISPLAY_ROWS = 2

SAVE_RESULTS = False
ESTIMATE_FROM_1_LANE = False

if ESTIMATE_FROM_1_LANE == False:
    lanes =[
        {'label': 'mid', 'detections': {'start': {'x': 120, 'y': 230}, 'stop': {'x': 160, 'y': 230}}},
        {'label': 'right', 'detections': {'start': {'x': 160, 'y': 230}, 'stop': {'x': 200, 'y': 230}}}
        ]
else:
    lanes =[
        {'label': 'right', 'detections': {'start': {'x': 145, 'y': 230}, 'stop': {'x': 175, 'y': 230}}}
        ]

iw_obj = ImageWarp(img_w=500, offset=offset, src=src, dst=dst)
det_obj = Detector(scan_range=scan_range, scan_window=scan_window)
ip_obj = Interpolator(max_poly_degree=2)


def proportional_spaced_array(pts, min_value, max_value):
    delta_dist = max_value - min_value
    pts_max = pts[0]
    pts_min = pts[len(pts) - 1]
    delta_pts = pts_max - pts_min

    buffer = []
    for i in range(0, len(pts)):
        buffer.append(delta_dist * (pts_max - pts[i]) / delta_pts + min_value)

    return buffer

# lane_side = 1 se lane == right
def eq_lane(un_pts=None, lane_side=1):
    buffer = []
    dist = 115
    prev_x = None
    dist = proportional_spaced_array(un_pts[0][:, 1], 115, 250)
    for i in range(0, len(un_pts[0])):
        x = un_pts[0][i][0]
        y = un_pts[0][i][1]

        nx = x
        # if prev_x is not None:
            # temp = dist + 2 * abs(prev_x - nx)
            # if not temp > 400 and not temp < 115:
            #     dist = temp
        prev_x = nx
        nx = nx - dist[i] * lane_side

        norm_pts = np.float32(np.column_stack((nx, y)))
        buffer.append(norm_pts)

    buffer = np.array(buffer, dtype=np.float32)
    buffer = cv2.perspectiveTransform(buffer, iw_obj.wmat)

    return np.array(buffer, dtype=np.float32)



def process_image(image, res):
    # start = time.time()
    img = image.copy()
    f_img = det_obj.img_filter(img)
    f_w_img = iw_obj.img_warp(f_img,offset=True)
    try:
        img = image.copy()
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        f_img = det_obj.img_filter(img)
        f_w_img = iw_obj.img_warp(f_img,offset=True)

        color = [1,1,1]

        img_detected_points = f_w_img.copy()
        # img_line_fitted = f_w_img.copy()
        debug = f_w_img.copy()

        detected_points = {}
        for i, lane in enumerate(lanes):
            detected_points[lane['label']] = det_obj.get_lane_detections(f_w_img, start=lane['detections']['start'], stop=lane['detections']['stop'], label=lane['label'], use_RANSAC=True, debug=True)

            # Disegna l'area di valutazione delle corsie
            start_x = lane['detections']['start']['x']
            start_y = lane['detections']['start']['y']
            stop_x = lane['detections']['stop']['x']
            stop_y = lane['detections']['stop']['y']
            cv2.rectangle(debug, (start_x, start_y), (stop_x, stop_y), (255, 255, 255), 2)

        lane = None
        if detected_points['mid'].shape[0] > detected_points['right'].shape[0]:
            lane = lanes[0]
        else:
            lane = lanes[1]

        img_detected_points = det_obj.draw_detections(img_detected_points, detected_points[lane['label']])

        interpolated_points = ip_obj.interpolate([detected_points], key=lane['label'], equ_selector=False, debug=False)

        pts = np.array([interpolated_points[lane['label']]])
        cv2.polylines(debug, [np.int32(pts)], False, [255], 2)

        unwarped_pts = np.int32(iw_obj.pts_unwarp(pts))
        unwarped_pts_offset = np.add(unwarped_pts, [0, offset])

        color[i] = 255
        cv2.polylines(img, [unwarped_pts_offset], False, color, 2)

        # ESTIMATE_FROM_1_LANE
        ed_pts = np.float32(eq_lane(un_pts=unwarped_pts, lane_side=1 if lane['label'] == 'right' else -1))
        cv2.polylines(debug, [np.int32(ed_pts)], False, [255], 2)

        ed_unwarped_pts = np.int32(iw_obj.pts_unwarp(ed_pts))
        ed_unwarped_pts_offset = np.add(ed_unwarped_pts, [0, offset])

        cv2.polylines(img, [ed_unwarped_pts_offset], False, color, 2)

        cv2.rectangle(debug, (160, 220), (160, 250), (255, 255, 255), 2)
        concat_img = cv2.hconcat([img, cv2.cvtColor(debug[:, :320], cv2.COLOR_GRAY2RGB)])


        res.put(concat_img)
    except Exception as e:
        print(e)
        start_x = lane['detections']['start']['x']
        start_y = lane['detections']['start']['y']
        stop_x = lane['detections']['stop']['x']
        stop_y = lane['detections']['stop']['y']
        cv2.rectangle(f_w_img, (start_x, start_y), (stop_x, stop_y), (255, 255, 255), 2)
        concat_img = cv2.hconcat([image, cv2.cvtColor(f_w_img[:, :320], cv2.COLOR_GRAY2RGB)])
        res.put(concat_img)
        # end = time.time()
        # return img, end - start
#!/usr/bin/env python

import cv2
import numpy as np
from utils.detector import Detector
from utils.interpolator import Interpolator
from utils.imagewarp import ImageWarp
import time

src = [[70, 240], [450, 240], [0, 0], [500, 0]]
dst = [[135, 270], [150, 270], [0, 0], [320, 0]]
scan_range={'start': 0, 'stop': 240, 'steps': 10}
scan_window={'height': 8, 'max_adjust': 8}
offset = 150

DISPLAY_COLS = 1
DISPLAY_ROWS = 2

SAVE_RESULTS = False
ESTIMATE_FROM_1_LANE = False

if ESTIMATE_FROM_1_LANE == False:
    lanes =[
        # {'label': 'mid', 'detections': {'start': {'x': 115, 'y': 230}, 'stop': {'x': 145, 'y': 230}}},
        # {'label': 'right', 'detections': {'start': {'x': 145, 'y': 230}, 'stop': {'x': 175, 'y': 230}}}
        {'label': 'mid', 'detections': {'start': {'x': 110, 'y': 230}, 'stop': {'x': 145, 'y': 230}}},
        {'label': 'right', 'detections': {'start': {'x': 145, 'y': 230}, 'stop': {'x': 180, 'y': 230}}}
        ]
else:
    lanes =[
        {'label': 'right', 'detections': {'start': {'x': 145, 'y': 230}, 'stop': {'x': 175, 'y': 230}}}
        ]

iw_obj = ImageWarp(img_w=500, offset=offset, src=src, dst=dst)
det_obj = Detector(scan_range=scan_range, scan_window=scan_window)
ip_obj = Interpolator(max_poly_degree=2)

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

        for i,lane in enumerate(lanes):
            detected_points = det_obj.get_lane_detections(f_w_img,start=lane['detections']['start'],stop=lane['detections']['stop'],label=lane['label'],use_RANSAC=True, debug=True)
            img_detected_points = det_obj.draw_detections(img_detected_points,detected_points[lane['label']])

            interpolated_points = ip_obj.interpolate([detected_points], scan_range,key=lane['label'],equ_selector=False,debug=False)

            pts = np.array([interpolated_points[lane['label']]])
            # cv2.polylines(img_line_fitted, [np.int32(pts)], False, [255], 2)
            cv2.polylines(debug, [np.int32(pts)], False, [155], 2)

            unwarped_pts = np.int32(iw_obj.pts_unwarp(pts))
            unwarped_pts_offset = np.add(unwarped_pts,[0,offset])

            color[i]=255
            cv2.polylines(img, [unwarped_pts_offset], False, color, 2)

            if ESTIMATE_FROM_1_LANE == True:
                ed_pts = np.float32(ip_obj.echidistant_lane(pts,return_end_point=True, distnce=40))
                # cv2.polylines(img_line_fitted, [np.int32(ed_pts)], False, [255], 2)

                ed_unwarped_pts = np.int32(iw_obj.pts_unwarp(ed_pts))
                ed_unwarped_pts_offset = np.add(ed_unwarped_pts,[0,offset])

                cv2.polylines(img, [ed_unwarped_pts_offset], False, color, 2)


            start_x = lane['detections']['start']['x']
            start_y = lane['detections']['start']['y']
            stop_x = lane['detections']['stop']['x']
            stop_y = lane['detections']['stop']['y']
            cv2.rectangle(debug, (start_x, start_y), (stop_x, stop_y), (255, 255, 255), 2)

        # cv2.rectangle(img, (0, 0), (320, 240), (255, 255, 255), 2)
        cv2.rectangle(debug, (145, 220), (145, 250), (255, 255, 255), 2)
        concat_img = cv2.hconcat([img, cv2.cvtColor(debug[:, :320], cv2.COLOR_GRAY2RGB)])


        res.put(concat_img)
    except Exception as e:
        start_x = lane['detections']['start']['x']
        start_y = lane['detections']['start']['y']
        stop_x = lane['detections']['stop']['x']
        stop_y = lane['detections']['stop']['y']
        cv2.rectangle(f_w_img, (start_x, start_y), (stop_x, stop_y), (255, 255, 255), 2)
        concat_img = cv2.hconcat([image, cv2.cvtColor(f_w_img[:, :320], cv2.COLOR_GRAY2RGB)])
        res.put(concat_img)
        # end = time.time()
        # return img, end - start
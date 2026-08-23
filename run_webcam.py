#-*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageDraw, ImageFont
from PIL import Image,ImageTk
import matplotlib.animation as animation
import matplotlib.lines as line
from matplotlib.pylab import mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk #NavigationToolbar2TkAgg
import argparse
import logging
import time
import matplotlib.pyplot as plt
from tkinter import messagebox
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import pyaudio
import wave

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

import time
import queue
import threading
from scipy import signal



warnings.filterwarnings('ignore')

logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps = 0


def comp(a, b):
    if (a > b):
        return a > 8 * b
    else:
        return b > 8 * a


def comp1(a, b):
    if (a < b):
        return b < 4 * a
    else:
        return a < 4 * b


def charge0(humans):
    shang = (14, 15, 16, 17, 1, 0, 8, 11)  # 上半身点代号的集合
    xia = (8, 9, 10, 11, 12, 13)  # 下半身点代号的集合
    if (len(humans) > 0):
        ##创建pos数组 表示人的动作
        pos = [0] * len(humans)
        ##创建temp数组 用来记录人动作的变化
        temp = [0] * len(humans)
        for ss in range(0, len(humans)):
            temp[ss] = pos[ss]
            J89 = J910 = J1112 = J1213 = J010 = J013 = J18 = J111 = J98 = J1211 = J1113 = J810 = 10000  # 这里是两个点之间连线的夹角，在这里我们都选叫小的角度
            if ((8 in humans[ss].body_parts.keys()) == True & (9 in humans[ss].body_parts.keys()) == True):
                if (humans[ss].body_parts[9].y - humans[ss].body_parts[8].y == 0):
                    J89 = 0.001
                else:
                    J89 = abs(humans[ss].body_parts[9].x - humans[ss].body_parts[8].x) / abs(
                        humans[ss].body_parts[9].y - humans[ss].body_parts[8].y)
            if ((8 in humans[ss].body_parts.keys()) == True & (9 in humans[ss].body_parts.keys()) == True):
                if (humans[ss].body_parts[9].x - humans[ss].body_parts[8].x == 0):
                    J98 = 10000
                else:
                    J98 = abs(humans[ss].body_parts[9].y - humans[ss].body_parts[8].y) / abs(
                        humans[ss].body_parts[9].x - humans[ss].body_parts[8].x)
            if ((9 in humans[ss].body_parts.keys()) == True & (10 in humans[ss].body_parts.keys()) == True):
                if (humans[ss].body_parts[10].y - humans[ss].body_parts[9].y == 0):
                    J910 = 0.0001
                else:
                    J910 = abs(humans[ss].body_parts[10].x - humans[ss].body_parts[9].x) / abs(
                        humans[ss].body_parts[10].y - humans[ss].body_parts[9].y)
            if ((11 in humans[ss].body_parts.keys()) == True & (
                    12 in humans[ss].body_parts.keys()) == True):
                if (humans[ss].body_parts[11].y - humans[ss].body_parts[12].y == 0):
                    J1112 = 0.001
                else:
                    J1112 = abs(humans[ss].body_parts[11].x - humans[ss].body_parts[12].x) / abs(
                        humans[ss].body_parts[11].y - humans[ss].body_parts[12].y)
            if ((11 in humans[ss].body_parts.keys()) == True & (
                    12 in humans[ss].body_parts.keys()) == True):
                if (humans[ss].body_parts[11].x - humans[ss].body_parts[12].x == 0):
                    J1211 = 10000
                else:
                    J1211 = abs(humans[ss].body_parts[11].y - humans[ss].body_parts[12].y) / abs(
                        humans[ss].body_parts[11].x - humans[ss].body_parts[12].x)
            if ((12 in humans[ss].body_parts.keys()) == True & (
                    13 in humans[ss].body_parts.keys()) == True):
                if (humans[ss].body_parts[12].y - humans[ss].body_parts[13].y == 0):
                    J1213 = 0.001
                else:
                    J1213 = abs(humans[ss].body_parts[13].x - humans[ss].body_parts[12].x) / abs(
                        humans[ss].body_parts[13].y - humans[ss].body_parts[12].y)

            ##
            if ((11 in humans[ss].body_parts.keys()) == True & (
                    13 in humans[ss].body_parts.keys()) == True):
                if (humans[ss].body_parts[11].x - humans[ss].body_parts[13].x == 0):
                    J1113 = 0.001
                else:
                    J1113 = abs(humans[ss].body_parts[13].y - humans[ss].body_parts[11].y) / abs(
                        humans[ss].body_parts[13].x - humans[ss].body_parts[11].x)
            ##
            if ((0 in humans[ss].body_parts.keys()) == True & (10 in humans[ss].body_parts.keys()) == True):
                if (humans[ss].body_parts[0].x - humans[ss].body_parts[10].x == 0):
                    J010 = 0.001
                else:
                    J010 = abs(humans[ss].body_parts[0].y - humans[ss].body_parts[10].y) / abs(
                        humans[ss].body_parts[0].x - humans[ss].body_parts[10].x)
            if ((0 in humans[ss].body_parts.keys()) == True & (13 in humans[ss].body_parts.keys()) == True):
                if (humans[ss].body_parts[0].x - humans[ss].body_parts[13].x == 0):
                    J013 = 0.001
                else:
                    J013 = abs(humans[ss].body_parts[0].y - humans[ss].body_parts[13].y) / abs(
                        humans[ss].body_parts[0].x - humans[ss].body_parts[13].x)
            if ((1 in humans[ss].body_parts.keys()) == True & (8 in humans[ss].body_parts.keys()) == True):
                if (humans[ss].body_parts[1].x - humans[ss].body_parts[8].x == 0):
                    J18 = 0.001
                else:
                    J18 = abs(humans[ss].body_parts[1].y - humans[ss].body_parts[8].y) / abs(
                        humans[ss].body_parts[1].x - humans[ss].body_parts[8].x)
            if ((1 in humans[ss].body_parts.keys()) == True & (11 in humans[ss].body_parts.keys()) == True):
                if (humans[ss].body_parts[1].x - humans[ss].body_parts[11].x == 0):
                    J111 = 0.001
                else:
                    J111 = abs(humans[ss].body_parts[1].y - humans[ss].body_parts[11].y) / abs(
                        humans[ss].body_parts[1].x - humans[ss].body_parts[11].x)

            x = np.empty(shape=[0, 2], dtype=int)  # x数组为上半身点的坐标的集合
            y = np.empty(shape=[0, 2], dtype=int)  # y数组为上半身点的坐标的集合
            for ca in range(0, 18):
                if ((ca in humans[ss].body_parts.keys()) == True & (ca in shang) == True):
                    x = np.append(x, [[int(100 * round(humans[ss].body_parts[ca].x, 2)),
                                       int(100 * round(humans[ss].body_parts[ca].y, 2))]], axis=0)
                elif ((ca in humans[ss].body_parts.keys()) == True & (ca in xia) == True):
                    y = np.append(x, [[int(100 * round(humans[ss].body_parts[ca].x, 2)),
                                       int(100 * round(humans[ss].body_parts[ca].y, 2))]], axis=0)

            xx = cv2.minAreaRect(x)  # xx为上半身所有可见点的最小旋转矩形
            yy = cv2.minAreaRect(y)  # yy为上半身所有可见点的最小旋转矩形

            mkb = 0  # 判断是否已经进行过判断，若为0则继续判断，否则此次姿态识别结束
            if (((8 in humans[ss].body_parts.keys()) == True & (
                    9 in humans[ss].body_parts.keys()) == True) == True | (
                    (11 in humans[ss].body_parts.keys()) == True & (
                    12 in humans[ss].body_parts.keys()) == True) == True):
                if ((8 in humans[ss].body_parts.keys()) == True & (
                        9 in humans[ss].body_parts.keys()) == True):

                    if ((J98 < 0.5) == True):
                        if ((16 in humans[ss].body_parts.keys()) == True):
                            draw.text((int(600 * round(humans[ss].body_parts[16].x, 2)),
                                       int(450 * round(humans[ss].body_parts[16].y, 2))), "蹲坐", (0, 255, 0),
                                      font=font)
                            mkb = 1
                            pos[ss] = 1

                        elif ((17 in humans[ss].body_parts.keys()) == True):
                            draw.text((int(600 * round(humans[ss].body_parts[17].x, 2)),
                                       int(450 * round(humans[ss].body_parts[17].y, 2))), "蹲坐", (0, 255, 0),
                                      font=font)
                            mkb = 1
                            pos[ss] = 1

                elif ((11 in humans[ss].body_parts.keys()) == True & (
                        12 in humans[ss].body_parts.keys()) == True):
                    if ((J1211 < 0.5) == True):
                        if ((16 in humans[ss].body_parts.keys()) == True):
                            draw.text((int(600 * round(humans[ss].body_parts[16].x, 2)),
                                       int(450 * round(humans[ss].body_parts[16].y, 2))), "蹲坐", (0, 255, 0),
                                      font=font)
                            mkb = 1
                            pos[ss] = 1
                        elif ((17 in humans[ss].body_parts.keys()) == True):
                            draw.text((int(600 * round(humans[ss].body_parts[17].x, 2)),
                                       int(450 * round(humans[ss].body_parts[17].y, 2))), "蹲坐", (0, 255, 0),
                                      font=font)
                            mkb = 1
                            pos[ss] = 1

            if ((abs(int(xx[2])) > 57) == True & (abs(int(yy[2])) > 70) == True & comp(yy[1][0],
                                                                                       yy[1][1]) == True & (
                    mkb == 0) == True):
                if ((16 in humans[ss].body_parts.keys()) == True):
                    draw.text((int(600 * round(humans[ss].body_parts[16].x, 2)),
                               int(450 * round(humans[ss].body_parts[16].y, 2))), "站立", (0, 255, 0), font=font)
                    pos[ss] = 2
                # cv2.putText(image0, "stand", (int(600 * round(humans[ss].body_parts[16].x, 2)),
                #							 int(450 * round(humans[ss].body_parts[16].y, 2))),
                #			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                elif ((17 in humans[ss].body_parts.keys()) == True):
                    draw.text((int(600 * round(humans[ss].body_parts[17].x, 2)),
                               int(450 * round(humans[ss].body_parts[17].y, 2))), "站立", (0, 255, 0),
                              font=font)
                    pos[ss] = 2
                # cv2.putText(image0, "stand", (int(600 * round(humans[ss].body_parts[17].x, 2)),
                #							 int(450 * round(humans[ss].body_parts[17].y, 2))),
                #			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            elif ((abs(int(xx[2])) < 8) == True & (abs(int(yy[2])) > 70) == True & comp(yy[1][0],
                                                                                        yy[1][
                                                                                            1]) == True & (
                          mkb == 0) == True):
                if ((16 in humans[ss].body_parts.keys()) == True):
                    draw.text((int(600 * round(humans[ss].body_parts[16].x, 2)),
                               int(450 * round(humans[ss].body_parts[16].y, 2))), "站立", (0, 255, 0),
                              font=font)
                    pos[ss] = 2

                # cv2.putText(image0, "stand", (
                #	int(600 * round(humans[ss].body_parts[16].x, 2)),
                #	int(450 * round(humans[ss].body_parts[16].y, 2))),
                #			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                elif ((17 in humans[ss].body_parts.keys()) == True):
                    draw.text((int(600 * round(humans[ss].body_parts[17].x, 2)),
                               int(450 * round(humans[ss].body_parts[17].y, 2))), "站立", (0, 255, 0),
                              font=font)
                    pos[ss] = 2

                # cv2.putText(image0, "stand", (int(600 * round(humans[ss].body_parts[17].x, 2)),
                #							 int(450 * round(humans[ss].body_parts[17].y, 2))),
                #			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            elif ((abs(int(xx[2])) > 57) == True & (abs(int(yy[2])) < 8) == True & comp(yy[1][0],
                                                                                        yy[1][
                                                                                            1]) == True & (
                          mkb == 0) == True):
                if ((16 in humans[ss].body_parts.keys()) == True):
                    draw.text((int(600 * round(humans[ss].body_parts[16].x, 2)),
                               int(450 * round(humans[ss].body_parts[16].y, 2))), "站立", (0, 255, 0),
                              font=font)
                    pos[ss] = 2

                # cv2.putText(image0, "stand", (
                #	int(600 * round(humans[ss].body_parts[16].x, 2)),
                # int(450 * round(humans[ss].body_parts[16].y, 2))),
                #		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                elif ((17 in humans[ss].body_parts.keys()) == True):
                    draw.text((int(600 * round(humans[ss].body_parts[17].x, 2)),
                               int(450 * round(humans[ss].body_parts[17].y, 2))), "站立", (0, 255, 0),
                              font=font)
                    pos[ss] = 2
                # cv2.putText(image0, "stand", (int(600 * round(humans[ss].body_parts[17].x, 2)),
                #							 int(450 * round(humans[ss].body_parts[17].y, 2))),
                #			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            elif ((abs(int(xx[2])) < 8) == True & (abs(int(yy[2])) < 8) == True & comp(yy[1][0],
                                                                                       yy[1][1]) == True & (
                          mkb == 0) == True):
                if ((16 in humans[ss].body_parts.keys()) == True):
                    draw.text((int(600 * round(humans[ss].body_parts[16].x, 2)),
                               int(450 * round(humans[ss].body_parts[16].y, 2))), "站立", (0, 255, 0),
                              font=font)

                elif ((17 in humans[ss].body_parts.keys()) == True):
                    draw.text((int(600 * round(humans[ss].body_parts[17].x, 2)),
                               int(450 * round(humans[ss].body_parts[17].y, 2))), "站立", (0, 255, 0),
                              font=font)
                    pos[ss] = 2

            elif (comp1(yy[1][0], yy[1][1]) == True):
                if (mkb == 0):
                    if ((16 in humans[ss].body_parts.keys()) == True):
                        draw.text((int(600 * round(humans[ss].body_parts[16].x, 2)),
                                   int(450 * round(humans[ss].body_parts[16].y, 2))), "弯腰", (0, 255, 0),
                                  font=font)
                        pos[ss] = 3
                    elif ((17 in humans[ss].body_parts.keys()) == True):
                        draw.text((int(600 * round(humans[ss].body_parts[17].x, 2)),
                                   int(450 * round(humans[ss].body_parts[17].y, 2))), "弯腰", (0, 255, 0),
                                  font=font)
                        pos[ss] = 3
            elif (((0 in humans[ss].body_parts.keys()) == True & (
                    13 in humans[ss].body_parts.keys()) == True) | (
                          (0 in humans[ss].body_parts.keys()) == True & (
                          10 in humans[ss].body_parts.keys()) == True)):
                liex = 0
                if ((0 in humans[ss].body_parts.keys()) == True & (
                        13 in humans[ss].body_parts.keys()) == True):
                    if ((J013 < 0.06) == True):
                        if ((16 in humans[ss].body_parts.keys()) == True):
                            draw.text((int(600 * round(humans[ss].body_parts[16].x, 2)),
                                       int(450 * round(humans[ss].body_parts[16].y, 2))), "平躺", (0, 255, 0),
                                      font=font)
                            liex = 1
                            pos[ss] = 4
                        elif ((17 in humans[ss].body_parts.keys()) == True):
                            draw.text((int(600 * round(humans[ss].body_parts[17].x, 2)),
                                       int(450 * round(humans[ss].body_parts[17].y, 2))), "平躺", (0, 255, 0),
                                      font=font)
                            liex = 1
                            pos[ss] = 4

                if ((0 in humans[ss].body_parts.keys()) == True & (
                        10 in humans[ss].body_parts.keys()) == True & (
                        liex == 0) == True):
                    if ((J010 < 0.06) == True):
                        if ((16 in humans[ss].body_parts.keys()) == True):
                            draw.text((int(600 * round(humans[ss].body_parts[16].x, 2)),
                                       int(450 * round(humans[ss].body_parts[16].y, 2))), "平躺", (0, 255, 0),
                                      font=font)
                            pos[ss] = 4


                        elif ((17 in humans[ss].body_parts.keys()) == True):
                            draw.text((int(600 * round(humans[ss].body_parts[17].x, 2)),
                                       int(450 * round(humans[ss].body_parts[17].y, 2))), "平躺", (0, 255, 0),
                                      font=font)
                            pos[ss] = 4

            if (((4 in humans[ss].body_parts.keys()) == True & (
                    3 in humans[ss].body_parts.keys()) == True) == True | (
                    (7 in humans[ss].body_parts.keys()) == True & (
                    6 in humans[ss].body_parts.keys()) == True) == True):
                armx = 0
                if ((4 in humans[ss].body_parts.keys()) == True & (
                        3 in humans[ss].body_parts.keys()) == True):
                    if (humans[ss].body_parts[4].y <= humans[ss].body_parts[3].y):
                        draw.text((int(600 * round(humans[ss].body_parts[4].x, 2)),
                                   int(450 * round(humans[ss].body_parts[4].y, 2))), "抬臂", (0, 255, 0),
                                  font=font)
                        armx = 1
                if ((7 in humans[ss].body_parts.keys()) == True & (
                        6 in humans[ss].body_parts.keys()) == True & (
                        armx == 0) == True):
                    if (humans[ss].body_parts[7].y <= humans[ss].body_parts[6].y):
                        draw.text((int(600 * round(humans[ss].body_parts[7].x, 2)),
                                   int(450 * round(humans[ss].body_parts[7].y, 2))), "抬臂", (0, 255, 0),
                                  font=font)
            ##以下是抬腿的试验
            if (((11 in humans[ss].body_parts.keys()) == True & (
                    13 in humans[ss].body_parts.keys()) == True & (
                         12 in humans[ss].body_parts.keys()) == True) == True | (
                    (8 in humans[ss].body_parts.keys()) == True & (
                    10 in humans[ss].body_parts.keys()) == True & (9 in humans[ss].body_parts.keys()) == True) == True):
                legx = 0
                if ((8 in humans[ss].body_parts.keys()) == True & (
                        9 in humans[ss].body_parts.keys()) == True & (10 in humans[ss].body_parts.keys()) == True):
                    rightleg = np.empty(shape=[0, 2], dtype=int)  # 创建右腿最小外接旋转矩型
                    rightleg = np.append(rightleg, [[int(100 * round(humans[ss].body_parts[8].x, 2)),
                                                     int(100 * round(humans[ss].body_parts[8].y, 2))]], axis=0)
                    rightleg = np.append(rightleg, [[int(100 * round(humans[ss].body_parts[10].x, 2)),
                                                     int(100 * round(humans[ss].body_parts[10].y, 2))]], axis=0)
                    rightlegbox = cv2.minAreaRect(rightleg)
                    if ((abs(rightlegbox[2]) < 70) == True & (abs(rightlegbox[2]) > 10) == True):
                        draw.text((int(600 * round(humans[ss].body_parts[10].x, 2)),
                                   int(450 * round(humans[ss].body_parts[10].y, 2))), "踢腿", (0, 255, 0),
                                  font=font)
                        legx = 1
                if ((11 in humans[ss].body_parts.keys()) == True & (
                        13 in humans[ss].body_parts.keys()) == True & (12 in humans[ss].body_parts.keys()) == True & (
                        legx == 0) == True):
                    leftleg = np.empty(shape=[0, 2], dtype=int)  # 创建左腿最小外接旋转矩型
                    leftleg = np.append(leftleg, [[int(100 * round(humans[ss].body_parts[11].x, 2)),
                                                   int(100 * round(humans[ss].body_parts[11].y, 2))]], axis=0)
                    leftleg = np.append(leftleg, [[int(100 * round(humans[ss].body_parts[13].x, 2)),
                                                   int(100 * round(humans[ss].body_parts[13].y, 2))]],
                                        axis=0)
                    leftlegbox = cv2.minAreaRect(leftleg)
                    if ((abs(leftlegbox[2]) < 70) == True & (abs(leftlegbox[2]) > 10) == True):
                        draw.text((int(600 * round(humans[ss].body_parts[13].x, 2)),
                                   int(450 * round(humans[ss].body_parts[13].y, 2))), "踢腿", (0, 255, 0),
                                  font=font)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(0)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    yolo=YOLO()

    while True:
        ret_val, image = cam.read()
        logger.debug('')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        ##
        if ret_val != True:
            break
        t1 = time.time()

        # image = Image.fromarray(frame)
        image1 = Image.fromarray(image[..., ::-1])  # bgr to rgb
        boxs = yolo.detect_image(image1)
        # print("box_num",len(boxs))
        features = encoder(image, boxs)

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(image, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)




        logger.debug('')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        logger.debug('')

        cv2.imshow('abc', image )
        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %f" % (fps))

        cam.set(cv2.CAP_PROP_FPS,30)

        if cv2.waitKey(1) == 27:
            break
        logger.debug('')

    cv2.destroyAllWindows()

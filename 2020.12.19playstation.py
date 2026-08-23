# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import tkinter as tk
# from tkinter import messagebox
# from tkinter import ttk
from PIL import Image, ImageDraw, ImageFont
from PIL import Image, ImageTk
import matplotlib.animation as animation
import matplotlib.lines as line
# from matplotlib.pylab import mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # NavigationToolbar2TkAgg
import argparse
import logging
# import time
import matplotlib.pyplot as plt
# from tkinter import messagebox
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import pyaudio
# import wave

import os
# from timeit import time
import warnings
# import sys
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

height = 960
width = 1280


class TK(tk.Tk):
	# menu function

	def cap_get(self):
		global width, imagelista
		global height
		'''
		# self.im_rd = self.cap.read()
		imagelista = os.listdir('I://dataset/')
		print(imagelista)
		rootdir = "I://dataset/"
		print(rootdir + imagelista[0])
		for i in range(len(imagelista)):
			self.im_rd = cv2.imread(rootdir + imagelista[i])
			print(rootdir + imagelista[i])
		'''
		self.image11 = self.image22 = self.im_rd

		image = Image.fromarray(cv2.cvtColor(self.im_rd, cv2.COLOR_BGR2RGB))
		self.imgtk = ImageTk.PhotoImage(image=image)

	print("cap_get over")

	def camera_open(self):
		wholee = [0] * 100

		def yo(self, id_cam):
			yolo = YOLO()
			# Definition of the parameters
			max_cosine_distance = 0.3
			nn_budget = None
			nms_max_overlap = 1.0

			# deep_sort
			model_filename = 'model_data/mars-small128.pb'
			encoder = gdet.create_box_encoder(model_filename, batch_size=1)

			metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
			tracker = Tracker(metric)

			# video_capture = cv2.VideoCapture(0)

			fps = 0.0
			while True:
				frame = self.image22  # frame shape 640*480*3

				# image = Image.fromarray(frame)
				image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
				boxs = yolo.detect_image(image)
				# print("box_num",len(boxs))
				features = encoder(frame, boxs)

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
				x = 0
				for track in tracker.tracks:
					if not track.is_confirmed() or track.time_since_update > 1:
						continue
					bbox = track.to_tlbr()
					cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
					a = ''
					if (wholee[x] == 1):
						a = str(x) + '号人蹲下'
					if (wholee[x] == 2):
						a = str(x) + '号人站起'
					if (wholee[x] == 3):
						a = str(x) + '号人弯腰'
					if (wholee[x] == 4):
						a = str(x) + '号人躺下'
					x = x + 1
					#

					frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 对image0进行修改 以便于标注出中文
					pilimg = Image.fromarray(frame)
					draw = ImageDraw.Draw(pilimg)
					font = ImageFont.truetype("simhei.ttf", 28, encoding="utf-8")  # 设置字体
					draw.text((int(bbox[0]), int(bbox[1])), a, (0, 255, 0),
							  font=font)
					frame = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
				# cv2.putText(frame,a, (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)
				# str(track.track_id)
				for det in detections:
					bbox = det.to_tlbr()
					cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

				image5 = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
				image6 = ImageTk.PhotoImage(image=image5)
				self.lab[2].imgtk = image6
				self.lab[2].config(image=image6)

		def cao(self, id_cam):
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
										   int(450 * round(humans[ss].body_parts[16].y, 2))), "站立", (0, 255, 0),
										  font=font)
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
												   int(450 * round(humans[ss].body_parts[16].y, 2))), "", (0, 255, 0),
												  font=font)
										liex = 1
										pos[ss] = 4
									elif ((17 in humans[ss].body_parts.keys()) == True):
										draw.text((int(600 * round(humans[ss].body_parts[17].x, 2)),
												   int(450 * round(humans[ss].body_parts[17].y, 2))), "", (0, 255, 0),
												  font=font)
										liex = 1
										pos[ss] = 4

							if ((0 in humans[ss].body_parts.keys()) == True & (
									10 in humans[ss].body_parts.keys()) == True & (
									liex == 0) == True):
								if ((J010 < 0.06) == True):
									if ((16 in humans[ss].body_parts.keys()) == True):
										draw.text((int(600 * round(humans[ss].body_parts[16].x, 2)),
												   int(450 * round(humans[ss].body_parts[16].y, 2))), "", (0, 255, 0),
												  font=font)
										pos[ss] = 4


									elif ((17 in humans[ss].body_parts.keys()) == True):
										draw.text((int(600 * round(humans[ss].body_parts[17].x, 2)),
												   int(450 * round(humans[ss].body_parts[17].y, 2))), "", (0, 255, 0),
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
								10 in humans[ss].body_parts.keys()) == True & (
										9 in humans[ss].body_parts.keys()) == True) == True):
							legx = 0
							if ((8 in humans[ss].body_parts.keys()) == True & (
									9 in humans[ss].body_parts.keys()) == True & (
									10 in humans[ss].body_parts.keys()) == True):
								rightleg = np.empty(shape=[0, 2], dtype=int)  # 创建右腿最小外接旋转矩型
								rightleg = np.append(rightleg, [[int(100 * round(humans[ss].body_parts[8].x, 2)),
																 int(100 * round(humans[ss].body_parts[8].y, 2))]],
													 axis=0)
								rightleg = np.append(rightleg, [[int(100 * round(humans[ss].body_parts[10].x, 2)),
																 int(100 * round(humans[ss].body_parts[10].y, 2))]],
													 axis=0)
								rightlegbox = cv2.minAreaRect(rightleg)
								if ((abs(rightlegbox[2]) < 70) == True & (abs(rightlegbox[2]) > 10) == True):
									draw.text((int(600 * round(humans[ss].body_parts[10].x, 2)),
											   int(450 * round(humans[ss].body_parts[10].y, 2))), "踢腿", (0, 255, 0),
											  font=font)
									legx = 1
							if ((11 in humans[ss].body_parts.keys()) == True & (
									13 in humans[ss].body_parts.keys()) == True & (
									12 in humans[ss].body_parts.keys()) == True & (legx == 0) == True):
								leftleg = np.empty(shape=[0, 2], dtype=int)  # 创建左腿最小外接旋转矩型
								leftleg = np.append(leftleg, [[int(100 * round(humans[ss].body_parts[11].x, 2)),
															   int(100 * round(humans[ss].body_parts[11].y, 2))]],
													axis=0)
								leftleg = np.append(leftleg, [[int(100 * round(humans[ss].body_parts[13].x, 2)),
															   int(100 * round(humans[ss].body_parts[13].y, 2))]],
													axis=0)
								leftlegbox = cv2.minAreaRect(leftleg)
								if ((abs(leftlegbox[2]) < 70) == True & (abs(leftlegbox[2]) > 10) == True):
									draw.text((int(600 * round(humans[ss].body_parts[13].x, 2)),
											   int(450 * round(humans[ss].body_parts[13].y, 2))), "踢腿", (0, 255, 0),
											  font=font)

						## 以下是动作判断实验：
						if (pos[ss] != temp[ss]):
							if (pos[ss] == 1):
								wholee[ss] = pos[ss]
							elif (pos[ss] == 2):
								wholee[ss] = pos[ss]
							elif (pos[ss] == 3):
								wholee[ss] = pos[ss]
							elif (pos[ss] == 4):
								wholee[ss] = pos[ss]

			##判断结束 以下为姿态识别程序
			logger = logging.getLogger('')
			logger.setLevel(logging.DEBUG)
			ch = logging.StreamHandler()
			ch.setLevel(logging.DEBUG)
			formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
			ch.setFormatter(formatter)
			logger.addHandler(ch)

			parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
			parser.add_argument('--camera', type=int, default=0)

			parser.add_argument('--resize', type=str, default='0x0',
								help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
			parser.add_argument('--resize-out-ratio', type=float, default=4.0,
								help='if provided, resize heatmaps before they are post-processed. default=1.0')

			parser.add_argument('--model', type=str, default='mobilenet_thin',
								help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
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

			image0 = self.image11  # 获取图像
			logger.info('cam image=%dx%d' % (image0.shape[1], image0.shape[0]))
			while True:
				image0 = self.image11  # 从摄像头一直获取图像

				logger.debug('')
				humans = e.inference(image0, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

				logger.debug('')
				image0 = TfPoseEstimator.draw_humans(image0, humans, imgcopy=False)  # 画出姿态并显示在image0上

				self.image22 = image0

				image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)  # 对image0进行修改 以便于标注出中文
				pilimg = Image.fromarray(image0)
				draw = ImageDraw.Draw(pilimg)
				font = ImageFont.truetype("simhei.ttf", 28, encoding="utf-8")  # 设置字体

				charge0(humans)  # 进行判定
				image0 = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)

				logger.debug('')
				image1 = Image.fromarray(cv2.cvtColor(image0, cv2.COLOR_BGR2RGB))
				image2 = ImageTk.PhotoImage(image=image1)

				self.lab[1].imgtk = image2  # 左上角放置图像
				self.lab[1].config(image=image2)

				logger.debug('')

		def cam(self, id_cam):
			"""

			:type self: object
			"""
			while (self.im_rd.all ):
				self.lab[id_cam].imgtk = self.imgtk
				self.lab[id_cam].config(image=self.imgtk)
			# time.sleep(0.01)
			print("cam over")

		self.filemenu[1].entryconfig('打开摄像头', state=tk.DISABLED)
		self.filemenu[1].entryconfig('关闭摄像头', state=tk.NORMAL)
		self.filemenu[1].entryconfig('开始录制', state=tk.NORMAL)
		# self.cap=cv2.VideoCapture(0)
		# flag, self.im_rd = self.cap.read()
		import os
		imagelist = os.listdir('I://dataset/')
		print(imagelist)
		rootdir = "I://dataset/"
		print(rootdir + imagelist[0])
		for i in range(len(imagelist)):
			self.im_rd = cv2.imread(rootdir + imagelist[i])
			print(rootdir + imagelist[i])

		height,width = self.im_rd.shape[:2]

		image = Image.fromarray(cv2.cvtColor(self.im_rd, cv2.COLOR_BGR2RGB))
		self.imgtk = ImageTk.PhotoImage(image=image)

		t_0 = threading.Thread(target=self.cap_get, args=())
		t_0.daemon = True
		t_0.start()
		t_cam = []
		for i in range(0, 1):  # 打开摄像头，并把原始图像放置在第一个模块
			t_cam.append(threading.Thread(target=cam, args=(self, i)))
			t_cam[i].daemon = True
			t_cam[i].start()

		for i in range(1, 2):  # 对原始图像进行姿态识别，并显示在第二个模块
			t_cam.append(threading.Thread(target=yo, args=(self, i)))
			t_cam[i].daemon = True
			t_cam[i].start()
		for i in range(2, 3):  # 对第二个模块的图像进行重识别，并将结果显示在第三个模块
			t_cam.append(threading.Thread(target=cao, args=(self, i)))
			t_cam[i].daemon = True
			t_cam[i].start()

	def camera_close(self):
		self.cap.release()
		self.filemenu[1].entryconfig('打开摄像头', state=tk.NORMAL)
		self.filemenu[1].entryconfig('关闭摄像头', state=tk.DISABLED)
		self.filemenu[1].entryconfig('开始录制', state=tk.DISABLED)

	def audio_open(self):
		CHUNK = 1024
		data = []
		frames = []
		counter = 1
		self.f.clf()
		a = self.f.add_subplot(111, xlim=(0, CHUNK), ylim=(-10000, 10000))
		a.set_title('Real Time')
		rt_line = line.Line2D([], [])
		rt_x_data = np.arange(0, CHUNK, 1)
		self.rt_data = np.arange(0, CHUNK, 1)

		def plot_init():
			a.add_line(rt_line)
			return rt_line,

		def plot_update(i):
			rt_line.set_xdata(rt_x_data)
			rt_line.set_ydata(self.rt_data)
			return rt_line,

		ani = animation.FuncAnimation(self.f, plot_update,
									  init_func=plot_init,
									  frames=1,
									  interval=30,
									  blit=True)
		self.p = pyaudio.PyAudio()
		q = queue.Queue()

		def audio_callback(in_data, frame_count, time_info, status):
			q.put(in_data)
			ad_rdy_ev.set()
			if counter <= 0:
				return (None, pyaudio.paComplete)
			else:
				return (None, pyaudio.paContinue)

		self.stream = self.p.open(format=pyaudio.paInt16,
								  channels=1,
								  rate=44100,
								  input=True,
								  output=False,
								  frames_per_buffer=CHUNK,
								  stream_callback=audio_callback)
		print("Start Recording")
		self.stream.start_stream()
		window = signal.hamming(CHUNK)

		def read_audio_thead(q, frames, ad_rdy_ev):
			while self.stream.is_active():
				ad_rdy_ev.wait(timeout=1000)
				if not q.empty():
					# process audio data here
					data = q.get()
					while not q.empty():
						q.get()
					self.rt_data = np.frombuffer(data, np.dtype('<i2'))
					self.rt_data = self.rt_data * window
			# ad_rdy_ev.clear()
			print("over")

		ad_rdy_ev = threading.Event()
		t = threading.Thread(target=read_audio_thead, args=(q, frames, ad_rdy_ev))
		t.daemon = True
		t.start()
		self.filemenu[3].entryconfig('打开音频图', state=tk.DISABLED)
		self.filemenu[3].entryconfig('关闭音频图', state=tk.NORMAL)
		print('over')

	def audio_close(self):

		self.stream.stop_stream()
		self.stream.close()
		self.p.terminate()
		self.filemenu[3].entryconfig('打开音频图', state=tk.NORMAL)
		self.filemenu[3].entryconfig('关闭音频图', state=tk.DISABLED)

	def record_start(self):
		def rcd_btn(win_r, rcd_var, fps):
			def ttt(rcd_var, fps):
				sz = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
				fourcc = cv2.VideoWriter_fourcc(*'mp4v')
				vout = cv2.VideoWriter()
				vout.open('output.mp4', fourcc, fps, sz, True)
				cnt = 1
				fps_0 = self.cap.get(5)
				while cnt < fps_0 * rcd_vr + 1:
					vout.write(self.im_rd)
					cnt += 1
					cv2.waitKey(30)
				vout.release()

			rcd_vr = float(rcd_var)
			fps = float(fps)
			win_r.destroy()
			t = threading.Thread(target=ttt, args=(rcd_var, fps))
			t.daemon = True
			t.start()

		win_r = tk.Tk()
		win_r.title('settings')
		win_r.geometry('200x120')
		tk.Label(win_r, text='希望录制的时间（秒）').pack()
		e0 = tk.Entry(win_r)
		e0.pack()
		tk.Label(win_r, text='希望录制的帧率').pack()
		e1 = tk.Entry(win_r)
		e1.pack()
		tk.Button(win_r, text='确定', command=lambda: rcd_btn(win_r, e0.get(), e1.get())).pack()

	def __init__(self):
		tk.Tk.__init__(self)
		self.title('动态识别')
		self.geometry('1280x960')
		self.geometry("+100+20")
		self.update()
		# menu
		self.filemenu = []
		menubar = tk.Menu(self)
		self.config(menu=menubar)
		for i, con in enumerate(['文件', '姿态识别', '短时事件', '异常音频']):
			self.filemenu.append(tk.Menu(menubar, tearoff=0))
			menubar.add_cascade(label=con, menu=self.filemenu[i])

		# File menu
		self.filemenu[0].add_command(label='退出')
		self.filemenu[0].add_command(label='保存')
		# 姿态识别

		# Carema menu
		self.filemenu[1].add_command(label='打开摄像头', state=tk.NORMAL, command=self.camera_open)
		self.filemenu[1].add_command(label='关闭摄像头', state=tk.DISABLED, command=self.camera_close)
		self.filemenu[1].add_command(label='开始录制', state=tk.DISABLED, command=self.record_start)
		self.filemenu[1].add_command(label='保存')
		self.filemenu[1].add_command(label='参数')

		# Audio menu
		self.filemenu[3].add_command(label='打开音频图', state=tk.NORMAL, command=self.audio_open)
		self.filemenu[3].add_command(label='关闭音频图', state=tk.DISABLED, command=self.audio_close)
		# label
		self.lab = []
		for i, anchor_lab in enumerate(['se', 'sw', 'ne', 'nw']):
			self.lab.append(tk.Label(self))
			self.lab[i].place(x=width / 2, y=height / 2, anchor=anchor_lab)
		# canvas
		self.f = plt.figure(figsize=(width / 200, height / 200))
		self.canvas = FigureCanvasTkAgg(self.f, master=self.lab[3])
		self.canvas.draw()
		self.canvas.get_tk_widget().grid(row=0, columnspan=3)


if __name__ == "__main__":
	TK().mainloop()

from __future__ import division, print_function, absolute_import
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image,ImageTk
import matplotlib.pyplot as plt
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

height=800
width=1200

class TK(tk.Tk):
	#menu function
	def cap_get(self):
		global width
		global height
		while(self.cap.isOpened()):
			flag, self.im_rd = self.cap.read()
			#im_rd = cv2.flip(im_rd,1)#镜像
			image = Image.fromarray(cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB))
			self.imgtk=ImageTk.PhotoImage(image=image)

		print("cap_get over")

	def camera_open(self):	

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
			shang = (14, 15, 16, 17, 1, 0, 8, 11)
			xia = (8, 9, 10, 11, 12, 13)
			mo = 0
			if (len(humans) > 0):
				for ss in range(0, len(humans)):
					J89 = J910 = J1112 = J1213 = J010 = J013 = J18 = J111 = J98 = J1211 = 10000
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
					if ((11 in humans[ss].body_parts.keys()) == True & (12 in humans[ss].body_parts.keys()) == True):
						if (humans[ss].body_parts[11].y - humans[ss].body_parts[12].y == 0):
							J1112 = 0.001
						else:
							J1112 = abs(humans[ss].body_parts[11].x - humans[ss].body_parts[12].x) / abs(
								humans[ss].body_parts[11].y - humans[ss].body_parts[12].y)
					if ((11 in humans[ss].body_parts.keys()) == True & (12 in humans[ss].body_parts.keys()) == True):
						if (humans[ss].body_parts[11].x - humans[ss].body_parts[12].x == 0):
							J1211 = 10000
						else:
							J1211 = abs(humans[ss].body_parts[11].y - humans[ss].body_parts[12].y) / abs(
								humans[ss].body_parts[11].x - humans[ss].body_parts[12].x)
					if ((12 in humans[ss].body_parts.keys()) == True & (13 in humans[ss].body_parts.keys()) == True):
						if (humans[ss].body_parts[12].y - humans[ss].body_parts[13].y == 0):
							J1213 = 0.001
						else:
							J1213 = abs(humans[ss].body_parts[13].x - humans[ss].body_parts[12].x) / abs(
								humans[ss].body_parts[13].y - humans[ss].body_parts[12].y)
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

					x = np.empty(shape=[0, 2], dtype=int)
					y = np.empty(shape=[0, 2], dtype=int)
					for ca in range(0, 18):
						if ((ca in humans[ss].body_parts.keys()) == True & (ca in shang) == True):
							x = np.append(x, [[int(100 * round(humans[ss].body_parts[ca].x, 2)),
											   int(100 * round(humans[ss].body_parts[ca].y, 2))]], axis=0)
						elif ((ca in humans[ss].body_parts.keys()) == True & (ca in xia) == True):
							y = np.append(x, [[int(100 * round(humans[ss].body_parts[ca].x, 2)),
											   int(100 * round(humans[ss].body_parts[ca].y, 2))]], axis=0)
					xx = cv2.minAreaRect(x)
					yy = cv2.minAreaRect(y)
					# print("xx=",abs(int(xx[2])))
					# print("yy=",abs(int(yy[2])))
					# print("chang =",yy[1][0])
					# print("yuan = ",yy[1][1])
					# if((abs(int(yy[2])>70)) == True &)
					mkb = 0
					if (((8 in humans[ss].body_parts.keys()) == True & (
							9 in humans[ss].body_parts.keys()) == True) == True | (
							(11 in humans[ss].body_parts.keys()) == True & (
							12 in humans[ss].body_parts.keys()) == True) == True):
						if ((8 in humans[ss].body_parts.keys()) == True & (9 in humans[ss].body_parts.keys()) == True):
							print(J98)
							print(J1211)
							if ((J98 < 0.5) == True):
								if ((16 in humans[ss].body_parts.keys()) == True):
									cv2.putText(image, "sit", (
										int(600 * round(humans[ss].body_parts[16].x, 2)),
										int(450 * round(humans[ss].body_parts[16].y, 2))),
												cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
									mkb = 1
								elif ((17 in humans[ss].body_parts.keys()) == True):
									cv2.putText(image, "sit", (int(600 * round(humans[ss].body_parts[17].x, 2)),
															   int(450 * round(humans[ss].body_parts[17].y, 2))),
												cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
									mkb = 1
						elif ((11 in humans[ss].body_parts.keys()) == True & (
								12 in humans[ss].body_parts.keys()) == True):
							if ((J1211 < 0.5) == True):
								if ((16 in humans[ss].body_parts.keys()) == True):
									cv2.putText(image, "sit", (
										int(600 * round(humans[ss].body_parts[16].x, 2)),
										int(450 * round(humans[ss].body_parts[16].y, 2))),
												cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
									mkb = 1
								elif ((17 in humans[ss].body_parts.keys()) == True):
									cv2.putText(image, "sit", (int(600 * round(humans[ss].body_parts[17].x, 2)),
															   int(450 * round(humans[ss].body_parts[17].y, 2))),
												cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
									mkb = 1

					if ((abs(int(xx[2])) > 57) == True & (abs(int(yy[2])) > 70) == True & comp(yy[1][0],
																							   yy[1][1]) == True & (
							mkb == 0) == True):
						if ((16 in humans[ss].body_parts.keys()) == True):
							cv2.putText(image, "stand", (int(600 * round(humans[ss].body_parts[16].x, 2)),
														 int(450 * round(humans[ss].body_parts[16].y, 2))),
										cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

						elif ((17 in humans[ss].body_parts.keys()) == True):
							cv2.putText(image, "stand", (int(600 * round(humans[ss].body_parts[17].x, 2)),
														 int(450 * round(humans[ss].body_parts[17].y, 2))),
										cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

					elif ((abs(int(xx[2])) < 8) == True & (abs(int(yy[2])) > 70) == True & comp(yy[1][0],
																								yy[1][1]) == True & (
								  mkb == 0) == True):
						if ((16 in humans[ss].body_parts.keys()) == True):
							cv2.putText(image, "stand", (
								int(600 * round(humans[ss].body_parts[16].x, 2)),
								int(450 * round(humans[ss].body_parts[16].y, 2))),
										cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

						elif ((17 in humans[ss].body_parts.keys()) == True):
							cv2.putText(image, "stand", (int(600 * round(humans[ss].body_parts[17].x, 2)),
														 int(450 * round(humans[ss].body_parts[17].y, 2))),
										cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

					elif ((abs(int(xx[2])) > 57) == True & (abs(int(yy[2])) < 8) == True & comp(yy[1][0],
																								yy[1][1]) == True & (
								  mkb == 0) == True):
						if ((16 in humans[ss].body_parts.keys()) == True):
							cv2.putText(image, "stand", (
								int(600 * round(humans[ss].body_parts[16].x, 2)),
								int(450 * round(humans[ss].body_parts[16].y, 2))),
										cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

						elif ((17 in humans[ss].body_parts.keys()) == True):
							cv2.putText(image, "stand", (int(600 * round(humans[ss].body_parts[17].x, 2)),
														 int(450 * round(humans[ss].body_parts[17].y, 2))),
										cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

					elif ((abs(int(xx[2])) < 8) == True & (abs(int(yy[2])) < 8) == True & comp(yy[1][0],
																							   yy[1][1]) == True & (
								  mkb == 0) == True):
						if ((16 in humans[ss].body_parts.keys()) == True):
							cv2.putText(image, "stand", (
								int(600 * round(humans[ss].body_parts[16].x, 2)),
								int(450 * round(humans[ss].body_parts[16].y, 2))),
										cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

						elif ((17 in humans[ss].body_parts.keys()) == True):
							cv2.putText(image, "stand", (int(600 * round(humans[ss].body_parts[17].x, 2)),
														 int(450 * round(humans[ss].body_parts[17].y, 2))),
										cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

					elif (comp1(yy[1][0], yy[1][1]) == True):
						if (mkb == 0):
							if ((16 in humans[ss].body_parts.keys()) == True):
								cv2.putText(image, "bend", (int(600 * round(humans[ss].body_parts[16].x, 2)),
															int(450 * round(humans[ss].body_parts[16].y, 2))),
											cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

							elif ((17 in humans[ss].body_parts.keys()) == True):
								cv2.putText(image, "bend", (int(600 * round(humans[ss].body_parts[17].x, 2)),
															int(450 * round(humans[ss].body_parts[17].y, 2))),
											cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

					if (((0 in humans[ss].body_parts.keys()) == True & (13 in humans[ss].body_parts.keys()) == True) | (
							(0 in humans[ss].body_parts.keys()) == True & (
							10 in humans[ss].body_parts.keys()) == True)):
						liex = 0
						if ((0 in humans[ss].body_parts.keys()) == True & (13 in humans[ss].body_parts.keys()) == True):
							if ((J013 < 0.15) == True):
								if ((16 in humans[ss].body_parts.keys()) == True):
									cv2.putText(image, "lie", (
										int(600 * round(humans[ss].body_parts[16].x, 2)),
										int(450 * round(humans[ss].body_parts[16].y, 2))),
												cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
									liex = 1

								elif ((17 in humans[ss].body_parts.keys()) == True):
									cv2.putText(image, "lie", (int(600 * round(humans[ss].body_parts[17].x, 2)),
															   int(450 * round(humans[ss].body_parts[17].y, 2))),
												cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
									liex = 1

						if ((0 in humans[ss].body_parts.keys()) == True & (
								10 in humans[ss].body_parts.keys()) == True & (
								liex == 0) == True):
							if ((J010 < 0.15) == True):
								if ((16 in humans[ss].body_parts.keys()) == True):
									cv2.putText(image, "lie", (
										int(600 * round(humans[ss].body_parts[16].x, 2)),
										int(450 * round(humans[ss].body_parts[16].y, 2))),
												cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

								elif ((17 in humans[ss].body_parts.keys()) == True):
									cv2.putText(image, "lie", (int(600 * round(humans[ss].body_parts[17].x, 2)),
															   int(450 * round(humans[ss].body_parts[17].y, 2))),
												cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

					if (((4 in humans[ss].body_parts.keys()) == True & (
							3 in humans[ss].body_parts.keys()) == True) == True | (
							(7 in humans[ss].body_parts.keys()) == True & (
							6 in humans[ss].body_parts.keys()) == True) == True):
						armx = 0
						if ((4 in humans[ss].body_parts.keys()) == True & (3 in humans[ss].body_parts.keys()) == True):
							if (humans[ss].body_parts[4].y <= humans[ss].body_parts[3].y):
								cv2.putText(image, "arm", (int(600 * round(humans[ss].body_parts[4].x, 2)),
														   int(450 * round(humans[ss].body_parts[4].y, 7))),
											cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
								armx = 1
						if ((7 in humans[ss].body_parts.keys()) == True & (
								6 in humans[ss].body_parts.keys()) == True & (
								armx == 0) == True):
							if (humans[ss].body_parts[7].y <= humans[ss].body_parts[6].y):
								cv2.putText(image, "arm", (int(600 * round(humans[ss].body_parts[7].x, 2)),
														   int(450 * round(humans[ss].body_parts[7].y, 2))),
											cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		def ztsb(self, id_cam):

			logger = logging.getLogger('')
			logger.setLevel(logging.DEBUG)
			ch = logging.StreamHandler()
			ch.setLevel(logging.DEBUG)
			formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
			ch.setFormatter(formatter)
			logger.addHandler(ch)

			fps_time = 0

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
			image = self.im_rd

			logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

			while True:
				image = self.im_rd

				logger.debug('')
				humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
				# 判断站立的几种情况：
				charge0(humans)
				# charge(humans)

				#      print(humans[ss].body_parts)
				#   x = np.empty(shape=[0, 2], dtype=int)
				#  for ca in range(0,17):
				#     if((ca in humans[ss].body_parts.keys()) == True):
				#        x = np.append(x,[[int(100*round(humans[ss].body_parts[ca].x,2)),int(100*round(humans[ss].body_parts[ca].y,2))]],axis=0)
				# print(x)
				# axis_list = np.array(x)
				# hull = cv2.convexHull(axis_list, clockwise=True, returnPoints=True)
				# print(hull)
				# hull = np.squeeze(hull)
				# plt.scatter(axis_list[:, 0], axis_list[:, 1])
				# plt.plot(hull[:, 0], hull[:, 1], "r")
				# plt.plot([hull[-1, 0], hull[0, 0]], [hull[-1, 1], hull[0, 1]], "r")
				## plt.show()
				logger.debug('')
				image0 = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
				logger.debug('')
				image1 = Image.fromarray(cv2.cvtColor(image0, cv2.COLOR_BGR2RGB))
				image2 = ImageTk.PhotoImage(image=image1)
				# cv2.putText(image,
				#           "FPS: %f" % (1.0 / (time.time() - fps_time)),
				#          (100, 200),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
				#         (0, 255, 0), 2)
				# if (len(humans) > 0):
				#   for ss in range(0, len(humans)):
				#      print(humans[ss].body_parts)
				#   if((14 in humans[ss].body_parts.keys()) == True):
				#      print(int(400 * round(humans[ss].body_parts[14].x, 2)))
				#     print(int(400 * round(humans[ss].body_parts[14].y, 2)))
				#    cv2.putText(image, "zhanli", (int(600*round(humans[ss].body_parts[14].x,2)), int(290*round(humans[ss].body_parts[14].x,2))
				# ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
				self.lab[1].imgtk = image2
				self.lab[1].config(image=image2)

				fps_time = time.time()

				logger.debug('')

		def cam(self,id_cam):
			while(self.cap.isOpened()):
				self.lab[id_cam].imgtk=self.imgtk
				self.lab[id_cam].config(image=self.imgtk)
				time.sleep(0.000001)
			print("cam over")
		self.filemenu[4].entryconfig('打开摄像头',state=tk.DISABLED)
		self.filemenu[4].entryconfig('关闭摄像头',state=tk.NORMAL)
		self.filemenu[5].entryconfig('开始录制',state=tk.NORMAL)
		self.cap=cv2.VideoCapture(0)
		flag, self.im_rd = self.cap.read()
		height,width = self.im_rd.shape[:2]
		image = Image.fromarray(cv2.cvtColor(self.im_rd, cv2.COLOR_BGR2RGB))
		self.imgtk=ImageTk.PhotoImage(image=image)

		t_0=threading.Thread(target=self.cap_get,args=())
		t_0.daemon=True
		t_0.start()
		t_cam=[]
		for i in range(0,1):
			t_cam.append(threading.Thread(target=cam,args=(self,i)))
			t_cam[i].daemon=True
			t_cam[i].start()
		#for i in range(1,2):
		#	t_cam.append(threading.Thread(target=ztsb, args=(self, i)))
		#	t_cam[i].daemon = True
		#	t_cam[i].start()
		
	def camera_close(self):
		self.cap.release()
		self.filemenu[4].entryconfig('打开摄像头',state=tk.NORMAL)
		self.filemenu[4].entryconfig('关闭摄像头',state=tk.DISABLED)
		self.filemenu[5].entryconfig('开始录制',state=tk.DISABLED)
	
	def audio_open(self):
		CHUNK = 1024
		data =[]
		frames=[]
		counter=1
		self.f.clf()
		a=self.f.add_subplot(111,xlim=(0,CHUNK), ylim=(-10000,10000))
		a.set_title('Real Time')
		rt_line = line.Line2D([],[])
		rt_x_data=np.arange(0,CHUNK,1)
		self.rt_data=np.arange(0,CHUNK,1)

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
				return (None,pyaudio.paComplete)
			else:
				return (None,pyaudio.paContinue)

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

		def read_audio_thead(q,frames,ad_rdy_ev):
			while self.stream.is_active():
				ad_rdy_ev.wait(timeout=1000)
				if not q.empty():
					#process audio data here
					data=q.get()
					while not q.empty():
						q.get()
					self.rt_data = np.frombuffer(data,np.dtype('<i2'))
					self.rt_data = self.rt_data * window
				#ad_rdy_ev.clear()
			print("over")
		ad_rdy_ev=threading.Event()
		t=threading.Thread(target=read_audio_thead,args=(q,frames,ad_rdy_ev))
		t.daemon=True
		t.start()
		self.filemenu[6].entryconfig('打开音频图',state=tk.DISABLED)
		self.filemenu[6].entryconfig('关闭音频图',state=tk.NORMAL)
		print('over')

	def audio_close(self):
		
		self.stream.stop_stream()
		self.stream.close()
		self.p.terminate()
		self.filemenu[6].entryconfig('打开音频图',state=tk.NORMAL)
		self.filemenu[6].entryconfig('关闭音频图',state=tk.DISABLED)
		
	
	def record_start(self):
		def rcd_btn(win_r,rcd_var,fps):
			def ttt(rcd_var,fps):
				sz = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
				fourcc = cv2.VideoWriter_fourcc(*'mp4v')
				vout = cv2.VideoWriter()
				vout.open('output.mp4', fourcc, fps, sz, True)
				cnt = 1
				fps_0=self.cap.get(5)
				while cnt < fps_0*rcd_vr+1:
					vout.write(self.im_rd)
					cnt += 1
					cv2.waitKey(30)
				vout.release()
			rcd_vr = float(rcd_var)
			fps = float(fps)
			win_r.destroy()
			t=threading.Thread(target=ttt,args=(rcd_var,fps))
			t.daemon=True
			t.start()
			
		win_r=tk.Tk()
		win_r.title('settings')
		win_r.geometry('200x120')
		tk.Label(win_r,text='希望录制的时间（秒）').pack()
		e0=tk.Entry(win_r)
		e0.pack()
		tk.Label(win_r,text='希望录制的帧率').pack()
		e1=tk.Entry(win_r)
		e1.pack()
		tk.Button(win_r,text='确定',command=lambda: rcd_btn(win_r,e0.get(),e1.get())).pack()
	
	def __init__(self):
		tk.Tk.__init__(self)
		self.title('动态识别')
		self.geometry('1200x800')
		self.update()
		#menu
		self.filemenu=[]
		menubar=tk.Menu(self)
		self.config(menu=menubar)
		for i,con in enumerate(['顺时时间','文件','姿态识别','异常音频','摄像头','录制','音频图']):
			self.filemenu.append(tk.Menu(menubar, tearoff=0))
			menubar.add_cascade(label=con, menu=self.filemenu[i])
		

		#File menu
		self.filemenu[1].add_command(label='退出')
		self.filemenu[1].add_command(label='保存') 
		#姿态识别		
	
		#Carema menu
		self.filemenu[4].add_command(label='打开摄像头',state=tk.NORMAL,command=self.camera_open)
		self.filemenu[4].add_command(label='关闭摄像头',state=tk.DISABLED,command=self.camera_close)
		self.filemenu[4].add_command(label='保存')
		self.filemenu[4].add_command(label='参数')
		
			#录制
		self.filemenu[5].add_command(label='开始录制',state=tk.DISABLED,command=self.record_start)
		
		#Audio menu
		self.filemenu[6].add_command(label='打开音频图',state=tk.NORMAL,command=self.audio_open)
		self.filemenu[6].add_command(label='关闭音频图',state=tk.DISABLED,command=self.audio_close)
		#label
		self.lab=[]
		for i,anchor_lab in enumerate(['se','sw','ne','nw']):
			self.lab.append(tk.Label(self))
			self.lab[i].place(x=width/2,y=height/2,anchor=anchor_lab)
		#canvas
		self.f = plt.figure(figsize=(width//200,height//200))
		self.canvas = FigureCanvasTkAgg(self.f, master=self.lab[3]) 
		self.canvas.draw() 
		self.canvas.get_tk_widget().grid(row=0, columnspan=3)   
		

if __name__ == "__main__":
	TK().mainloop()

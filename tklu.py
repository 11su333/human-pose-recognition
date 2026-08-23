import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image,ImageTk

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as line
from matplotlib.pylab import mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk #NavigationToolbar2TkAgg

import numpy as np
import cv2
import pyaudio
import wave

import time
import queue
import threading
from scipy import signal

height=400
width=600

class TK(tk.Tk):
	#menu function
	def camera_open(self):
		def cap_get(self):
			global width
			global height
			while(self.cap.isOpened()):
				flag, im_rd = self.cap.read()
				#im_rd = cv2.flip(im_rd,1)#镜像
				im_rd = cv2.resize(im_rd,(width//2, height//2), interpolation = cv2.INTER_CUBIC)
				image = Image.fromarray(cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB))
				self.imgtk=ImageTk.PhotoImage(image=image)
			print("cap_get over")
		def cam(self,id_cam):
			while(self.cap.isOpened()):
				self.lab[id_cam].imgtk=self.imgtk
				self.lab[id_cam].config(image=self.imgtk)
				time.sleep(0.000001)
			print("cam over")
		self.filemenu[1].entryconfig('open camera',state=tk.DISABLED)
		self.filemenu[1].entryconfig('close camera',state=tk.NORMAL)
		self.cap=cv2.VideoCapture(0)
		flag, im_rd = self.cap.read()
		height,width = im_rd.shape[:2]
		image = Image.fromarray(cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB))
		self.imgtk=ImageTk.PhotoImage(image=image)
		t_0=threading.Thread(target=cap_get,args=(self,))
		t_0.daemon=True
		t_0.start()
		t_cam=[]
		for i in range(0,2):
			t_cam.append(threading.Thread(target=cam,args=(self,i)))
			t_cam[i].daemon=True
			t_cam[i].start()
		
	def camera_close(self):
		self.cap.release()
		self.filemenu[1].entryconfig('open camera',state=tk.NORMAL)
		self.filemenu[1].entryconfig('close camera',state=tk.DISABLED)
	
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
		self.filemenu[2].entryconfig('open audio',state=tk.DISABLED)
		self.filemenu[2].entryconfig('close audio',state=tk.NORMAL)
		print('over')

	def audio_close(self):
		
		self.stream.stop_stream()
		self.stream.close()
		self.p.terminate()
		self.filemenu[2].entryconfig('open audio',state=tk.NORMAL)
		self.filemenu[2].entryconfig('close audio',state=tk.DISABLED)
		
	
	def record_start(self):
		def rcd_btn(win_r,rcd_var,fps):
			rcd_vr = float(rcd_var)
			fps = float(fps)
			print(fps)
			##打开摄像头
			cap = cv2.VideoCapture(0)
			##视频大小设置，获取帧宽度，获取帧高度
			sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
			
			# 输出格式
			fourcc = cv2.VideoWriter_fourcc(*'mp4v')
			##open and set props
			vout = cv2.VideoWriter()
			vout.open('output.mp4', fourcc, fps, sz, True)
			cnt = 1
			fps_0=cap.get(5)
			while cnt < fps_0*rcd_vr+1:
				_, frame = cap.read()
				##putText输出到视频上，各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
				cv2.putText(frame, str(cnt), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)
				vout.write(frame)
				cnt += 1
				cv2.imshow('vidio', frame)
				cv2.waitKey(30)
			vout.release()
			cap.release()
			win_r.destroy()
		
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
		self.title('mysql')
		self.geometry('600x400')
		self.update()
		#menu
		self.filemenu=[]
		menubar=tk.Menu(self)
		self.config(menu=menubar)
		for i,con in enumerate(['File','Camera','Audio','录制']):
			self.filemenu.append(tk.Menu(menubar, tearoff=0))
			menubar.add_cascade(label=con, menu=self.filemenu[i])
		#File menu
		self.filemenu[0].add_command(label='New')
		self.filemenu[0].add_command(label='Open')
		self.filemenu[0].add_command(label='Save')
		self.filemenu[0].add_command(label='Exit') 
		#Carema menu
		self.filemenu[1].add_command(label='open camera',state=tk.NORMAL,command=self.camera_open)
		self.filemenu[1].add_command(label='close camera',state=tk.DISABLED,command=self.camera_close)
		#Audio menu
		self.filemenu[2].add_command(label='open audio',state=tk.NORMAL,command=self.audio_open)
		self.filemenu[2].add_command(label='close audio',state=tk.DISABLED,command=self.audio_close)
		#录制
		self.filemenu[3].add_command(label='开始录制',command=self.record_start)
		#label
		self.lab=[]
		for i,anchor_lab in enumerate(['se','sw','ne','nw']):
			self.lab.append(tk.Label(self))
			self.lab[i].place(x=width/2,y=height/2,anchor=anchor_lab)
		#canvas
		self.f = plt.figure(figsize=(width//200,height//200))
		self.canvas = FigureCanvasTkAgg(self.f, master=self.lab[2]) 
		self.canvas.draw() 
		self.canvas.get_tk_widget().grid(row=0, columnspan=3)   
		
		def update_lab(event):
			global width
			global height
			self.update()
			width=self.winfo_width()
			height=self.winfo_height()
			for i,anchor_lab in enumerate(['se','sw','ne','nw']):
				self.lab[i].height=height//2
				self.lab[i].width=width//2
				self.lab[i].place(x=width/2,y=height/2,anchor=anchor_lab)
		self.bind('<Configure>', update_lab)
		
if __name__ == "__main__":
	TK().mainloop()

import numpy as np 
import tensorflow as tf
from mss import mss
import sys
import time
import cv2
from pynput.mouse import Button, Controller
import keyboard

def to_rgb(cref):
	mask = 0xff
	R = (cref & mask) / 255
	G = ((cref >> 8) & mask) / 255
	B = ((cref >> 16) & mask) / 255
	return(R, G, B)

def new_construct_window(bound_box):
	screenshot = np.array(mss().grab(bound_box))
	# array = plt.imread(screenshot)
	return screenshot


def detect_faces(w,h, face_cascade):
	x = (1920-w) // 2
	y = (1080-h) // 2
	bound_box = {'top' : y, 'left' : x, 'width' : w, 'height' : h}
	gray = cv2.cvtColor(new_construct_window(bound_box), cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale( gray, scaleFactor = 1.1, minNeighbors = 3, minSize = (20,20) )
	if faces == ():
		return 0
	else:
		face_center = (faces[0][0] + (faces[0][2] // 2) + x, faces[0][1] + (faces[0][3] // 2) + y)
		return face_center

# print(detect_faces(500,500))

mouse = Controller()

while not (keyboard.is_pressed("=")):
	face_cascade = cv2.CascadeClassifier('C:/Users/aring/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
	if keyboard.is_pressed("f"):
		start = time.perf_counter()
		coords = detect_faces(500,500, face_cascade)
		print(time.perf_counter() - start)
		if coords != 0:
			mouse.position = coords
			mouse.click(Button.left,1)
			time.sleep(0.05)
		print(time.clock() - start)


import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os

class Parser:
	IMAGE_PATH = "../images/"
	@staticmethod
	def load_image(image_number):
		image = cv.imread(IMAGE_PATH+str(image_number) +".png",0)
		return image

	@staticmethod
	def load_images(images_list):
		##new numpy array here
		for image_number in images_list:
			image = cv.imread(IMAGE_PATH+str(image_number) +".png",0)


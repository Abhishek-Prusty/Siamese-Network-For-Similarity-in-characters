import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle	
from keras.layers import concatenate	
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Activation, Flatten, Dense, Dropout
from datetime import datetime
from keras.models import load_model
import glob
import cv2
from keras.preprocessing.image import img_to_array


model=load_model('model_balVGG.h5')
files = glob.glob ("data/*.png")

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i1", "--image1", required=True,
                help="Path to the image1")
ap.add_argument("-i2", "--image2", required=True,
                help="Path to the image2")

args = vars(ap.parse_args())
im1=args["image1"]
im2=args["image2"]

image1 = cv2.imread(im1)
image1=cv2.resize(image1,(32,32))
image1 = img_to_array(image1)
image1=np.array(image1,dtype="object")/255.0

image2 = cv2.imread(im2)
image2=cv2.resize(image2,(32,32))
image2 = img_to_array(image2)
image2=np.array(image2,dtype="object")/255.0


pred_sim = model.predict([image1.reshape(-1,32,32,3), image2.reshape(-1,32,32,3)])
print(pred_sim*100)

# print(len(files))
# data=[]
# for file in files:
# 	image = cv2.imread(file,0)
# 	image=cv2.resize(image,(24,16))
# 	image = img_to_array(image)
# 	data.append(image)

# data=np.array(data,dtype="object")/255.0
# #print(len(data))

# store={}
# for i in range(len(data)):
# 	store[i]=[]
# 	print(i)
# 	for j in range(len(data)):
# 		pred_sim = model.predict([data[i].reshape(-1,24,16,1), data[j].reshape(-1,24,16,1)])
# 		store[i].append([j,pred_sim])

# final={}
# for i in range(len(store)):
# 	d = sorted(store[i],key=lambda kv: kv[1],reverse=True)
# 	if(i==0):
# 		print(d)
# 	d=d[1:12]
# 	final[i]=d

# for temp in range(len(final)):
# 	fig=plt.figure(figsize=(15, 15))
# 	fig.add_subplot(1,11,1)
# 	img1 = cv2.imread('data/'+str(temp)+'.png',0)
# 	plt.imshow(img1,cmap='gray')
# 	for i in range(len(final[temp])):
# 		print(final[temp][i][0])
# 		img = cv2.imread('data/'+str(final[temp][i][0])+'.png',0)
# 		fig.add_subplot(1, 11, i+1)
# 		plt.xlabel(str(final[temp][i][1]))
# 		plt.imshow(img,cmap='gray')
# 	plt.savefig('result/'+str(temp)+'.png')
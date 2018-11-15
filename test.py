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
from keras_drop_block import DropBlock2D

from datetime import datetime
from keras.models import load_model
#from keras_sequential_ascii import keras2ascii
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
from keras_drop_block import DropBlock2D
from keras.applications import VGG16
from keras.applications.resnet50 import ResNet50


threshold=95




with open('bal_dataRGB.pickle','rb') as f:
	data=pickle.load(f)

with open('bal_labelsRGB.pickle','rb') as f:
    labels=pickle.load(f)


data = np.array(data, dtype="float") / 255.0
print(data.shape)
# img_path = 'elephant.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)


x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.3)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32,3),pooling='avg')


b2 = Model(inputs=base_model.input, outputs=base_model.get_layer('block3_pool').output)
fc1 = b2.layers[-3]
fc2 = b2.layers[-2]
predictions = b2.layers[-1]

dropout1 = DropBlock2D(block_size=3,keep_prob=0.8)
dropout2 = DropBlock2D(block_size=3,keep_prob=0.8)

x = dropout1(fc1.output)
x = fc2(x)
x = dropout2(x)
x = predictions(x)
x =Flatten()(x)
vgg_conv=Model(inputs=base_model.input, outputs=x)

for layer in vgg_conv.layers:
    layer.trainable = False
vgg_conv.layers[-3].trainable=True
vgg_conv.layers[-4].trainable=True
vgg_conv.layers[-5].trainable=True
vgg_conv.summary()

img_a_in = Input(shape = x_train.shape[1:], name = 'ImageA_Input')
img_b_in = Input(shape = x_train.shape[1:], name = 'ImageB_Input')
img_a_feat = vgg_conv(img_a_in)
img_b_feat = vgg_conv(img_b_in)

combined_features = concatenate([img_a_feat, img_b_feat], name = 'merge_features')
combined_features = Dense(128, activation = 'linear')(combined_features)
combined_features = BatchNormalization()(combined_features)
combined_features = Dropout(0.5)(combined_features)
combined_features = Activation('relu')(combined_features)
combined_features = Dense(16, activation = 'linear')(combined_features)
combined_features = BatchNormalization()(combined_features)
combined_features = Dropout(0.5)(combined_features)
combined_features = Activation('relu')(combined_features)
combined_features = Dense(1, activation = 'sigmoid')(combined_features)


model = Model(inputs = [img_a_in, img_b_in], outputs =[combined_features], name = 'model')




model.load_weights('khmerRGBvggdb3_96.h5')

model.summary()



import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i1", "--image1", required=True,
                help="Path to the image1")
ap.add_argument("-f", "--folder", required=True,
                help="path to folder containing images")

args = vars(ap.parse_args())
im1=args["image1"]
fol=args["folder"]

files = glob.glob (str(fol)+"*")
files=sorted(files)

image1 = cv2.imread(im1)
image1=cv2.resize(image1,(32,32))
image1 = img_to_array(image1)
image1=np.array(image1,dtype="object")/255.0

files2 = glob.glob ("results/*")
for f in files2:
    os.remove(f)

for imgpath in files:


	image2 = cv2.imread(imgpath)
	image2=cv2.resize(image2,(32,32))
	ims=image2
	image2 = img_to_array(image2)
	image2=np.array(image2,dtype="object")/255.0


	pred_sim1 = model.predict([image1.reshape(-1,32,32,3),image2.reshape(-1,32,32,3)])
	pred_sim2 = model.predict([image2.reshape(-1,32,32,3),image1.reshape(-1,32,32,3)])
	print(pred_sim1*100,pred_sim2*100)
	if(pred_sim1*100 > threshold and pred_sim2*100>threshold):
		cv2.imwrite("results/"+str(max(pred_sim1,pred_sim2)*100)+".png",ims)

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
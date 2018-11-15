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
from keras.layers import Input, Conv2D,GlobalAveragePooling2D, BatchNormalization, MaxPool2D, Activation, Flatten, Dense, Dropout,Lambda
from datetime import datetime
from keras.models import load_model
#from keras_sequential_ascii import keras2ascii
from keras import backend as K
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
from keras_drop_block import DropBlock2D
from keras.applications import VGG16
from keras.applications.resnet50 import ResNet50

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
print(x_train.shape)

train_groups = [x_train[np.where(y_train==i)[0]] for i in np.unique(y_train)]
test_groups = [x_test[np.where(y_test==i)[0]] for i in np.unique(y_train)]
print('train groups:', [x.shape[0] for x in train_groups])
print('test groups:', [x.shape[0] for x in test_groups])


def gen_random_batch(in_groups, batch_halfsize = 8):
    out_img_a, out_img_b, out_score = [], [], []
    all_groups = list(range(len(in_groups)))
    for match_group in [True, False]:
        group_idx = np.random.choice(all_groups, size = batch_halfsize)
        out_img_a += [in_groups[c_idx][np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in group_idx]
        if match_group:
            b_group_idx = group_idx
            out_score += [1]*batch_halfsize
        else:
            # anything but the same group
            non_group_idx = [np.random.choice([i for i in all_groups if i!=c_idx]) for c_idx in group_idx] 
            b_group_idx = non_group_idx
            out_score += [0]*batch_halfsize
            
        out_img_b += [in_groups[c_idx][np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in b_group_idx]
            
    return np.stack(out_img_a,0), np.stack(out_img_b,0), np.stack(out_score,0)




# pv_a, pv_b, pv_sim = gen_random_batch(train_groups, 3)
# fig, m_axs = plt.subplots(2, pv_a.shape[0], figsize = (12, 6))
# for c_a, c_b, c_d, (ax1, ax2) in zip(pv_a, pv_b, pv_sim, m_axs.T):
#     ax1.imshow(c_a[:,:,0])
#     ax1.set_title('Image A')
#     ax1.axis('off')
#     ax2.imshow(c_b[:,:,0])
#     ax2.set_title('Image B\n Similarity: %3.0f%%' % (100*c_d))
#     ax2.axis('off')

###################
#plt.show()


img_in = Input(shape = x_train.shape[1:], name = 'FeatureNet_ImageInput')
n_layer = img_in
for i in range(2):
    n_layer = Conv2D(8*2**i, kernel_size = (3,3), activation = 'linear')(n_layer)
    n_layer = BatchNormalization()(n_layer)
    n_layer = Activation('relu')(n_layer)
    n_layer = Conv2D(16*2**i, kernel_size = (3,3), activation = 'linear')(n_layer)
    n_layer = BatchNormalization()(n_layer)
    n_layer = Activation('relu')(n_layer)
    n_layer = MaxPool2D((2,2))(n_layer)
n_layer = Flatten()(n_layer)
n_layer = Dense(32, activation = 'linear')(n_layer)
n_layer = Dropout(0.5)(n_layer)
n_layer = BatchNormalization()(n_layer)
n_layer = Activation('relu')(n_layer)
feature_model = Model(inputs = [img_in], outputs = [n_layer], name = 'FeatureGenerationModel')
#feature_model.summary()


#print(keras2ascii(feature_model))
#feature_model.summary()


img_a_in = Input(shape = x_train.shape[1:], name = 'ImageA_Input')
img_b_in = Input(shape = x_train.shape[1:], name = 'ImageB_Input')
img_a_feat = vgg_conv(img_a_in)
img_b_feat = vgg_conv(img_b_in)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([img_a_feat, img_b_feat])


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


similarity_model = Model(inputs = [img_a_in, img_b_in], outputs =[distance], name = 'Similarity_Model')



def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

similarity_model.summary()
#print(keras2ascii(similarity_model))
similarity_model.compile(loss=contrastive_loss, optimizer='adam', metrics=['mse'])
#similarity_model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['mae','acc'])


def show_model_output(nb_examples = 5):
    pv_a, pv_b, pv_sim = gen_random_batch(test_groups, nb_examples)
    pred_sim = similarity_model.predict([pv_a, pv_b])
    fig, m_axs = plt.subplots(2, pv_a.shape[0], figsize = (12, 6))
    for c_a, c_b, c_d, p_d, (ax1, ax2) in zip(pv_a, pv_b, pv_sim, pred_sim, m_axs.T):
        ax1.imshow(c_a[:,:,0])
        ax1.set_title('A\n A: %3.0f%%' % (100*c_d))
        ax1.axis('off')
        ax2.imshow(c_b[:,:,0])
        ax2.set_title('B\n P: %3.0f%%' % (100*p_d))
        ax2.axis('off')
    return fig

#mport _ = show_model_output()
#plt.show()


def siam_gen(in_groups, batch_size = 256):
    while True:
        pv_a, pv_b, pv_sim = gen_random_batch(train_groups, batch_size//2)
        yield [pv_a, pv_b], pv_sim




valid_a, valid_b, valid_sim = gen_random_batch(test_groups, 1024)

loss_history = similarity_model.fit_generator(siam_gen(train_groups), 
                               steps_per_epoch = 2000,
                               validation_data=([valid_a, valid_b], valid_sim),
                                              epochs = 20,
                                             verbose = True)



_ = show_model_output()
plt.savefig('output.png')

name='model-{}'.format(str(datetime.now()))
similarity_model.save(name+'.h5')


#similarity_model=load_model('model-2018-11-04 14:53:25.385766.h5')

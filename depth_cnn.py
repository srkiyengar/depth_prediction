from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import Dropout, Flatten, Dense, Input,Reshape,  MaxPooling2D, Masking
from keras.layers.merge import concatenate
from keras.engine import Layer
import matplotlib.pyplot as plt

import dummy
import numpy as np

import keras.backend as K
from functools import partial

from keras.layers import Lambda
import tensorflow as tf
# https://stackoverflow.com/questions/44186042/keras-methods-to-enlarge-spartial-dimension-of-the-layer-output-blob
def UpSampling2DBilinear(output_shape, **kwargs):
    def layer(x):
        return tf.image.resize_bilinear(x, output_shape, align_corners=True)
    return Lambda(layer, **kwargs)



img_h = 120
img_w = 160
img_c = 3




def myLossFunction(y_true,y_pred,ymask):

    n = K.sum(ymask,axis=[1,2])
    yresult = y_pred*ymask
    #d = K.log(y_true + K.epsilon())-K.log(yresult + K.epsilon())
    d = y_true - yresult
    term1 = K.sum(d*d,axis=[1,2])/n
    temp = K.sum(d,axis=[1,2])
    term2 = K.pow(temp,2)/(2*K.pow(n,2))
    return term1-term2







if __name__ == "__main__":



    #load data
    X_train, Y_train, Y_mask = dummy.load_data(dummy.dirname1, dummy.dirname2)

    # N images of 3 channels, data is scaled and normalized to mean 0 and std 1 across the channels separately.
    X_train = dummy.feature_normalize_rgb(X_train)
    Y_train = np.log(Y_train+1)
    # input shape
    input1 = Input(shape=(img_h,img_w,img_c))
    ymask_input = Input(shape=(56,76))

    # There are suggestions to mask and unmask after the input layer -https://github.com/keras-team/keras/issues/2728
    #input = Masking(mask_value=0)(input)


    # importing the vgg16 model without the Fully connected layers
    vgg16_model = VGG16(include_top=False, weights='imagenet')
    for layer in vgg16_model.layers:
        layer.trainable = False


    #scale1 = vgg16_model(input)
    scale1 = vgg16_model(input1)

    scale1 = Flatten()(scale1)
    scale1 = Dense(4256,activation='linear')(scale1)
    # Applying drop-out to the fully connected layer
    scale1 = Dropout(0.5)(scale1)
    scale1 = Reshape((56,76,1))(scale1)
    #scale1_reshaped = Reshape((56,76))(scale1)

    #scale1_model = Model(inputs=input1, outputs=scale1_reshaped)
    #scale1_model.compile(optimizer='sgd', loss='mse', metrics=['mse'])  # Compile the model
    #print(scale1_model.summary())  # Summarize the model

    scale2 = Conv2D(63,9,strides=1, activation='relu', data_format="channels_last")(input1)
    scale2 = MaxPooling2D((2, 2), strides=(2, 2))(scale2)

    concatenated = concatenate([scale1,scale2])
    scale2 = Conv2D(64, 5, strides=1, padding='same', activation='relu', data_format="channels_last")(concatenated)
    # last layer has linear activiation
    scale2 = Conv2D(1, 5, strides=1, padding='same', activation='linear', data_format="channels_last")(scale2)

    scale2_reshaped = Reshape((56, 76))(scale2)
    scale2_model = Model(inputs=[input1,ymask_input], outputs=scale2_reshaped)

    #Additions
    loss_function = partial(myLossFunction, ymask=ymask_input)


    scale2_model.compile(optimizer='adam', loss=loss_function)  # Compile the model
    print(scale2_model.summary())  # Summarize the model

    #fit scale1_model only to train scale1 and then fit scale2_model only and train scale2 only.
    #scale1_model.fit(x=X_train,y=Y_train,batch_size=10,epochs=10)

    # checkpoint = ModelCheckpoint()
    hist = scale2_model.fit(x=[X_train,Y_mask],y=Y_train,validation_split=0.2, batch_size=10,epochs=100)

    X_test, Y_test = dummy.load_test_data("/home/p4bhattachan/PycharmProjects/syde770/images/test_rgb/","/home/p4bhattachan/PycharmProjects/syde770/images/test_depth/")

    # create masks with 0 and 1 for Y_train where all greater than 0 values are 1


    X_test = dummy.feature_normalize_rgb(X_test)
    My_results = scale2_model.predict([X_test,np.zeros((X_test.shape[0],56,76))],batch_size=3)


    for i in [0,1,2]:
        plt.subplot(121)
        plt.imshow(np.exp(My_results[i])-1, cmap="gray")
        plt.title('Results')
        plt.subplot(122)
        plt.imshow(Y_test[i])
        plt.title('Real')
        plt.show()

    q=5
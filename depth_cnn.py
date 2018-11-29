from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import Dropout, Flatten, Dense, Input,Reshape,  MaxPooling2D, Masking
from keras.layers.merge import concatenate
from keras.engine import Layer

import dummy

from keras.layers import Lambda
import tensorflow as tf
# https://stackoverflow.com/questions/44186042/keras-methods-to-enlarge-spartial-dimension-of-the-layer-output-blob
def UpSampling2DBilinear(output_shape, **kwargs):
    def layer(x):
        return tf.image.resize_bilinear(x, output_shape, align_corners=True)
    return Lambda(layer, **kwargs)


class LambdaMask(Layer):
    '''
    muck up the mask, deliberately.
    '''
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.supports_masking = True
        super(LambdaMask, self).__init__(*args, **kwargs)

    def compute_mask(self, x, mask=None):
        return self.func(x, mask)

    def call(self, x, mask=None):
        return x



img_h = 120
img_w = 160
img_c = 3



if __name__ == "__main__":


    #load data
    X_train, Y_train = dummy.load_data(dummy.dirname1, dummy.dirname2)

    # input shape
    input1 = Input(shape=(img_h,img_w,img_c))

    # There are suggestions to mask and unmask after the input layer -https://github.com/keras-team/keras/issues/2728
    #input = Masking(mask_value=0)(input)


    # importing the vgg16 model without the Fully connected layers
    vgg16_model = VGG16(include_top=False, weights='imagenet')


    #scale1 = vgg16_model(input)
    scale1 = vgg16_model(input1)

    scale1 = Flatten()(scale1)
    scale1 = Dense(4256,activation='relu')(scale1)
    # Applying drop-out to the fully connected layer
    scale1 = Dropout(0.5)(scale1)
    scale1 = Reshape((56,76,1))(scale1)
    scale1_reshaped = Reshape((56,76))(scale1)

    scale1_model = Model(inputs=input1, outputs=scale1_reshaped)
    scale1_model.compile(optimizer='sgd', loss='mse', metrics=['mse'])  # Compile the model
    print(scale1_model.summary())  # Summarize the model

    scale2 = Conv2D(63,9,strides=1, activation='relu', data_format="channels_last")(input1)
    scale2 = MaxPooling2D((2, 2), strides=(2, 2))(scale2)

    concatenated = concatenate([scale1,scale2])
    scale2 = Conv2D(64, 5, strides=1, padding='same', activation='relu', data_format="channels_last")(concatenated)
    # last layer has linear activiation
    scale2 = Conv2D(64, 5, strides=1, padding='same', activation='linear', data_format="channels_last")(scale2)

    scale2_model = Model(inputs=input1, outputs=scale2)
    scale2_model.compile(optimizer='sgd', loss='mse', metrics=['mse'])  # Compile the model
    print(scale2_model.summary())  # Summarize the model

    #fit scale1_model only to train scale1 and then fit scale2_model only and train scale2 only.
    scale1_model.fit(x=X_train,y=Y_train,batch_size=10,epochs=1)

    q=5
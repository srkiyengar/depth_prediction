from keras.models import Model
from keras.layers import Conv2D
from keras.layers import Dropout, Flatten, Dense, Input,Reshape,  MaxPooling2D, Masking
from keras.layers.merge import concatenate
from keras.engine import Layer
import matplotlib.pyplot as plt

import dummy
import numpy as np

import keras.backend as K


import keras.layers as L
import keras.backend as K

class NonZeroMean(L.Layer):
  """Compute mean of non-zero entries."""
  def call(self, x):
    """Calculate non-zero mean."""
    # count the number of nonzero features, last axis
    nonzero = K.any(K.not_equal(x, 0.0), axis=-1)
    n = K.sum(K.cast(nonzero, 'float32'), axis=-1, keepdims=True)
    x_mean = K.sum(x, axis=-2) / n
    return x_mean

  def compute_output_shape(self, input_shape):
    """Collapse summation axis."""
    return input_shape[:-2] + (input_shape[-1],)





if __name__ == "__main__":

    x = [[[[1, 2, 3], [2, 3, 4], [0, 0, 0]],
          [[1, 2, 3], [2, 0, 4], [3, 4, 5]],
          [[1, 2, 3], [0, 0, 0], [0, 0, 0]],
          [[1, 2, 3], [1, 2, 3], [0, 0, 0]]],
         [[[1, 2, 3], [0, 1, 0], [0, 0, 0]],
          [[1, 2, 3], [2, 3, 4], [0, 0, 0]],
          [[1, 2, 3], [0, 0, 0], [0, 0, 0]],
          [[1, 2, 3], [1, 2, 3], [1, 2, 3]]]]
    x = np.array(x, dtype='float32')

    # Example run
    x_input = K.placeholder(shape=x.shape, name='x_input')
    out = NonZeroMean()(x_input)
    s = K.get_session()
    print("INPUT:", x)
    print("OUTPUT:", s.run(out, feed_dict={x_input: x}))



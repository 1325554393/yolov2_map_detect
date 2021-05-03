import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import concatenate, Dropout, LeakyReLU
from tensorflow.keras.layers import Reshape, Conv2D, Input
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization

from config import *
import numpy as np

#custom keras layer
class SpaceToDepth(keras.layers.Layer):
	"""docstring for SpaceToDepth"""
	def __init__(self, block_size, **kwargs):
		super(SpaceToDepth, self).__init__(**kwargs)
		self.block_size = block_size

	def call(self, inputs):
		x = inputs
		# print(x) # shape=(None, 32, 32, 64)
		batch, height, width, depth = K.int_shape(x)
		batch = -1
		reduced_height = height // self.block_size
		reduced_width = width // self.block_size
		y = K.reshape(x, (batch, reduced_height, self.block_size,
					reduced_width, self.block_size, depth))
		z = K.permute_dimensions(y, (0, 1, 3, 2, 4, 5))
		t = K.reshape(z, (batch, reduced_height, reduced_width, depth * self.block_size **2))
		# print(t) # shape=(None, 16, 16, 256)

		return t

	def compute_output_shape(self, input_shape):
		shape =  (input_shape[0], input_shape[1] // self.block_size, input_shape[2] // self.block_size,
				input_shape[3] * self.block_size **2)

		return tf.TensorShape(shape)


# yolov2-model
input_image = Input((IMAGE_H, IMAGE_W, 3), dtype="float32")

# Layer 1 output (512,512,32)
x = Conv2D(filters=32, kernel_size=(3,3), strides=1, padding="same",
			name="conv_1", use_bias=False)(input_image)
x = BatchNormalization(name="norm_1")(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2,2), strides=2)(x)

# Layer 2 output (256,256,64)
x = Conv2D(filters=64, kernel_size=(3,3), strides=1, padding="same",
			name="conv_2", use_bias=False)(x)
x = BatchNormalization(name="norm_2")(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2,2), strides=2)(x)

# Layer 3 output (128,128,128)
x = Conv2D(filters=128, kernel_size=(3,3), strides=1, padding="same",
			name="conv_3", use_bias=False)(x)
x = BatchNormalization(name="norm_3")(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 4 output (128,128,64)
x = Conv2D(filters=64, kernel_size=(1,1), strides=1, padding="same",
			name="conv_4", use_bias=False)(x)
x = BatchNormalization(name="norm_4")(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 5 output (128,128,128)
x = Conv2D(filters=128, kernel_size=(3,3), strides=1, padding="same",
			name="conv_5", use_bias=False)(x)
x = BatchNormalization(name="norm_5")(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2,2), strides=2)(x)

# Layer 6 output (64,64,256)
x = Conv2D(filters=256, kernel_size=(3,3), strides=1, padding="same",
			name="conv_6", use_bias=False)(x)
x = BatchNormalization(name="norm_6")(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 7 output (64,64,128)
x = Conv2D(filters=128, kernel_size=(1,1), strides=1, padding="same",
			name="conv_7", use_bias=False)(x)
x = BatchNormalization(name="norm_7")(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 8 output (64,64,256)
x = Conv2D(filters=256, kernel_size=(3,3), strides=1, padding="same",
			name="conv_8", use_bias=False)(x)
x = BatchNormalization(name="norm_8")(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2,2), strides=2)(x)

# Layer 9 output (32,32,512)
x = Conv2D(filters=512, kernel_size=(3,3), strides=1, padding="same",
			name="conv_9", use_bias=False)(x)
x = BatchNormalization(name="norm_9")(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 10 output (32,32,256)
x = Conv2D(filters=256, kernel_size=(1,1), strides=1, padding="same",
			name="conv_10", use_bias=False)(x)
x = BatchNormalization(name="norm_10")(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 11 output (32,32,512)
x = Conv2D(filters=512, kernel_size=(3,3), strides=1, padding="same",
			name="conv_11", use_bias=False)(x)
x = BatchNormalization(name="norm_11")(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 12 output (32,32,256)
x = Conv2D(filters=256, kernel_size=(1,1), strides=1, padding="same",
			name="conv_12", use_bias=False)(x)
x = BatchNormalization(name="norm_12")(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 13 output (32,32,512), skip_connection (32,32,512)
x = Conv2D(filters=256, kernel_size=(3,3), strides=1, padding="same",
			name="conv_13", use_bias=False)(x)
x = BatchNormalization(name="norm_13")(x)
x = LeakyReLU(alpha=0.1)(x)
skip_connection = x
x = MaxPooling2D(pool_size=(2,2), strides=2)(x)

# Layer 14 output (16,16,1024)
x = Conv2D(filters=1024, kernel_size=(3,3), strides=1, padding="same",
			name="conv_14", use_bias=False)(x)
x = BatchNormalization(name="norm_14")(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 15 output (16,16,512)
x = Conv2D(filters=512, kernel_size=(1,1), strides=1, padding="same",
			name="conv_15", use_bias=False)(x)
x = BatchNormalization(name="norm_15")(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 16 output (16,16,1024)
x = Conv2D(filters=1024, kernel_size=(3,3), strides=1, padding="same",
			name="conv_16", use_bias=False)(x)
x = BatchNormalization(name="norm_16")(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 17 output (16,16,512)
x = Conv2D(filters=512, kernel_size=(1,1), strides=1, padding="same",
			name="conv_17", use_bias=False)(x)
x = BatchNormalization(name="norm_17")(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 18 output (16,16,1024)
x = Conv2D(filters=1024, kernel_size=(3,3), strides=1, padding="same",
			name="conv_18", use_bias=False)(x)
x = BatchNormalization(name="norm_18")(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 19 output (16,16,1024)
x = Conv2D(filters=1024, kernel_size=(3,3), strides=1, padding="same",
			name="conv_19", use_bias=False)(x)
x = BatchNormalization(name="norm_19")(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 20 output (16,16,1024)
x = Conv2D(filters=1024, kernel_size=(3,3), strides=1, padding="same",
			name="conv_20", use_bias=False)(x)
x = BatchNormalization(name="norm_20")(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 21 output (32,32,64)
skip_connection = Conv2D(filters=64, kernel_size=(1,1), strides=1, padding="same",
			name="conv_21", use_bias=False)(skip_connection)
skip_connection = BatchNormalization(name="norm_21")(skip_connection)
skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
# print(skip_connection) # shape=(None, 32, 32, 64)
# becasue the shape of two tensor(skip_connection and x ) is diffrent , 
# skip_connection (32,32,64)-->shape=(None, 16, 16, 256)
# 
skip_connection = SpaceToDepth(block_size=2)(skip_connection)
# print(x) # shape=(None, 16, 16, 1024)
# print(skip_connection) # shape=(None, 16, 16, 256)
x = concatenate([skip_connection, x])
# print(x) # shape=(None, 16, 16, 1280)

# Layer 22 output (16,16,1024)
x = Conv2D(filters=1024, kernel_size=(3,3), strides=1, padding="same",
			name="conv_22", use_bias=False)(x)
x = BatchNormalization(name="norm_22")(x)
x = LeakyReLU(alpha=0.1)(x)
x = Dropout(0.5)(x)

# Layer 23 
x = Conv2D(filters=(BOX*(4+1+CLASS)), kernel_size=(1,1), strides=1, padding="same",
			name="conv_23", use_bias=False)(x)
output = Reshape((GRID_W, GRID_H, BOX, 4+1+CLASS))(x)


model = tf.keras.models.Model(input_image, output)

model.summary()
# 
# test_img = np.zeros((10,512,512,3))
# print(test_img.shape)

# pred = model(test_img, False) # (10, 16, 16, 5, 6)
# print(pred.shape)


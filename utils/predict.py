import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.keras.backend as K
from config import *
import shutil


def display_yolo(file, model, score_threshold, iou_threshold):
	"""
	Display predictions from YOLO model.

	- file : string list : list of images path.
    - model : YOLO model.
    - score_threshold : threshold used for filtering predicted bounding boxes.
    - iou_threshold : threshold used for non max suppression.
	"""
	# load img
	image = cv2.imread(file)
	input_image = image[:,:,::-1] # BGR<--->RGB
	input_image = image / 255
	input_image = np.expand_dims(input_image, 0)

	# prediction
	y_pred = model.predict_on_batch(input_image)

	# post prediction process
	# grid coords  tensor
	coord_x = tf.cast(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)), tf.float32)
	coord_y = tf.transpose(coord_x, (0,2,1,3,4))
	coords = tf.tile(tf.concat([coord_x, coord_y], -1), [TRAIN_BATCH_SIZE, 1, 1, 5, 1])
	dims = K.cast_to_floatx(K.int_shape(y_pred)[1:3])
	dims = K.reshape(dims, (1,1,1,1,2))
	# anchors tensor
	anchors = np.array(ANCHORS)
	anchors = anchors.reshape(len(anchors)//2, 2)
	# pred_xy and pred_wh shape (m, GRID_W, GRID_H, Anchors, 2)
	pred_xy = K.sigmoid(y_pred[:,:,:,:,0:2])
	pred_xy = (pred_xy + coords)
	pred_xy = pred_xy / dims
	pred_wh = K.exp(y_pred[:,:,:,:,2:4])
	pred_wh = pred_wh * anchors / dims

	# pred_confidence
	box_conf = K.sigmoid(y_pred[:,:,:,:,4:5])
	# pred_class
	box_class_prob = K.softmax(y_pred[:,:,:,:,5:])

	# Reshape
	pred_xy = pred_xy[0,...]
	pred_wh = pred_wh[0,...]
	box_conf = box_conf[0,...]
	box_class_prob = box_class_prob[0,...]

	# convert box coord from x,y,w,h to x1,y1,x2,y2
	box_xy_1 = pred_xy - 0.5 * pred_wh
	box_xy_2 = pred_xy + 0.5 * pred_wh
	boxes = K.concatenate((box_xy_1, box_xy_2), axis=-1)

	# Filter boxes
	box_scores = box_conf * box_class_prob
	box_classes = K.argmax(box_scores, axis=-1) # best score index
	box_class_scores = K.max(box_scores, axis=-1) # best score
	prediction_mask = box_class_scores >= score_threshold
	boxes = tf.boolean_mask(boxes, prediction_mask)
	scores = tf.boolean_mask(box_class_scores, prediction_mask)
	classes = tf.boolen_mask(box_classes, prediction_mask)

	# Scale box to image shape
	boxes = boxes * IMAGE_H
	pass

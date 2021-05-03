import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as K

from config import *
from utils.dataset import *

#################### Load Dataset ####################
train_dataset = get_dataset(train_image_folder, train_annot_folder, LABELS, TRAIN_BATCH_SIZE)
val_dataset= get_dataset(val_image_folder, val_annot_folder, LABELS, VAL_BATCH_SIZE)

# Ground true generator
train_gen = ground_truth_generator(train_dataset)
val_gen = ground_truth_generator(val_dataset)

#################### Define model type ####################

# Import yolo model
from utils.model import model
# model.summary()

"""
# load yolo pretrained weights if is avilable
class WeightReader:
	def __init__(self, weight_file):
		self.offset = 4
		self.all_weights = np.fromfile(weight_file, dtype="float32")

	def read_bytes(self, size):
		self.offset = self.offset + size
		return self.all_weights[self.offset-size:self.offset]

	def reset(self):
		self.offset = 4	

weight_reader = WeightReader("yolo_pretrained.weights")	

weight_reader.reset()
nb_conv = 23

for i in range(1, nb_conv+1):
	conv_layer = model.get_layer("norm_" + str(i))
	conv_layer.trainable = True

	if i < nb_conv:
		norm_layer = model.get_layer("norm_" + str(i))
		norm_layer.trainable = True

		size = np.prod(norm_layer.get_weights()[0].shape)

		beta = weight_reader.read_bytes(size)
		gamma = weight_reader.read_bytes(size)
		mean = weight_reader.read_bytes(size)
		var = weight_reader.read_bytes(size)

		weights = norm_layer.set_weights([gamma, beta, mean, var])

	if len(conv_layer.get_weights()) > 1:
		bias   = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
		kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
		kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
		kernel = kernel.transpose([2,3,1,0])
		conv_layer.set_weights([kernel, bias])
	else:
		kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
		kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
		kernel = kernel.transpose([2,3,1,0])
		conv_layer.set_weights([kernel])

layer   = model.layers[-2] # last convolutional layer
layer.trainable = True


weights = layer.get_weights()

new_kernel = np.random.normal(size=weights[0].shape)/(GRID_H*GRID_W)
new_bias   = np.random.normal(size=weights[1].shape)/(GRID_H*GRID_W)

layer.set_weights([new_kernel, new_bias])

"""

# loss function
def iou(x1, y1, w1, h1, x2, y2, w2, h2):
	"""
	Calculate IOU between box1 and box2

	Parameters
	----------
	- x, y : box center coords
	- w : box width
	- h : box height
	
	Returns
	-------
	- IOU	

	"""
	xmin1 = x1 - 0.5*w1
	xmax1 = x1 + 0.5*w1
	ymin1 = y1 - 0.5*h1
	ymax1 = y1 + 0.5*h1
	xmin2 = x2 - 0.5*w2
	xmax2 = x2 + 0.5*w2
	ymin2 = y2 - 0.5*h2
	ymax2 = y2 + 0.5*h2
	interx = np.minimum(xmax1, xmax2) - np.maximum(xmin1, xmin2)
	intery = np.minimum(ymax1, ymax2) - np.maximum(ymin1, ymin2)
	inter = interx * intery
	union = w1*h1 + w2*h2 - inter
	iou = inter / (union + 1e-6)

	return iou
# loss
def yolov2_loss(detector_mask, matching_true_boxes, class_one_hot, true_boxes_grid, y_pred, info=False):
	"""
	Calculate YOLO V2 loss from prediction (y_pred) and ground truth tensors (detector_mask,
	matching_true_boxes, class_one_hot, true_boxes_grid,)

	Parameters
	----------
	- detector_mask : tensor, shape (batch, size, GRID_W, GRID_H, anchors_count, 1)
		1 if bounding box detected by grid cell, else 0
	- matching_true_boxes : tensor, shape (batch_size, GRID_W, GRID_H, anchors_count, 5)
		Contains adjusted coords of bounding box in YOLO format
	- class_one_hot : tensor, shape (batch_size, GRID_W, GRID_H, anchors_count, class_count)
		One hot representation of bounding box label
	- true_boxes_grid : annotations : tensor (shape : batch_size, max annot, 5)
		true_boxes_grid format : x, y, w, h, c (coords unit : grid cell)
	- y_pred : prediction from model. tensor (shape : batch_size, GRID_W, GRID_H, anchors count, (5 + labels count)
	- info : boolean. True to get some infox about loss value
	
	Returns
	-------
	- loss : scalar
	- sub_loss : sub loss list : coords loss, class loss and conf loss : scalar

	"""

	# anchors tensor
	anchors = np.array(ANCHORS)
	anchors = anchors.reshape(len(anchors)//2, 2)

	# grid coords tensor ---> GRID_W * GRID*H grid
	# tf.tile(input, multiples, name=None)
	# left up corner coord , total GRID_W * GRID*H * anchor_count
	coord_x = tf.cast(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)), tf.float32)
	coord_y = tf.transpose(coord_x, (0,2,1,3,4))
	coords = tf.tile(tf.concat([coord_x, coord_y], -1), [y_pred.shape[0], 1, 1, 5, 1])

	# coordinate loss
	# box regression
	# bx = (sigmoid(tx) + cx ) /W
	# bw = pw * e^tw
	# pw is anchors W, cx is left up of coord , tx and tw are pred offset value, W is feature map width
	# in this case, we don't multipy width, because the the coord in matching value also is during 0~16
	pred_xy = K.sigmoid(y_pred[:,:,:,:,0:2]) # adjust center coords between 0 and 1
	pred_xy = (pred_xy + coords) # add cell coord for comparaison with ground truth. New coords in grid cell unit
	pred_wh = K.exp(y_pred[:,:,:,:,2:4]) * anchors # adjust width and height for comparaison with ground truth. New coords in grid cell unit
	# pred_wh = (pred_wh * anchors) # unit: grid cell
	nb_detector_mask = K.sum(tf.cast(detector_mask>0.0, tf.float32))
	xy_loss = LAMBDA_COORD*K.sum(detector_mask*K.square(matching_true_boxes[...,:2] - pred_xy))/(nb_detector_mask + 1e-6) # Non /2
	wh_loss = LAMBDA_COORD * K.sum(detector_mask * K.square(K.sqrt(matching_true_boxes[...,2:4])-
		K.sqrt(pred_wh))) / (nb_detector_mask + 1e-6)

	coord_loss = xy_loss + wh_loss

	# class loss
	pred_box_class = y_pred[...,5:]
	true_box_class = tf.argmax(class_one_hot, -1)
	# class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
	class_loss = K.sparse_categorical_crossentropy(target=true_box_class, output=pred_box_class, from_logits=True)
	class_loss = K.expand_dims(class_loss, -1)*detector_mask
	class_loss = LAMBDA_CLASS * K.sum(class_loss) / (nb_detector_mask + 1e-6)

	# confidence loss
	pred_conf = K.sigmoid(y_pred[..., 4:5]) # only two class : object or background
	# for each detector : iou between prediction and ground truth
	x1 = matching_true_boxes[...,0]
	y1 = matching_true_boxes[...,1]
	w1 = matching_true_boxes[...,2]
	h1 = matching_true_boxes[...,3]
	x2 = pred_xy[...,0]
	y2 = pred_xy[...,1]
	w2 = pred_wh[...,0]
	h2 = pred_wh[...,1]
	ious = iou(x1, y1, w1, h1, x2, y2, w2, h2)
	ious = K.expand_dims(ious, -1)

	# for each detector: best ious between pred and true_boxes
	pred_xy = K.expand_dims(pred_xy, 4)
	pred_wh = K.expand_dims(pred_wh, 4)
	pred_wh_half = pred_wh / 2.
	pred_mins = pred_xy - pred_wh_half
	pred_maxes = pred_xy + pred_wh_half
	true_boxe_shape = K.int_shape(true_boxes_grid)
	true_boxes_grid = K.reshape(true_boxes_grid, [true_boxe_shape[0], 1, 1, 1, true_boxe_shape[1], true_boxe_shape[2]])
	true_xy = true_boxes_grid[...,0:2]
	true_wh = true_boxes_grid[...,2:4]
	true_wh_half = true_wh * 0.5
	true_mins = true_xy - true_wh_half
	true_maxes = true_xy + true_wh_half
	intersect_mins = K.maximum(pred_mins, true_mins) # shape : m, GRID_W, GRID_H, BOX, max_annot, 2 
	intersect_maxes = K.minimum(pred_maxes, true_maxes) # shape : m, GRID_W, GRID_H, BOX, max_annot, 2
	intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.) # shape : m, GRID_W, GRID_H, BOX, max_annot, 1
	intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1] # shape : m, GRID_W, GRID_H, BOX, max_annot, 1
	pred_areas = pred_wh[..., 0] * pred_wh[..., 1] # shape : m, GRID_W, GRID_H, BOX, 1, 1
	true_areas = true_wh[..., 0] * true_wh[..., 1] # shape : m, GRID_W, GRID_H, BOX, max_annot, 1
	union_areas = pred_areas + true_areas - intersect_areas
	iou_scores = intersect_areas / union_areas # shape : m, GRID_W, GRID_H, BOX, max_annot, 1
	best_ious = K.max(iou_scores, axis=4)  # Best IOU scores.
	best_ious = K.expand_dims(best_ious) # shape : m, GRID_W, GRID_H, BOX, 1
	
	# no object confidence loss
	no_object_detection = K.cast(best_ious < 0.6, K.dtype(best_ious)) 
	noobj_mask = no_object_detection * (1 - detector_mask)
	nb_noobj_mask  = K.sum(tf.cast(noobj_mask  > 0.0, tf.float32))
	
	noobject_loss =  LAMBDA_NOOBJECT * K.sum(noobj_mask * K.square(-pred_conf)) / (nb_noobj_mask + 1e-6)
	# object confidence loss
	object_loss = LAMBDA_OBJECT * K.sum(detector_mask * K.square(ious - pred_conf)) / (nb_detector_mask + 1e-6)
	# total confidence loss
	conf_loss = noobject_loss + object_loss
	
	# total loss
	loss = conf_loss + class_loss + coord_loss
	sub_loss = [conf_loss, class_loss, coord_loss] 

	if info:
		print('conf_loss   : {:.4f}'.format(conf_loss))
		print('class_loss  : {:.4f}'.format(class_loss))
		print('coord_loss  : {:.4f}'.format(coord_loss))
		print('    xy_loss : {:.4f}'.format(xy_loss))
		print('    wh_loss : {:.4f}'.format(wh_loss))
		print('--------------------')
		print('total loss  : {:.4f}'.format(loss)) 

		# display masks for each anchors
		for i in range(len(anchors)):
			f, (ax1, ax2, ax3) = plt.subplot(1,3,figsize=(10,5))
			# https://blog.csdn.net/Strive_For_Future/article/details/115052014?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522161883865316780262527067%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=161883865316780262527067&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_v2~rank_v29-2-115052014.first_rank_v2_pc_rank_v29&utm_term=f.tight_layout&spm=1018.2226.3001.4187
			f.tight_layout() 
			f.suptitle("MASKS FOR ANCHOR {} :".format(anchors[i,...]))

			ax1.matshow((K.sum(detector_mask[0,:,:,i], axis=2)), cmap='Greys', vmin=0, vmax=1)
			ax1.set_title('detector_mask, count : {}'.format(K.sum(tf.cast(detector_mask[0,:,:,i]  > 0., tf.int32))))
			ax1.xaxis.set_ticks_position('bottom')
			
			ax2.matshow((K.sum(no_object_detection[0,:,:,i], axis=2)), cmap='Greys', vmin=0, vmax=1)
			ax2.set_title('no_object_detection mask')
			ax2.xaxis.set_ticks_position('bottom')
			
			ax3.matshow((K.sum(noobj_mask[0,:,:,i], axis=2)), cmap='Greys', vmin=0, vmax=1)
			ax3.set_title('noobj_mask')
			ax3.xaxis.set_ticks_position('bottom')
			  
	return loss, sub_loss


# save weights
def save_best_weights(model, name, val_loss_avg):
	# delete existing weights file
	files = glob.glob(os.path.join('weights/', name + '*'))
	for file in files:
		os.remove(file)
	# create new weights file
	name = name + '_' + str(val_loss_avg) + '.h5'
	path_name = os.path.join('weights/', name)
	model.save_weights(path_name)

# log (tensorboard)
def log_loss(loss, val_loss, step):
	tf.summary.scalar('loss', loss, step)
	tf.summary.scalar('val_loss', val_loss, step)
	

# gradients
def grad(model, img, detector_mask, matching_true_boxes, class_one_hot, true_boxes, training=True):
	with tf.GradientTape() as tape:
		y_pred = model(img, training)
		loss, sub_loss = yolov2_loss(detector_mask, matching_true_boxes, class_one_hot, true_boxes, y_pred)
	return loss, sub_loss, tape.gradient(loss, model.trainable_variables)

# training
def train(epochs, model, train_dataset, val_dataset, steps_per_epoch_train, steps_per_epoch_val, train_name = "train"):
	"""
	Train YOLO model for n epochs.
	Eval loss on training and validation dataset.
	Log training loss and validation loss for tensorboard.
	Save best weights during training (according to validation loss).

	Parameters
	----------
	- epochs : integer, number of epochs to train the model.
	- model : YOLO model.
	- train_dataset : YOLO ground truth and image generator from training dataset.
	- val_dataset : YOLO ground truth and image generator from validation dataset.
	- steps_per_epoch_train : integer, number of batch to complete one epoch for train_dataset.
	- steps_per_epoch_val : integer, number of batch to complete one epoch for val_dataset.
	- train_name : string, training name used to log loss and save weights.
	
	Notes :
	- train_dataset and val_dataset generate YOLO ground truth tensors : detector_mask,
	  matching_true_boxes, class_one_hot, true_boxes_grid. Shape of these tensors (batch size, tensor shape).
	- steps per epoch = number of images in dataset // batch size of dataset
	
	Returns
	-------
	- loss history : [train_loss_history, val_loss_history] : list of average loss for each epoch.

	"""
	num_epochs = epochs
	steps_per_epoch_train = steps_per_epoch_train
	steps_per_epoch_val = steps_per_epoch_val
	train_loss_history = []
	best_val_loss = 1e6

	# optimizer
	optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5,
										beta_1=0.9, beta_2=0.999,
										epsilon=1e-08)
	# log record(tensorboard)
	summary_writer = tf.summary.create_file_writer(os.path.join(
								"logs/", train_name), flush_millis=20000)
	summary_writer.set_as_default()

	# train
	for epoch in range(num_epochs):
		epoch_loss = []
		epoch_val_loss = []
		epoch_val_sub_loss = []
		print("Epoch {}:".format(epoch))

		for batch_idx in range(steps_per_epoch_train):
			img, detector_mask, matching_true_boxes, class_one_hot, true_boxes =  next(train_dataset)
			loss, _, grads = grad(model, img, detector_mask, matching_true_boxes, class_one_hot, true_boxes)
			optimizer.apply_gradients(zip(grads, model.trainable_variables))
			epoch_loss.append(loss)
			print("-", end=" ")
		print(" | ", end=" ")

		# val
		for batch_idx in range(steps_per_epoch_val):
			img, detector_mask, matching_true_boxes, class_one_hot, true_boxes = next(val_dataset)
			loss, sub_loss, grads = grad(model, img, detector_mask, matching_true_boxes, class_one_hot, true_boxes,training=False)
			epoch_val_loss.append(loss)
			epoch_val_sub_loss.append(sub_loss)
			print("-", end=" ")

		val_loss_avg = np.mean(np.array(epoch_val_loss))
		sub_loss_avg = np.mean(np.array(epoch_val_sub_loss), axis=0)
		train_loss_history.append(loss_avg)
		val_loss_history.append(val_loss_avg)

		# log
		log_loss(loss_avg, val_loss_avg, epoch)

		# save
		if val_loss_avg < best_val_loss:
			save_best_weights(model, train_name, val_loss_avg)
			best_val_loss = val_loss_avg
		
		print("loss = {:.4f}, val_loss = {:.4f} (conf={:.4f}, class={:.4f}, coords={:.4f})".format(
			loss_avg, val_loss_avg, sub_loss_avg[0], sub_loss_avg[1], sub_loss_avg[2]))
		
	return [train_loss_history, val_loss_history]
			



if __name__ == '__main__':
	results = train(EPOCHS, model, train_gen, val_gen, 10, 2, 'training_1')
	
	plt.plot(results[0])
	plt.plot(results[1])
	plt.show()
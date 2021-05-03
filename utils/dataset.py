import tensorflow as tf
from config import *
import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.keras.backend as K

def parse_annotation(img_dir, ann_dir, LABELS):
	"""
	Parse XML files in PASCAL VOC format

	Parameters
	----------
	- ann_dir : annotations files directory
	- img_dir : images files directory
	- labels : labels list

	Returns
	-------
	- imgs_name : numpy array of images files path (shape : images count, 1)
	- true_boxes : numpy array of annotations for each image (shape : image count, max annotation count, 5)
		annotation format : xmin, ymin, xmax, ymax, class
		xmin, ymin, xmax, ymax : image unit (pixel)
		class = label index

	"""
	max_annot = 0
	imgs_name = []
	annots = []

	# Read all Parse file in order
	for ann in sorted(os.listdir(ann_dir)):
		annot_count = 0
		boxes = []
		tree = ET.parse(ann_dir + ann)
		# print(tree) # <xml.etree.ElementTree.ElementTree object at 0x0000000012942FC8>
		for elem in tree.iter(): # get every elems in tree object 
			# print(elem) # <Element 'annotation' at 0x00000000129059A8>
			if "filename" in elem.tag:
				imgs_name.append(img_dir + elem.text)
			if "width" in elem.tag:
				w = int(elem.text)
			if "height" in elem.tag:
				h = int(elem.text)
			if "object" in elem.tag or "part" in elem.tag:
				box = np.zeros((5))
				for sub_elem in list(elem):
					if "name" in sub_elem.tag:
						box[4] = LABELS.index(sub_elem.text)+1 # one-hot: make 0 represent no ball, 1 represent have ball
					if "bndbox" in sub_elem.tag:
						annot_count += 1
						for sub_next in list(sub_elem):
							if "xmin" in sub_next.tag:
								box[0] = int(round(float(sub_next.text)))
							if "ymin" in sub_next.tag:
								box[1] = int(round(float(sub_next.text)))
							if "xmax" in sub_next.tag:
								box[2] = int(round(float(sub_next.text)))
							if "ymax" in sub_next.tag:
								box[3] = int(round(float(sub_next.text)))

				boxes.append(np.asarray(box)) # attention len(boxes) = 1
	
		if w != IMAGE_W or h != IMAGE_H:
			print("Image size error")
			break

		annots.append(np.asarray(boxes)) # list file
	
		if annot_count > max_annot: # what's meanning for?
			max_annot = annot_count

	# Rectify annotations boxes : len -> max_annot
	imgs_name = np.array(imgs_name) # 
	# print(imgs_name.shape) # (imgs count,)
	true_boxes = np.zeros((imgs_name.shape[0], max_annot, 5))
	for idx, boxes in enumerate(annots):
		true_boxes[idx, :boxes.shape[0], :5] = boxes
	# print(true_boxes[1:10,:,0:5]) # (imgs Count, 1, 5)
	 
	return imgs_name, true_boxes


def parse_function(img_obj, true_boxes):
	x_img_string = tf.io.read_file(img_obj)
	x_img = tf.image.decode_png(x_img_string, channels=3) #  shape=(None, None, 3) dtype=tf.uint8
	x_img = tf.image.convert_image_dtype(x_img, tf.float32) # pixel value /255, dtype=tf.float32, channels : RGB

	return x_img, true_boxes


def get_dataset(img_dir, ann_dir, labels, batch_size):
	"""
	create a YOLO dataset, feed into yolo model per batch

	Parameters:
	- ann_dir : annotations files directory
	- img_dir : images files directory
	- labels : labels list
	- batch_size : int

	Returns
	- YOLO dataset : generate batch
		batch : tupple(images, annotations)
		batch[0] : images : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
		batch[1] : annotations : tensor (shape : batch_size, max_annot, 5)

	Note : image pixel values = pixels value / 255. channels : RGB
	"""
	imgs_name, bbox = parse_annotation(img_dir, ann_dir, LABELS) # LABELS = ('ball')
	dataset = tf.data.Dataset.from_tensor_slices((imgs_name, bbox))
	dataset = dataset.shuffle(len(imgs_name))
	dataset = dataset.repeat() 
	# https://blog.csdn.net/nofish_xp/article/details/83116779?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522161821880916780357230023%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=161821880916780357230023&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-5-83116779.first_rank_v2_pc_rank_v29&utm_term=dataset.map%28%29&spm=1018.2226.3001.4187
	# dataset.map() function understanding, num_parallel_calls = CPU count
	dataset = dataset.map(parse_function, num_parallel_calls=4)
	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(10) # normally equal to batch
	print("----------------")
	# print(dataset)
	print("Dataset: ")
	print("Images count: {}".format(len(imgs_name)))
	print("Step per epoch: {}".format(len(imgs_name) // batch_size))
	print("Images per epoch: {}".format(batch_size * (len(imgs_name) // batch_size)))

	# shapes: ((None, None, None, 3), (None, 1, 5)), types: (tf.float32, tf.float64)>
	# why the imgs dimension shows None in the tensor,but shows 
	return dataset


def process_true_boxes(true_boxes, anchors, image_width, image_height):
	"""
	Build image ground truth in YOLO format from image true_boxes and anchors.

	Parameters
	----------
	- true_boxes : tensor, shape (category = 1, 5), format : x1 y1 x2 y2 c, coords unit : image pixel
	- anchors : list [anchor_1_width, anchor_1_height, anchor_2_width, anchor_2_height...]
		anchors coords unit : grid cell
	- image_width, image_height : int (pixels)

	Returns
	-------
	- detector_mask : array, shape (GRID_W, GRID_H, anchors_count, 1)
		1 if bounding box detected by grid cell, else 0
	- matching_true_boxes : array, shape (GRID_W, GRID_H, anchors_count, 5)
		Contains adjusted coords of bounding box in YOLO format
	-true_boxes_grid : array, same shape than true_boxes (category = 1, 5),
		format : x, y, w, h, c, coords unit : grid cell
		
	Note:
	-----
	Bounding box in YOLO Format : x, y, w, h, c
	x, y : center of bounding box, unit : grid cell
	w, h : width and height of bounding box, unit : grid cell
	c : label index , confident

	"""
	scale = IMAGE_W / GRID_W # 32 times

	anchors_count = len(anchors) // 2
	anchors = np.array(anchors)
	anchors = anchors.reshape(len(anchors)//2, 2)

	# GRID_W * GRID_H feature point , every single point have anchors_count bounding box,
	# every bounding box have 1 confidence and 4 position coord
	detector_mask = np.zeros((GRID_W, GRID_H, anchors_count, 1)) 
	matching_true_boxes = np.zeros((GRID_W, GRID_H, anchors_count, 5))

	# convert true_boxes numpy array -> tensor
	true_boxes = true_boxes.numpy() # ready to itera
	true_boxes_grid = np.zeros(true_boxes.shape) # (1,5)
	# print("-----------")
	# print(true_boxes.shape)

	# convert bounding box coords and localize bounding box
	for i, box in enumerate(true_boxes):
		# convert box coords to x, y, w, h and convert to grids coord
		w = (box[2] - box[0]) / scale # (xmax - xmin) / 32
		h = (box[3] - box[1]) / scale # (ymax - ymin) / 32
		x = ((box[0] + box[2]) / 2) / scale # value during 0~16, not 0~1
		y = ((box[1] + box[3]) / 2) / scale
		true_boxes_grid[i,...] = np.array([x, y, w, h, box[4]]) # at feature map
		if w * h > 0: # box exists
			# calculate iou between box and each anchors and find best anchors
			best_iou = 0
			best_anchor = 0
			for i in range(anchors_count):

				# iou (anchor and box are shifted to 0,0)
				# why this math can calculete the area of intersec---find the best match area type for detect object.
				intersect = np.minimum(w, anchors[i, 0]*np.minimum(h, anchors[i,1]))
				union = (anchors[i, 0]*anchors[i, 1]) + (w*h) - intersect
				iou = intersect / union
				if iou > best_iou:
					best_iou = iou
					best_anchor = i

			# localize box in detector_mask and matching true_boxes
			if best_iou > 0:
				x_coord = np.floor(x).astype("int")
				y_coord = np.floor(y).astype("int")
				detector_mask[y_coord, x_coord, best_anchor] = 1 # why is y , x not x, y?
				yolo_box = np.array([x,y,w,h,box[4]]) # # for adjuste params 
				# adjusted_box = np.array([box[0]-x_coord], box[1]-y_coord,
				# np.log(box[2]/anchors[best_anchor][0]),
				# np.log(box[3]/anchors[best_anchor][1]),
				# box[4])
				matching_true_boxes[y_coord, x_coord, best_anchor] = yolo_box
				# matching_true_boxes[y_coord, x_coord, best_anchor] = adjusted_box

	return matching_true_boxes, detector_mask, true_boxes_grid					


def ground_truth_generator(dataset):
	"""
	Ground truth batch generator from a yolo dataset, ready to compare with YOLO prediction in loss function.
	
	Parameters
	----------
	- YOLO dataset. Generate batch:
		batch : tupple(images, annotations)
		batch[0] : images : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
		batch[1] : annotations : tensor (shape : batch_size, category = 1, 5)	

	Returns
	-------
	- imgs : images to predict. tensor (shape : batch_size, IMAGE_H, IMAGE_W, 3)
	- detector_mask : tensor, shape (batch_size, GRID_W, GRID_H, anchors_count, 1)
		1 if bounding box detected by grid cell, else 0
	- matching_true_boxes : tensor, shape (batch_size, GRID_W, GRID_H, anchors_count, 5)
		Contains adjusted coords of bounding box in YOLO format
	- class_one_hot : tensor, shape (batch_size, GRID_W, GRID_H, anchors_count, class_count)
		One hot representation of bounding box label
	- true_boxes_grid : annotations : tensor (shape : batch_size, category = 1, 5)
		true_boxes format : x, y, w, h, c, coords unit : grid cell

	"""
	for batch in dataset:
		# print(batch) # [shape=(10, 512, 512, 3), shape=(10, 1, 5)]
		# imgs
		imgs = batch[0]
		# true boxes [batch, (xmin, ymin, xmax, ymax, class)]
		true_boxes = batch[1]

		# matching_true_boxes and detector_mask
		batch_matching_true_boxes = []
		batch_detector_mask = []
		batch_true_boxes_grid = []

		for i in range(true_boxes.shape[0]):
			one_matching_true_boxes, one_detector_mask, true_boxes_grid = process_true_boxes(
										true_boxes[i], ANCHORS, IMAGE_W, IMAGE_H)

			batch_matching_true_boxes.append(one_matching_true_boxes)
			batch_detector_mask.append(one_detector_mask)
			batch_true_boxes_grid.append(true_boxes_grid)

		detector_mask = tf.convert_to_tensor(np.array(batch_detector_mask), dtype="float32")
		matching_true_boxes = tf.convert_to_tensor(np.array(batch_matching_true_boxes), dtype="float32")
		true_boxes_grid = tf.convert_to_tensor(np.array(batch_true_boxes_grid), type="float32")

		# class one_hot
		matching_classes = K.cast(matching_true_boxes[...,4], "int32")
		class_one_hot = K.one_hot(matching_classes, CLASS+1)[:,:,:,:,1:]
		class_one_hot = tf.cast(class_one_hot, dtype="float32")

		batch = (imgs, detector_mask, matching_true_boxes, class_one_hot, true_boxes_grid)
		yield batch

		

	pass


def main():
	# train_dataset = get_dataset(train_image_folder, train_annot_folder, LABELS, TRAIN_BATCH_SIZE)
	val_dataset= get_dataset(val_image_folder, val_annot_folder, LABELS, VAL_BATCH_SIZE)
	# 
	# train_gen = ground_truth_generator(train_dataset)
	val_gen = ground_truth_generator(val_dataset)
	# for batch_idx in range(5):
	# 		img, detector_mask, matching_true_boxes, class_one_hot, true_boxes =  next(val_dataset)



if __name__ == '__main__':
	main()
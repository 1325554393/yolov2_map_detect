"""
Turn PKUAIP data to dataset which can directly input to yolov2 model
function 1: resize image into proper dimension
function 2: split the image to val and train
function 3: turn txt label into xml 

"""


import os,shutil
from PIL import Image
import cv2
from tqdm import tqdm # 进度条
import numpy as np
import xml.dom.minidom as minidom
import re


image_folder=r"C:\Users\Administrator\Desktop\YOLOV2-with-mAP-master\YOLOV2-with-mAP-master\YOLOv2-Tensorflow2\PKUAIP\images" # PKUAIP原始图片
label_folder=r"C:\Users\Administrator\Desktop\YOLOV2-with-mAP-master\YOLOV2-with-mAP-master\YOLOv2-Tensorflow2\PKUAIP\labels" # PKUAIP原始标签

train_image_folder = "train/image/"
train_annot_folder = "train/annotation/"
val_image_folder = "val/image/"
val_annot_folder = "val/annotation/"

SPLIT_RATIO=0.2 
SHUFFLE_SEED=108
WIDTH=512
HEIGHT=512


def check_folder(folder):
	"""The folder is exists or not"""
	if os.path.exists(folder):
		pass
	else:
		os.makedirs(folder)

# ----------------------------------------------------------
# func1: image resize

def resize_and_replace(img_path, width, height):
	"""
	Replace original img with resize img
	""" 
	img = Image.open(img_path)
	img = img.resize((width, height), Image.BILINEAR)
	os.remove(img_path)
	img.save(img_path)

 
def func1(image_folder):
	images = os.listdir(image_folder)
	for i in tqdm(images):# read all file in folder
		resize_and_replace(image_folder+"\\"+i,WIDTH,HEIGHT)


# ----------------------------------------------------------
# func2: copy resize img to val and train folder

def copy_files(file_list, des_folder):
	check_folder(des_folder)
	for file in file_list:
		shutil.copy(image_folder+"\\"+file, des_folder)


def func2():
	result=image_with_label(image_folder,label_folder)
	dataset=dataset_split(result)
	copy_files(dataset["train"],train_image_folder)
	copy_files(dataset["val"],val_image_folder)

# ----------------------------------------------------------
# func3: txt to xml

def image_with_label(image_folder, label_folder):
	"""
	Return the img with label, not all image have label
	"""
	result_images = []
	images = os.listdir(image_folder)
	labels = os.listdir(label_folder)
	# print(images[1], labels[1]) # 0002.png 0002.txt

	for image in images:
		if (str(image[:4])+ ".txt") in labels:
			result_images.append(image)

	assert len(result_images) == len(labels)
	return result_images

def dataset_split(images_list):
	"""
	Input: result_images(name of all images )
	Return: split the test and val data regarding given ratio

	"""
	dataset = {}
	# radom split with specific random seed
	np.random.seed(SHUFFLE_SEED)
	np.random.shuffle(images_list)
	
	val_num = int(SPLIT_RATIO*len(images_list))
	train_num = len(images_list)-val_num
	# print(len(images_list), train_num,val_num)
	
	dataset["train"] = images_list[:train_num+1]
	dataset["val"] = images_list[train_num+1:]

	return dataset


def to_pixel(raw, width, height):
	"""
	convert the params in txt to xml format
	Return: xml format
	"""
	center = (int(raw[0]*width), int(raw[1]*height))
	box_w = int(raw[2]*width)
	box_h = int(raw[3]*height)

	xml_format = {}
	xml_format.update({"xmin": center[0]-int(box_w/2)})
	xml_format.update({"ymin": center[1]-int(box_h/2)})
	xml_format.update({"xmax": center[0]+int(box_w/2)})
	xml_format.update({"ymax": center[1]+int(box_h/2)})

	return xml_format


def headGen(xml,folder,filename,size):
	xml.create('folder',folder)
	xml.create('filename',filename)
	xml.create('path')
	xml.create_2('source',{'database':'Unknown'})
	xml.create_2('size',{'width':str(size[0]),'height':str(size[1]),'depth':'3'})
	xml.create('segmented','0')


def objGen(xml,detail):
	xml.create_3('object',
				{'name':detail['name'],'pose':'Unspecified','truncated':'0','difficult':'0'},
				{'xmin':str(detail['xmin']),'ymin':str(detail['ymin']),'xmax':str(detail['xmax']),'ymax':str(detail['ymax'])})



class xmlGen:
	def __init__(self, name="annotation"):
		self.dom=minidom.getDOMImplementation().createDocument(None,name,None)
		self.root=self.dom.documentElement

	def create(self,node,content=None):
		element=self.dom.createElement(node)
		if content:
			element.appendChild(self.dom.createTextNode(content))
			self.root.appendChild(element)
		else:
			self.root.appendChild(element)

	def create_2(self,node,content):
		element=self.dom.createElement(node)
		self.root.appendChild(element)

		for key,value in content.items():
			element2=self.dom.createElement(key)
			element2.appendChild(self.dom.createTextNode(value))
			element.appendChild(element2)

	def create_3(self,node,content_1,content_2):
		"""
		三级节点，content_1和content_2为dict形式
		"""
		element=self.dom.createElement(node)
		self.root.appendChild(element)
	  
		for key,value in content_1.items():
			element2=self.dom.createElement(key)
			element2.appendChild(self.dom.createTextNode(value))
			element.appendChild(element2)        
		
		element_temp=self.dom.createElement('bndbox')
		for key,value in content_2.items():
			element_3=self.dom.createElement(key)
			element_3.appendChild(self.dom.createTextNode(value))
			element_temp.appendChild(element_3)
		element.appendChild(element_temp)
		
	def make(self,filePath,first_line=False):
		with open(filePath, 'w', encoding='utf-8') as f:
			self.dom.writexml(f, addindent='\t', newl='\n',encoding='utf-8')


def labelGen(txt_file,dest,folder,filename):
	"""
	Input: txt path(txt_file), destination path(dest), folder name(folder), image path(filename)
	Return: None
	"""
	content = None
	with open(txt_file, "r") as f:
		content = f.readlines()
		content = re.sub("\n","",content[0])
		content = content.split(" ") # store with 5 string elements in list
		# print(content) ['0', '0.80625', '0.9652777777777778', '0.1625', '0.0638888888888889']
		for i , c in enumerate(content):
			content[i] = float(c) # convert string elems to float in content list
		raw = content[1:]
		xml_format = to_pixel(raw, WIDTH, HEIGHT)
		xml = xmlGen() #
		headGen(xml,folder,filename,[WIDTH,HEIGHT])
		detail = {"name": "ball"}
		detail.update(xml_format)
		objGen(xml, detail)
		xml.make(dest)

		pass


def func3():
	"""
	Input: None
	Return: None
	For every image dataset, extract it and make xml file with labelGen function

	"""
	check_folder(train_annot_folder)
	check_folder(val_annot_folder)

	result_images = image_with_label(image_folder, label_folder)
	dataset = dataset_split(result_images)

	for image in dataset["train"]:
		label_name = image.split(".")[0]+".txt"
		label_path = label_folder+"\\"+label_name
		xml_name = image.split(".")[0]+".xml"
		xml_path = train_annot_folder+xml_name
		# print(label_path, xml_path)
		labelGen(label_path, xml_path, "train", image)

	for image in dataset["val"]:
		label_name = image.split(".")[0]+".txt"
		label_path = label_folder+"\\"+label_name
		xml_name = image.split(".")[0]+".xml"
		xml_path = val_annot_folder+xml_name
		labelGen(label_path,xml_path,"val",image)


def main():
	# func1(image_folder)
	# func2()
	# func3() 


if __name__ == '__main__':
	main()


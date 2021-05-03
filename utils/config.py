# 定义参数
LABELS           = ('ball') # xml文件中存在的物体类别
IMAGE_H, IMAGE_W = 512, 512
GRID_H,  GRID_W  = 16,16 # GRID size = IMAGE size / 32
BOX              = 5
CLASS            = len(LABELS)
SCORE_THRESHOLD  = 0.5
IOU_THRESHOLD    = 0.45
ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

TRAIN_BATCH_SIZE = 10
VAL_BATCH_SIZE   = 10
EPOCHS           = 50

LAMBDA_NOOBJECT  = 1
LAMBDA_OBJECT    = 5
LAMBDA_CLASS     = 1
LAMBDA_COORD     = 1

max_annot        = 0

# Train and validation directory
train_image_folder = 'data/train/image/'
train_annot_folder = 'data/train/annotation/'
val_image_folder = 'data/val/image/'
val_annot_folder = 'data/val/annotation/'
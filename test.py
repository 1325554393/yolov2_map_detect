from utils.predict import *
from utils.model import model
import os

weights_path='./weights_best_of_500_epochs.h5'

try:
    model.load_weights(weights_path)
    print('weights加载成功')
except:
    print('weights加载失败，使用随机weights')


data_dir = './data/val/'  # 测试集地址
txt_dir = './map/'     # 结果（txt文件）的保存地址


if not os.path.exists(os.path.join(txt_dir, 'dr/')):
        os.makedirs(os.path.join(txt_dir, 'dr/'))
if not os.path.exists(os.path.join(txt_dir, 'gt/')):
        os.makedirs(os.path.join(txt_dir, 'gt/'))
        


if __name__ == '__main__':
    gen_results_txt(data_dir, txt_dir, model, 0, 0.3) #第4个参数应为0（之前是0.45）

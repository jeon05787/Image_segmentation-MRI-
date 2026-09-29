import os
import random
import cv2

# Tensorflow 관련 디버그 및 경고 메시지 비활성화 (삭제 금지)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import tensorflow.image as tfi
from tensorflow.keras import Sequential
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from PIL import Image, ImageDraw
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 폴더 경로 설정
data_path = '/kaggle/input/competitions/2026-DAU-CV'

# 재구현 세팅
def init_seeds(seed):
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    np.random.seed(seed)
    random.seed(seed)

init_seeds(2026)

# 데이터 로드
train_image_path = os.path.join(data_path, 'train/images')
train_label_path = os.path.join(data_path, 'train/masks')
test_image_path = os.path.join(data_path, 'test/images')

output_path = '/kaggle/working'

train_images = os.listdir(train_image_path)
train_images = [os.path.join(train_image_path, x) for x in train_images]
train_labels = os.listdir(train_label_path)
train_labels = [os.path.join(train_label_path, x) for x in train_labels]

train_images.sort(), train_labels.sort()

test_images = os.listdir(test_image_path)
test_images = [os.path.join(test_image_path, x) for x in test_images]

test_images.sort()

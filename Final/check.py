import os
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt

test_img=cv2.imread('bandana.jpg')
plt.imshow(test_img)

test_img=cv2.resize(test_img,(224,224))

test_input=test_img.reshape((1,224,224,3))

model=load_model('final_face.keras')

y_pred=(model.predict(test_input))

if (y_pred <0.5):
    print("Mask Off")
else:
    print("Mask On")


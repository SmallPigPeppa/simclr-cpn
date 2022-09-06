import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
from tensorflow.keras import layers

DIVIDE_STEPS=10
BATCH_SIZE=256
LR=0.0001
NUM_EPOCHS=100000
PATH='/share/lwz/simclr_model/r50/saved_model/0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# load pretrained model
pretrained_model=tf.saved_model.load(PATH)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)/255.
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)/255.
x_train_pretrained = np.zeros([50000, 2048])
x_test_pretrained = np.zeros([10000, 2048])
'''
resnet 18:512
resnet 50:2048
resnet 50(2x):4096
'''
len_train = x_train.shape[0] // DIVIDE_STEPS
len_test = x_test.shape[0] // DIVIDE_STEPS
for i in tqdm(range(DIVIDE_STEPS)):
    # if(i==1):
    #     break
    index1 = int(i * len_train)
    index2 = int(i * len_test)
    if i == DIVIDE_STEPS - 1:
        index3 = -1
        index4 = -1
    else:
        index3 = int((i + 1) * len_train)
        index4 = int((i + 1) * len_test)
    x_train_i = x_train[index1:index3]
    x_test_i = x_test[index2:index4]
    x_train_i = pretrained_model(x_train_i, trainable=False)
    x_test_i = pretrained_model(x_test_i, trainable=False)
    x_train_pretrained[index1:index3] = x_train_i['final_avg_pool'].numpy()
    x_test_pretrained[index2:index4] = x_test_i['final_avg_pool'].numpy()

np.save('data_pretrained/r50/x_train', x_train_pretrained)
np.save('data_pretrained/r50/x_test', x_test_pretrained)
np.save('data_pretrained/r50/y_train', y_train)
np.save('data_pretrained/r50/y_test', y_test)

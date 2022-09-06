import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
from tensorflow.keras import layers

DIVIDE_STEPS=10
BATCH_SIZE=256
LR=0.0001
NUM_EPOCHS=100000
PATH='/data/home/wzliu/code/simclr32_3/r18_cifar10/saved_model/0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
strategy = tf.distribute.MirroredStrategy()

# load pretrained model
pretrained_model=tf.saved_model.load(PATH)

with strategy.scope():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data('/data/home/wzliu/tensorflow_datasets')
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)/255.
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)/255.
    x_train_pretrained = np.zeros([50000, 512])
    x_test_pretrained = np.zeros([10000, 512])
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
    #create new datasets
    dataset_train = tf.data.Dataset.from_tensor_slices((x_train_pretrained, y_train))
    dataset_test = tf.data.Dataset.from_tensor_slices((x_test_pretrained, y_test))
    dataset_test = dataset_test.batch(BATCH_SIZE)
    dataset_train = dataset_train.batch(BATCH_SIZE)

    # create linear model and fit
    linear_model=tf.keras.models.Sequential([tf.keras.layers.Dense(units=10)])
    linear_model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
linear_model.fit(
    dataset_train,
    epochs=NUM_EPOCHS,
    validation_data=dataset_test,
)



import resource

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))
import tensorflow as tf
import numpy as np
import os
import tensorflow_datasets as tfds
import random
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CROP_PROPORTION = 0.875
IMG_SIZE = 224
LABELS100 = [233, 236, 246, 249, 258, 272, 280, 284, 286, 303, 305, 307, 309, 310, 354, 361, 371, 379, 380, 506, 514,
             515,
             518, 520, 525, 527, 529, 530, 533, 541, 548, 554, 556, 558, 560, 564, 568, 569, 571, 574, 576, 577, 585,
             588,
             589, 592, 667, 674, 675, 680, 691, 693, 721, 723, 725, 730, 746, 757, 775, 779, 782, 785, 793, 799, 808,
             809,
             827, 836, 841, 845, 849, 854, 857, 861, 876, 882, 887, 900, 902, 922, 924, 925, 926, 930, 937, 941, 942,
             943,
             944, 956, 959, 963, 975, 977, 982, 985, 988, 993, 996, 997]
LABELS100_TF = tf.convert_to_tensor(LABELS100, dtype=tf.int64)


def map_func(img, label):
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.reshape(img, [IMG_SIZE, IMG_SIZE, 3])
    img = tf.clip_by_value(img, 0., 1.)
    return img, label


if __name__ == '__main__':
    # encoder
    ckpt_path = "/share/lwz/simclr_model/r50_google/saved_model"
    data_path = "/share/datasets/torch_ds/imagenet-subset"
    dataset = "imagenet-subset"
    batch_size = 512
    image_size = 224
    LR = 0.0001
    epochs = 100
    # pretrained_model = tf.saved_model.load(ckpt_path)
    # train_dataset = tf.keras.utils.image_dataset_from_directory(
    #     directory=os.path.join(data_path, "train"),
    #     batch_size=batch_size,
    #     image_size=(256, 256),
    # )
    # test_dataset = tf.keras.utils.image_dataset_from_directory(
    #     directory=os.path.join(data_path, "val"),
    #     batch_size=batch_size,
    #     image_size=(256, 256),
    # )
    # train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    # test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    #
    # transforms = tf.keras.layers.CenterCrop(
    #     height=IMG_SIZE, width=IMG_SIZE
    # )
    # train_dataset = train_dataset.map(lambda x, y: (transforms(x), y))
    # test_dataset = test_dataset.map(lambda x, y: (transforms(x), y))
    # train_dataset = train_dataset.map(lambda x, y: (tf.image.convert_image_dtype(x, dtype=tf.float32), y))
    # test_dataset = test_dataset.map(lambda x, y: (tf.image.convert_image_dtype(x, dtype=tf.float32), y))
    # train_dataset = train_dataset.map(lambda x, y: (x / 255., y))
    # test_dataset = test_dataset.map(lambda x, y: (x / 255., y))
    #
    # x_train = np.empty((0, 2048))
    # y_train = np.empty((0))
    # for (imgs, labels) in tqdm(train_dataset):
    #     x_i = pretrained_model(imgs, trainable=False)['final_avg_pool'].numpy()
    #     x_train = np.append(x_train, x_i, axis=0)
    #     # y_train = np.append(y_train, vmapfunc(labels.numpy()), axis=0)
    #     y_train = np.append(y_train, labels.numpy(), axis=0)
    # print("x_train.shape:", x_train.shape, "\ny_train.shape:", y_train.shape)
    #
    # x_test = np.empty((0, 2048))
    # y_test = np.empty((0))
    # for (imgs, labels) in tqdm(test_dataset):
    #     x_i = pretrained_model(imgs, trainable=False)['final_avg_pool'].numpy()
    #     x_test = np.append(x_test, x_i, axis=0)
    #     # y_test = np.append(y_test, vmapfunc(labels.numpy()), axis=0)
    #     y_test = np.append(y_test, labels.numpy(), axis=0)
    # print("x_test.shape:", x_test.shape, "\ny_test.shape:", y_test.shape)
    #
    # os.makedirs(f'data_pretrained/{dataset}/', exist_ok=True)
    # np.save(f'data_pretrained/{dataset}/x_train', x_train)
    # np.save(f'data_pretrained/{dataset}/x_test', x_test)
    # np.save(f'data_pretrained/{dataset}/y_train', y_train)
    # np.save(f'data_pretrained/{dataset}/y_test', y_test)

    x_train = np.load(f'data_pretrained/{dataset}/x_train.npy')
    x_test = np.load(f'data_pretrained/{dataset}/x_test.npy')
    y_train = np.load(f'data_pretrained/{dataset}/y_train.npy')
    y_test = np.load(f'data_pretrained/{dataset}/y_test.npy')
    print(x_test.shape, y_test.shape)
    print(x_train.shape, y_train.shape)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(batch_size)
    train_ds = train_ds.batch(batch_size)

    # create linear model and fit
    linear_model = tf.keras.models.Sequential(
        tf.keras.layers.Dense(units=100))
    linear_model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])

    linear_model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds,
    )

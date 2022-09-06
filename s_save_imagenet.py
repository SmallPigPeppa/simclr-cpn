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


def _compute_crop_shape(
        image_height, image_width, aspect_ratio, crop_proportion):
    """Compute aspect ratio-preserving shape for central crop.

    The resulting shape retains `crop_proportion` along one side and a proportion
    less than or equal to `crop_proportion` along the other side.

    Args:
      image_height: Height of image to be cropped.
      image_width: Width of image to be cropped.
      aspect_ratio: Desired aspect ratio (width / height) of output.
      crop_proportion: Proportion of image to retain along the less-cropped side.

    Returns:
      crop_height: Height of image after cropping.
      crop_width: Width of image after cropping.
    """
    image_width_float = tf.cast(image_width, tf.float32)
    image_height_float = tf.cast(image_height, tf.float32)

    def _requested_aspect_ratio_wider_than_image():
        crop_height = tf.cast(
            tf.math.rint(crop_proportion / aspect_ratio * image_width_float),
            tf.int32)
        crop_width = tf.cast(
            tf.math.rint(crop_proportion * image_width_float), tf.int32)
        return crop_height, crop_width

    def _image_wider_than_requested_aspect_ratio():
        crop_height = tf.cast(
            tf.math.rint(crop_proportion * image_height_float), tf.int32)
        crop_width = tf.cast(
            tf.math.rint(crop_proportion * aspect_ratio * image_height_float),
            tf.int32)
        return crop_height, crop_width

    return tf.cond(
        aspect_ratio > image_width_float / image_height_float,
        _requested_aspect_ratio_wider_than_image,
        _image_wider_than_requested_aspect_ratio)


def center_crop(image, height, width, crop_proportion):
    """Crops to center of image and rescales to desired size.

    Args:
      image: Image Tensor to crop.
      height: Height of image to be cropped.
      width: Width of image to be cropped.
      crop_proportion: Proportion of image to retain along the less-cropped side.

    Returns:
      A `height` x `width` x channels Tensor holding a central crop of `image`.
    """
    shape = tf.shape(image)
    image_height = shape[0]
    image_width = shape[1]
    crop_height, crop_width = _compute_crop_shape(
        image_height, image_width, height / width, crop_proportion)
    offset_height = ((image_height - crop_height) + 1) // 2
    offset_width = ((image_width - crop_width) + 1) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_height, offset_width, crop_height, crop_width)

    image = tf.image.resize([image], [height, width],
                            method=tf.image.ResizeMethod.BICUBIC)[0]

    return image


# def map_func(img, label):
#     img = tf.image.convert_image_dtype(img, dtype=tf.float32)
#     img = center_crop(img, IMG_SIZE, IMG_SIZE, crop_proportion=CROP_PROPORTION)
#     img = tf.reshape(img, [IMG_SIZE, IMG_SIZE, 3])
#     img = tf.clip_by_value(img, 0., 1.)
#     return img, label
def map_func(img, label):
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    # img = center_crop(img, IMG_SIZE, IMG_SIZE, crop_proportion=CROP_PROPORTION)
    img = tf.reshape(img, [IMG_SIZE, IMG_SIZE, 3])
    img = tf.clip_by_value(img, 0., 1.)
    # return img, label
    print(img)
    return img, label


# def mapfunc(label):
#     return np.where(label == np.array(LABELS100))[0]


# vmapfunc = np.vectorize(mapfunc)

if __name__ == '__main__':
    # encoder
    ckpt_path = "/share/lwz/simclr_model/r50_google/saved_model"
    data_path = "/share/datasets/torch_ds/imagenet-subset"
    dataset = "imagenet-subset"
    batch_size = 512
    image_size = 224
    lr = 0.0001
    epochs = 100
    pretrained_model = tf.saved_model.load(ckpt_path)
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=os.path.join(data_path, "train"),
        batch_size=1,
        image_size=(256, 256),
    )
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=os.path.join(data_path, "val"),
        batch_size=1,
        image_size=(256, 256),
    )
    train_dataset = train_dataset.map(map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.map(map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    x_train = np.empty((0, 2048))
    y_train = np.empty((0))
    for (imgs, labels) in tqdm(train_dataset):
        x_i = pretrained_model(imgs, trainable=False)['final_avg_pool'].numpy()
        x_train = np.append(x_train, x_i, axis=0)
        # y_train = np.append(y_train, vmapfunc(labels.numpy()), axis=0)
        y_train = np.append(y_train, labels.numpy(), axis=0)
    print("x_train.shape:", x_train.shape, "\ny_train.shape:", y_train.shape)

    x_test = np.empty((0, 2048))
    y_test = np.empty((0))
    for (imgs, labels) in tqdm(test_dataset):
        x_i = pretrained_model(imgs, trainable=False)['final_avg_pool'].numpy()
        x_test = np.append(x_test, x_i, axis=0)
        # y_test = np.append(y_test, vmapfunc(labels.numpy()), axis=0)
        y_test = np.append(y_test, labels.numpy(), axis=0)
    print("x_test.shape:", x_test.shape, "\ny_test.shape:", y_test.shape)

    np.save(f'data_pretrained/{dataset}/x_train', x_train)
    np.save(f'data_pretrained/{dataset}/x_test', x_test)
    np.save(f'data_pretrained/{dataset}/y_train', y_train)
    np.save(f'data_pretrained/{dataset}/y_test', y_test)

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
    linear_model = tf.keras.models.Sequential(tf.keras.layers.Dense(units=100))
    linear_model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])

    linear_model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds,
    )

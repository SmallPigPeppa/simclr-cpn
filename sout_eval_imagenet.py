import resource

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))
import tensorflow as tf
import numpy as np
import os
import tensorflow_datasets as tfds

os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3,4,5'
CROP_PROPORTION = 0.875
CIFAR100_LABEL = [5, 6, 52, 60, 61, 75, 84, 95, 103, 107, 120, 156, 187, 189,
                  193, 204, 213, 214, 215, 249, 254, 280, 286, 288, 298, 303,
                  310, 316, 373, 441, 457, 462, 473, 474, 476,
                  521, 523, 542, 595, 601, 612, 616, 619, 620, 640, 652, 700,
                  734, 745, 821, 828, 881, 886, 919, 943, 954, 957, 958]
CIFAR100_LABEL = tf.convert_to_tensor(CIFAR100_LABEL, dtype=tf.int64)


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


def map_func(img, label):
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = center_crop(img, IMG_SIZE, IMG_SIZE, crop_proportion=CROP_PROPORTION)
    img = tf.reshape(img, [IMG_SIZE, IMG_SIZE, 3])
    img = tf.clip_by_value(img, 0., 1.)
    return img, label


if __name__ == '__main__':
    # strategy = tf.distribute.MirroredStrategy()

    # load pretrained model
    DIVIDE_STEPS = 10
    IMG_SIZE = 64
    BATCH_SIZE = 5000
    LR = 0.001
    NUM_EPOCHS = 100000
    PATH = '/data/home/wzliu/code/simclr32_3/r50_64/saved_model/0'
    # with strategy.scope():
    pretrained_model = tf.saved_model.load(PATH)
    train_ds = tfds.load(name="imagenet_resized/64x64", as_supervised=True, split="train",
                         data_dir='/data/home/wzliu/tensorflow_datasets')
    train_ds = train_ds.filter(lambda image, label: tf.reduce_any(tf.equal(label, CIFAR100_LABEL)))
    train_ds = train_ds.map(map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = tfds.load(name="imagenet_resized/64x64", as_supervised=True, split="validation",
                        data_dir='/data/home/wzliu/tensorflow_datasets')
    test_ds = test_ds.filter(lambda image, label: tf.reduce_any(tf.equal(label, CIFAR100_LABEL)))
    test_ds = test_ds.map(map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE)
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
    train_r=np.empty((0, 2048))
    train_y = np.empty((0))
    progbar_train = tf.keras.utils.Progbar(190254/BATCH_SIZE)
    for idx,(imgs,labels) in enumerate(train_ds):
        r_i= pretrained_model(imgs, trainable=False)['final_avg_pool'].numpy()
        train_r=np.append(train_r, r_i, axis=0)
        train_y = np.append(train_y, labels, axis=0)
        progbar_train.update(idx)
    print(train_r.shape,train_y.shape)

    test_r=np.empty((0, 2048))
    test_y = np.empty((0))
    progbar_test = tf.keras.utils.Progbar(7350/BATCH_SIZE)
    for idx,(imgs,labels) in enumerate(test_ds):
        r_i= pretrained_model(imgs, trainable=False)['final_avg_pool'].numpy()
        test_r=np.append(test_r, r_i, axis=0)
        test_y = np.append(test_y, labels, axis=0)
        progbar_test.update(idx)
    print(test_r.shape,test_y.shape)


    train_ds = tf.data.Dataset.from_tensor_slices((train_r, train_y))
    test_ds = tf.data.Dataset.from_tensor_slices((test_r, test_y))
    test_ds = test_ds.batch(BATCH_SIZE)
    train_ds = train_ds.batch(BATCH_SIZE)

    # create linear model and fit
    linear_model=tf.keras.models.Sequential(tf.keras.layers.Dense(units=1000))
    linear_model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    linear_model.fit(
        train_ds,
        epochs=NUM_EPOCHS,
        validation_data=test_ds,
    )

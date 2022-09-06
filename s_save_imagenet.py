import resource

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))
import tensorflow as tf
import numpy as np
import os
import tensorflow_datasets as tfds
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CROP_PROPORTION = 0.875
# CROP_PROPORTION = 1.
# CIFAR100_LABEL = [5, 6, 52, 60, 61, 75, 84, 95, 103, 107, 120, 156, 187, 189,
#                   193, 204, 213, 214, 215, 249, 254, 280, 286, 288, 298, 303,
#                   310, 316, 373, 441, 457, 462, 473, 474, 476,
#                   521, 523, 542, 595, 601, 612, 616, 619, 620, 640, 652, 700,
#                   734, 745, 821, 828, 881, 886, 919, 943, 954, 957, 958]
LABELS100=[233, 236, 246, 249, 258, 272, 280, 284, 286, 303, 305, 307, 309, 310, 354, 361, 371, 379, 380, 506, 514, 515,
           518, 520, 525, 527, 529, 530, 533, 541, 548, 554, 556, 558, 560, 564, 568, 569, 571, 574, 576, 577, 585, 588,
           589, 592, 667, 674, 675, 680, 691, 693, 721, 723, 725, 730, 746, 757, 775, 779, 782, 785, 793, 799, 808, 809,
           827, 836, 841, 845, 849, 854, 857, 861, 876, 882, 887, 900, 902, 922, 924, 925, 926, 930, 937, 941, 942, 943,
           944, 956, 959, 963, 975, 977, 982, 985, 988, 993, 996, 997]
CIFAR100_LABEL=[0, 5, 6, 12, 21, 23, 27, 29, 33, 52, 57, 60, 61, 66, 75, 84, 95, 101, 102,
                103, 107, 120, 128, 135, 152, 154, 156, 158, 161, 162, 163, 164, 182, 187,
                189, 193, 194, 201, 204, 205, 208, 213, 214, 215, 249, 253, 254, 256, 261, 280,
                286, 288, 298, 303, 310, 316, 373, 441, 442, 443, 457, 458, 459, 460, 461, 462,
                463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479,
                480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 521, 523, 542, 595,
                601, 603, 604, 605, 606, 607, 608, 609, 612, 613, 614, 615, 616, 617, 618, 619, 620,
                621, 622, 623, 624, 625, 626, 629, 640, 641, 642, 643, 644, 645, 652, 653, 679, 680,
                681, 700, 734, 745, 821, 828, 881, 886, 919, 943, 954, 957, 958]
# LABEL_100=list(range(1000))
# random.shuffle(CIFAR100_LABEL)
# random.shuffle(LABEL_100)
# print(LABEL_100[:100])
CIFAR100_LABEL = tf.convert_to_tensor(CIFAR100_LABEL, dtype=tf.int64)
LABELS100_TF=tf.convert_to_tensor(LABELS100, dtype=tf.int64)

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
    IMG_SIZE = 224
    BATCH_SIZE = 512
    LR = 0.0001
    NUM_EPOCHS = 100000
    PATH = '/share/lwz/simclr_model/r50_imagenet/saved_model/2'
    pretrained_model = tf.saved_model.load(PATH)
    train_ds = tfds.load(name="imagenet2012", as_supervised=True, split="train",
                         data_dir='/share/datasets/tensorflow_datasets')
    train_ds = train_ds.filter(lambda image, label: tf.reduce_any(tf.equal(label, LABELS100_TF)))
    train_ds = train_ds.map(map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = tfds.load(name="imagenet2012", as_supervised=True, split="validation",
                        data_dir='/share/datasets/tensorflow_datasets')
    test_ds = test_ds.filter(lambda image, label: tf.reduce_any(tf.equal(label, LABELS100_TF)))
    test_ds = test_ds.map(map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE)
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
    train_r=np.empty((0, 2048))
    train_y = np.empty((0))
    progbar_train = tf.keras.utils.Progbar(200000/BATCH_SIZE)
    def mapfunc(label):
        return np.where(label==np.array(LABELS100))[0]
    vmapfunc=np.vectorize(mapfunc)

    for idx,(imgs,labels) in enumerate(train_ds):
        r_i= pretrained_model(imgs, trainable=False)['final_avg_pool'].numpy()
        train_r=np.append(train_r, r_i, axis=0)
        train_y = np.append(train_y, vmapfunc(labels.numpy()), axis=0)
        progbar_train.update(idx)
    print(train_r.shape,train_y.shape)

    test_r=np.empty((0, 2048))
    test_y = np.empty((0))
    progbar_test = tf.keras.utils.Progbar(10000/BATCH_SIZE)
    for idx,(imgs,labels) in enumerate(test_ds):
        r_i= pretrained_model(imgs, trainable=False)['final_avg_pool'].numpy()
        test_r=np.append(test_r, r_i, axis=0)
        test_y = np.append(test_y, vmapfunc(labels.numpy()), axis=0)
        progbar_test.update(idx)
    print(test_r.shape,test_y.shape)


    np.save('data_pretrained/imagenet100_exclude/x_train', train_r)
    np.save('data_pretrained/imagenet100_exclude/x_test', test_r)
    np.save('data_pretrained/imagenet100_exclude/y_train', train_y)
    np.save('data_pretrained/imagenet100_exclude/y_test', test_y)
    x_train = np.load('data_pretrained/imagenet100_exclude/x_train.npy')
    x_test = np.load('data_pretrained/imagenet100_exclude/x_test.npy')
    y_train = np.load('data_pretrained/imagenet100_exclude/y_train.npy')
    y_test = np.load('data_pretrained/imagenet100_exclude/y_test.npy')
    print(x_test.shape, y_test.shape)
    print(x_train.shape, y_train.shape)
    '''
    # map new label
    a=[233, 236, 246, 249, 258, 272, 280, 284, 286, 303, 305, 307, 309, 310, 354, 361, 371, 379, 380, 506, 514, 515,
           518, 520, 525, 527, 529, 530, 533, 541, 548, 554, 556, 558, 560, 564, 568, 569, 571, 574, 576, 577, 585, 588,
           589, 592, 667, 674, 675, 680, 691, 693, 721, 723, 725, 730]
    # a=np.array([730])
    newa=vmapfunc(a)
    print(newa)
    '''









    #
    # #
    # #
    # train_ds = tf.data.Dataset.from_tensor_slices((train_r, train_y))
    # test_ds = tf.data.Dataset.from_tensor_slices((test_r, test_y))
    # test_ds = test_ds.batch(BATCH_SIZE)
    # train_ds = train_ds.batch(BATCH_SIZE)
    #
    # # create linear model and fit
    # linear_model=tf.keras.models.Sequential(tf.keras.layers.Dense(units=100))
    # linear_model.compile(optimizer=tf.keras.optimizers.Adam(LR),
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    #
    # linear_model.fit(
    #     train_ds,
    #     epochs=NUM_EPOCHS,
    #     validation_data=test_ds,
    # )

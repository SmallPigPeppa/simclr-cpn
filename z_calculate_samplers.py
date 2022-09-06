import tensorflow as tf
import tensorflow_datasets as tfds
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
CIFAR100_LABEL=[0, 5, 6, 12, 21, 23, 27, 29, 33, 52, 57, 60, 61, 66, 75, 84, 95, 101, 102,
                103, 107, 120, 128, 135, 152, 154, 156, 158, 161, 162, 163, 164, 182, 187,
                189, 193, 194, 201, 204, 205, 208, 213, 214, 215, 249, 253, 254, 256, 261, 280,
                286, 288, 298, 303, 310, 316, 373, 441, 442, 443, 457, 458, 459, 460, 461, 462,
                463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479,
                480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 521, 523, 542, 595,
                601, 603, 604, 605, 606, 607, 608, 609, 612, 613, 614, 615, 616, 617, 618, 619, 620,
                621, 622, 623, 624, 625, 626, 629, 640, 641, 642, 643, 644, 645, 652, 653, 679, 680,
                681, 700, 734, 745, 821, 828, 881, 886, 919, 943, 954, 957, 958]
SUBLABELS=[233, 236, 246, 249, 258, 272, 280, 284, 286, 303, 305, 307, 309, 310, 354, 361, 371, 379, 380, 506, 514, 515,
           518, 520, 525, 527, 529, 530, 533, 541, 548, 554, 556, 558, 560, 564, 568, 569, 571, 574, 576, 577, 585, 588,
           589, 592, 667, 674, 675, 680, 691, 693, 721, 723, 725, 730, 746, 757, 775, 779, 782, 785, 793, 799, 808, 809,
           827, 836, 841, 845, 849, 854, 857, 861, 876, 882, 887, 900, 902, 922, 924, 925, 926, 930, 937, 941, 942, 943,
           944, 956, 959, 963, 975, 977, 982, 985, 988, 993, 996, 997]
SUBLABELS=tf.convert_to_tensor(SUBLABELS,dtype=tf.int64)
CIFAR100_LABEL=tf.convert_to_tensor(CIFAR100_LABEL,dtype=tf.int64)
builder = tfds.builder('imagenet2012',data_dir='/share/datasets/tensorflow_datasets')
builder.download_and_prepare()
dataset = builder.as_dataset(
        split='train',
        shuffle_files=True,
        as_supervised=True,
        # Passing the input_context to TFDS makes TFDS read different parts
        # of the dataset on different workers. We also adjust the interleave
        # parameters to achieve better performance.
        read_config=tfds.ReadConfig(
            interleave_cycle_length=32,
            interleave_block_length=1))
dataset= dataset.filter(lambda image,label: tf.reduce_all(tf.not_equal(label, SUBLABELS)))
i=0
for item in iter(dataset):
    i=i+1
print(i)
# item=next(iter(dataset))
# image=item[0]
# tf.keras.utils.save_img(path='X2.JPEG', x=image.numpy(), scale=True)
# print(item[1])
# print(tf.reduce_all(tf.not_equal(1, label)))
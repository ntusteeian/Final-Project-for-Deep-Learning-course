from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import scipy.io
import os
import sys
import scipy.misc
import glob
import math
sys.path.append("..")
from matplotlib.pyplot import savefig
import matplotlib.pyplot as plt

from utils.utils import *

REWIDTH, REHEIGHT, CHANNEL = 400, 400, 3
MIN_SCALE = 0.5
MAX_SCALE = 2.0

# src_dir = "D:\PASCAL\pascal-voc-2012\VOC2012"
src_dir = "/data/pascal/VOCdevkit/VOC2012/"
# voc_argu_dir = "D:\PASCAL\\benchmark\\benchmark_RELEASE\dataset"
voc_argu_dir = "/data/pascal/benchmark_RELEASE/dataset/"
# unlabel_testing_dir = "D:\PASCAL\\testimage\\testimage"
unlabel_testing_dir = "/data/pascal/testimage/testimage/"

labels_name = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def read_train_val_txt():
    target_dir = 'ImageSets/Segmentation'
    train_txt = "train.txt"
    val_txt = "val.txt"

    train_txt_path = os.path.join(src_dir, target_dir, train_txt)
    val_txt_path = os.path.join(src_dir, target_dir, val_txt)
    argu_train_txt_path = os.path.join(voc_argu_dir, train_txt)
    argu_val_txt_path = os.path.join(voc_argu_dir, val_txt)

    train = pd.read_csv(train_txt_path)
    train = train.values.flatten()
    train_list = train.tolist()

    val = pd.read_csv(val_txt_path)
    val = val.values.flatten()
    val_list = val.tolist()

    argu_train = pd.read_csv(argu_train_txt_path)
    argu_train = argu_train.values.flatten()
    argu_train_list = argu_train.tolist()

    argu_val = pd.read_csv(argu_val_txt_path)
    argu_val = argu_val.values.flatten()
    argu_val_list = argu_val.tolist()

    return train_list, val_list, argu_train_list, argu_val_list


def get_path_dict():
    image_dir, label_dir = 'JPEGImages', 'SegmentationClass'
    argu_image_dir, argu_label_dir = 'img', 'class_mat_to_png'

    train_list, val_list, argu_train_list, argu_val_list = read_train_val_txt()
    path = {}
    path['t_image'], path['t_label'] = [], []
    path['v_image'], path['v_label'] = [], []
    path['argu_t_image'], path['argu_t_label'] = [], []
    path['argu_v_image'], path['argu_v_label'] = [], []

    # pascal 2012
    for idx, filename in enumerate(train_list):
        path['t_image'].append(os.path.join(src_dir, image_dir, filename + '.jpg'))
        path['t_label'].append(os.path.join(src_dir, label_dir, filename + '.png'))

    for idx, filename in enumerate(val_list):
        path['v_image'].append(os.path.join(src_dir, image_dir, filename + '.jpg'))
        path['v_label'].append(os.path.join(src_dir, label_dir, filename + '.png'))

    # argumented benchmark
    for idx, filename in enumerate(argu_train_list):
        path['argu_t_image'].append(os.path.join(voc_argu_dir, argu_image_dir, filename + '.jpg'))
        path['argu_t_label'].append(os.path.join(voc_argu_dir, argu_label_dir, filename + '.png'))

    for idx, filename in enumerate(argu_val_list):
        path['argu_v_image'].append(os.path.join(voc_argu_dir, argu_image_dir, filename + '.jpg'))
        path['argu_v_label'].append(os.path.join(voc_argu_dir, argu_label_dir, filename + '.png'))

    path['total_t_image'] = path['t_image'] + path['argu_t_image'] + path['argu_v_image'][:1200]
    path['total_v_image'] = path['v_image'] + path['argu_v_image'][1200:]
    path['total_t_label'] = path['t_label'] + path['argu_t_label'] + path['argu_v_label'][:1200]
    path['total_v_label'] = path['v_label'] + path['argu_v_label'][1200:]

    print('>>total training number: {}'.format(len(path['total_t_image'])))
    print('>>total validation number: {}'.format(len(path['total_v_image'])))

    return path


# using this method after final
# def create_dataset(path_dict):
#
#
#     image_path = tf.placeholder(tf.string, shape=[None])
#     label_path = tf.placeholder(tf.string, shape=[None])
#
#     dataset = tf.data.Dataset.from_tensor_slices((image_path, label_path))
#
#     dataset = dataset.map(image_parser)
#     dataset = dataset.shuffle(buffer_size=100)
#     dataset = dataset.batch(BATCH_SIZE)
#
#     iterator = dataset.make_initializable_iterator()
#     image, label = iterator.get_next()
#
#
#
#     # EPOCHS = 2
#     #
#     #     for epoc in range(EPOCHS):
#     #
#     #         avg_loss = 0
#     #         sess.run(iterator.initializer, feed_dict={image_path: path['t_image'], label_path: path['t_label']})
#     #         while True:
#     #
#     #             try:
#     #                 x, y = sess.run([image, label])
#     #                 print('debug')
#     #
#     #             except tf.errors.OutOfRangeError:
#     #
#     #                 break
#     #
#     #         sess.run(iterator.initializer, feed_dict={image_path: path['v_image'], label_path: path['v_label']})
#     #
#     #         while True:
#     #
#     #             try:
#     #                 x, y = sess.run([image, label])
#     #                 print('debug')
#     #
#     #             except tf.errors.OutOfRangeError:
#     #
#     #                 break
#
#     return image_path, label


def decode_labels(label_path):
    ignored = 255
    decoded_label = Image.open(label_path)
    decoded_label = np.array(decoded_label)[:, :, np.newaxis]

    decoded_label[decoded_label == ignored] = 0

    return decoded_label


def image_parser(image_path, label_path= None, exists_label=True):
    image_file = tf.read_file(image_path)
    decoded_image = tf.image.decode_jpeg(image_file)
    decoded_image /= 255
    decoded_image.set_shape([None, None, CHANNEL])

    resized_image = tf.image.resize_images(decoded_image, [REWIDTH, REHEIGHT])

    if exists_label:
        label_decoded = tf.py_func(decode_labels, [label_path], tf.uint8)
        label_decoded.set_shape([None, None, 1])
        label_decoded = tf.cast(label_decoded, tf.int32)
        # important
        label_resized = tf.image.resize_images(label_decoded, [REWIDTH, REHEIGHT],
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return resized_image, label_resized

    return resized_image


def get_infer_image(path_dict, exists_label=None):

    #if exists_label = False , path_dict is a path can directly input

    if exists_label:

        random_index = int(np.random.choice(len(path_dict['v_image']), 1))

        infer_path = path_dict['v_image'][random_index]
        infer_label_path = path_dict['v_label'][random_index]
        filename = infer_path[42:-4]

        infer_image, infer_label = image_parser(infer_path, infer_label_path, exists_label=True)
        infer_image = tf.cast(infer_image, dtype=tf.float32)
        infer_image = tf.reshape(infer_image, shape=[1, REWIDTH, REHEIGHT, CHANNEL])
        return infer_image, infer_label, filename

    else:
        dirname = path_dict[33:35]
        filename = path_dict[-6:-4]

        infer_image = image_parser(path_dict, exists_label=False)
        infer_image = tf.cast(infer_image, dtype=tf.float32)
        infer_image = tf.reshape(infer_image, shape=[1, REWIDTH, REHEIGHT, CHANNEL])
        return infer_image, dirname, filename


def random_flip_left_right_image_and_label(image, label):
    """Randomly flip an image and label horizontally (left to right).
    Args:
      image: A 3-D tensor of shape `[height, width, channels].`
      label: A 3-D tensor of shape `[height, width, 1].`

    Returns:
      A 3-D tensor of the same type and shape as `image`.
      A 3-D tensor of the same type and shape as `label`.
    """

    uniform_random = tf.random_uniform([], 0, 1.0)
    mirror_cond = tf.less(uniform_random, .5)

    image = tf.cond(mirror_cond, lambda: tf.reverse(image, [1]), lambda: image)
    label = tf.cond(mirror_cond, lambda: tf.reverse(label, [1]), lambda: label)

    return image, label

def random_rotate(image, label):

    uniform_random = tf.random_uniform([], 0, 2*math.pi)
    rotate_image = tf.contrib.image.rotate(image, uniform_random)
    rotate_label = tf.contrib.image.rotate(label, uniform_random)
    print("debug")
    return rotate_image, rotate_label

def random_change_brightness(image, label):

    image = tf.image.random_brightness(image, max_delta=0.2)

    return image, label



def data_argumentation(image, label):

    image, label = random_flip_left_right_image_and_label(image, label)
    image, label = random_rotate(image, label)
    image, label = random_change_brightness(image, label)

    return image, label

def mat2png(mat_file, key='GTcls'):
    mat = scipy.io.loadmat(mat_file, mat_dtype=True, squeeze_me=True, struct_as_record=False)

    return mat[key].Segmentation


def convert_mat2png(mat_files, output_path):
    # if not mat_files:
    #     help('Input directory does not contain any Matlab files!\n')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for mat in mat_files:

        filename = mat[42:-3] + 'png'
        numpy_img = mat2png(mat)
        pil_img = Image.fromarray(numpy_img)
        pil_img = label_to_color_image(np.array(pil_img))
        scipy.misc.imsave(output_path + filename, pil_img)


def modify_image_name(path, ext):
    return os.path.basename(path).split('.')[0] + '.' + ext
#
# if __name__ == '__main__':
#
#     path_dict = get_path_dict()

    # testing_path_list = get_unlabel_testing_path(unlabel_testing_dir)
    # for path in testing_path_list:
    #     infer_image, dirname, filename = get_infer_image(path, exists_label= False)

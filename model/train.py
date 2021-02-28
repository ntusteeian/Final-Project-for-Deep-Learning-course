import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
import time
sys.path.append("..")

from model import *
from preprocessing import *
from utils.utils import *

NUM_CLASS = 21
BATCH_SIZE = 12
EPOCHS = 300
LEARNING_RATE = 1e-4
MODE = "testing"
TEST_FROM_VALID = True

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def train(batch_size, learning_rate, epoch, path_dict, is_train = True):

    # dataset
    image_path = tf.placeholder(tf.string, shape=[None])
    label_path = tf.placeholder(tf.string, shape=[None])

    dataset = tf.data.Dataset.from_tensor_slices((image_path, label_path))
    dataset = dataset.map(image_parser)
    dataset = dataset.map(data_argumentation)

    dataset = dataset.shuffle(buffer_size = 600)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    image, label = iterator.get_next()


    # evaluate
    Unet = unet(NUM_CLASS, is_train)
    logits, softmax_output = Unet.model(image)
    label_eval = tf.squeeze(label, axis = 3)
    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels= label_eval, logits = logits))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
    pred_result = tf.cast(tf.argmax(softmax_output, axis = 3), tf.int32)

    mean_iou, confusion_matrix = tf.metrics.mean_iou(label_eval, pred_result, num_classes = 21)
    correct_pred = tf.equal(pred_result, label_eval)
    pixel_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    saver = tf.train.Saver()

    sess = tf.Session(config=config)
    # writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())  #if not implement, iou error

    # continue training
    ckpt = tf.train.latest_checkpoint('../ckpt/checkpoint0617_nobatch_argu_rota/')
    saver.restore(sess, ckpt)


    print('>> start training')

    for epoc in range(epoch):

        aver_t_loss, aver_v_loss = 0, 0
        aver_t_pa, aver_v_pa = 0, 0
        aver_t_miou, aver_v_miou =0, 0

        sess.run(iterator.initializer, feed_dict={image_path: path_dict['total_t_image'], label_path: path_dict['total_t_label']})

        num_iter = 0
        while True:
            try:
                _, training_loss, training_pa, _, training_miou = sess.run([optimizer, loss, pixel_acc, confusion_matrix, mean_iou])

                aver_t_loss += training_loss
                aver_t_pa += training_pa
                aver_t_miou += training_miou
                # print('debug')
                num_iter += 1

            except tf.errors.OutOfRangeError:
                aver_t_loss /= num_iter
                aver_t_pa /= num_iter
                aver_t_miou /= num_iter
                break

        sess.run(iterator.initializer, feed_dict={image_path: path_dict['total_v_image'], label_path: path_dict['total_v_label']})

        num_iter = 0
        while True:
            try:
                validation_loss, validation_pa, _,validation_miou = sess.run([loss, pixel_acc, confusion_matrix, mean_iou])
                aver_v_loss += validation_loss
                aver_v_pa += validation_pa
                aver_v_miou += validation_miou
                # print('debug')
                num_iter += 1

            except tf.errors.OutOfRangeError:
                aver_v_loss /= num_iter
                aver_v_pa /= num_iter
                aver_v_miou /= num_iter
                break

        print(' epoch:{} | training_loss:{:.8f}, training_pa:{:.8f}, training_miou:{:.8f} | validation_loss:{:.8f}, validation_pa:{:.8f}, validation_miou:{:.8f} | '
              .format(epoc, aver_t_loss, aver_t_pa, aver_t_miou, aver_v_loss, aver_v_pa, aver_v_miou))

        if epoc % 5 == 0:
            if not os.path.exists('../ckpt/checkpoint0618_nobatch_argu_rota/'):
                os.makedirs('../ckpt/checkpoint0618_nobatch_argu_rota/')
            saver.save(sess, '../ckpt/checkpoint0618_nobatch_argu_rota/' + str(epoc))

    print(">> finishing training")


def test(infer_image, filename, infer_label = None, is_train = False, dir_name = None):


    # continue training
    Unet = unet(NUM_CLASS, is_train)
    logits, softmax_output = Unet.model(infer_image)

    predict = tf.argmax(softmax_output, axis = 3)

    saver = tf.train.Saver()
    sess = tf.Session(config =config)
    sess.run(tf.global_variables_initializer())
    tf.reset_default_graph()
    ckpt = tf.train.latest_checkpoint('../ckpt/checkpoint0618_nobatch_argu_rota/')
    saver.restore(sess, ckpt)

    if infer_label is not None:
        input_image, input_label, pred_result = sess.run([infer_image, infer_label, predict])
        _, pa, _, miou = segmentation_evaluation(input_label, pred_result)
        print("pa: {},  miou: {}".format(pa, miou))
        save_result(input_image, pred_result, filename = filename, infer_label = input_label)
    else:
        input_image, pred_result = sess.run([infer_image, predict])
        save_result(input_image, pred_result, filename = filename, dir_name = dir_name)

    print(pred_result)

if __name__ == "__main__" :

    path_dict = get_path_dict()

    if MODE == "training":
        train(batch_size = BATCH_SIZE, learning_rate = LEARNING_RATE, epoch = EPOCHS, path_dict = path_dict)

    elif MODE == "testing":
        if TEST_FROM_VALID:
            infer_image, infer_label, filename = get_infer_image(path_dict, exists_label = True)
            test(infer_image, filename, infer_label)
        else:
            testing_path_list = get_unlabel_testing_path(unlabel_testing_dir)
            for path in testing_path_list:
                infer_image, dirname, filename = get_infer_image(path, exists_label = False)
                test(infer_image, filename, dir_name = dirname)
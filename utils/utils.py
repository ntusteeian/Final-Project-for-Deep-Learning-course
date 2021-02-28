import numpy as np
import os
import glob

from sklearn.metrics import confusion_matrix
from matplotlib.pyplot import savefig
import matplotlib.pyplot as plt
REWIDTH, REHEIGHT, CHANNEL = 400, 400, 3


def segmentation_evaluation(y_true, y_pred):
    # input is 2-D array
    y_true = y_true.reshape(REHEIGHT, REWIDTH)
    y_pred = y_pred.reshape(REHEIGHT, REWIDTH)

    conf = confusion_matrix(np.array(y_true).flatten(), np.array(y_pred).flatten())

    overal_pixel_accuracy = sum(np.diag(conf)) / np.sum(conf) * 100

    # print(conf)
    # print(overal_pixel_accuracy)

    class_accuracy = np.zeros(len(conf))

    for i in range(len(conf)):
        ground_truth_class = sum(conf[i, :])
        predicted_class = sum(conf[:, i])

        true_positive = conf[i][i]

        class_accuracy[i] = true_positive / (ground_truth_class + predicted_class - true_positive) * 100

    average_class_accuracy = sum(class_accuracy) / len(class_accuracy)

    # print(average_class_accuracy)


    return confusion_matrix, overal_pixel_accuracy, class_accuracy, average_class_accuracy

def save_result(infer_image, pred_result, filename, infer_label = None, dir_name = None):

    infer_image = infer_image.reshape(REWIDTH, REHEIGHT, CHANNEL)
    pred_result = pred_result.reshape(REWIDTH, REHEIGHT)
    pred_result = label_to_color_image(pred_result)

    if infer_label is not None:
        infer_label = infer_label.reshape(REWIDTH, REHEIGHT)
        infer_label = label_to_color_image(infer_label)

        plt.subplot(1, 3, 1)
        plt.title("input image")
        plt.axis('off')
        plt.imshow(infer_image)
        plt.subplot(1, 3, 2)
        plt.title("ground truth")
        plt.axis('off')
        plt.imshow(infer_label)
        plt.subplot(1, 3, 3)
        plt.title("predict result")
        plt.axis('off')
        plt.imshow(pred_result)
        plt.show()
        if not os.path.exists('../result/'):
            os.makedirs('../result')
        savefig('../result/{}'.format(filename))

    else:


        plt.subplot(1, 2, 1)
        plt.imshow(infer_image)
        plt.title("input image")
        plt.subplot(1, 2, 2)
        plt.title("predict result")
        plt.imshow(pred_result)
        # plt.show()

        if not os.path.exists('../result/{}'.format(dir_name)):
            os.makedirs('../result/{}'.format(dir_name))
        savefig('../result/{}/{}'.format(dir_name,filename))


def create_pascal_label_colormap():

  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """

  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):

    for channel in range(3):

      colormap[:, channel] |= ((ind >> channel) & 1) << shift

    ind >>= 3

  return colormap

def label_to_color_image(label):


  if label.ndim != 2:

    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  return colormap[label]

def get_unlabel_testing_path(unlabel_testing_dir):

    #return a path list

    class_dir = os.listdir(unlabel_testing_dir)
    testing_path = []
    for dir in class_dir:
        infer_path = []
        infer_path = os.path.join(unlabel_testing_dir, dir)
        infer_path_list = glob.glob(os.path.join(infer_path, '*.jpg'))
        for path in infer_path_list:
            testing_path.append(path)

    return testing_path



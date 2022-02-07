"""
@description:
    
"""
import multiprocessing
import datetime
import random
import pickle
import json
from itertools import cycle

import cv2
import numpy
import SimpleITK as sitk
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
import seaborn
import torch
import torchmetrics

import constants

def parse_flash_datetime(datetime_str):
    """
    @description:
        +) parse datetime which has format %m%d%Y
    """
    date_time = datetime.datetime.strptime(datetime_str, '%Y/%m/%d')
    return str(date_time.year) + str(date_time.month) + str(date_time.day)

def parse_score_datetime(datetime_str):
    """
    @description:
        +) parse datetime which has format %Y-%m-%d
    """
    date_time = datetime.datetime.strptime(datetime_str, '%Y-%m-%d')
    return str(date_time.year) + str(date_time.month) + str(date_time.day)

def calculate_auc(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    return metrics.auc(fpr, tpr)

def read_png(image_file_name):
    return io.imread(image_file_name)

def read_dcm_image(dcm_file_name):
    dcm_image = sitk.ReadImage(dcm_file_name)
    dcm_arr = sitk.GetArrayFromImage(dcm_image)
    if len(dcm_arr.shape) == 4:
        dcm_arr = dcm_arr[0, :, :, :]
    elif len(dcm_arr.shape) == 3:
        dcm_arr = dcm_arr[0, :, :]
    dcm_arr = Image.fromarray(dcm_arr)
    dcm_arr = numpy.array(dcm_arr)
    return dcm_arr

def write_png(image, image_file_name):
    image = image.astype(numpy.uint8)
    io.imsave(image_file_name, image, check_contrast=False)

def read_pkl(pkl_file_name):
    with open(pkl_file_name, 'rb') as file:
        image = pickle.load(file)
    return image

def dump_pkl(data, pkl_file_name):
    with open(pkl_file_name, 'wb') as file:
        pickle.dump(data, file)

def load_json(json_file_name):
    with open(json_file_name, 'r') as file:
            data = json.load(file)
    return data

def split_number(number):
    first = number // 2
    second = number - first
    return first, second

def to_cuda(batch):
    batch[constants.IMAGE] = batch[constants.IMAGE].to(constants.DEVICE)
    batch[constants.LABEL] = batch[constants.LABEL].to(constants.DEVICE)
    return batch

def dump_json(data_dict, json_file_name):
    with open(json_file_name, 'w') as file:
        json.dump(data_dict, file, indent=4)

def convert_tensor_to_image(tensor):
    image = tensor.cpu().numpy()
    image = numpy.transpose(image, (1, 2, 0))
    return image

def remove_multi_items(src_arr, ref_arr):
    """
    @description:
        -) Remove all items of ref_arr in src_arr
    """
    src_arr = numpy.asanyarray(src_arr)
    mask = numpy.ones(len(src_arr), dtype=bool)
    locations = list(map(lambda item : int(numpy.where(src_arr == item)[0]),\
                                                                    ref_arr))
    mask[locations] = False
    src_arr = src_arr[mask]
    return src_arr

def cal_simple_acc(preds, labels):
    return (preds == labels).mean()

def count_cpus():
    return multiprocessing.cpu_count()

def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)

def random_dist_numbers(start, stop, num_numbers):
    """
    @description:
        -) random unique distinct num_numbers of numbers in range
    @parameters:
        1) start, stop: 
            -) positive integers
            -) the random numbers must be in range(start, stop)
        2) num_numbers: positive integer number
    """
    if stop - start < num_numbers:
        offset = num_numbers // (stop - start) + 1
        numbers = list(range(start, stop)) * offset
    else:
        numbers = random.sample(range(start, stop), num_numbers)
    return numbers

def visualize_data_dist(labels, label_groups, class_names, title):
    """
    @description:
        -) title: training, validation or testing dataset
    """
    label_locations = numpy.arange(len(label_groups))
    label_means = [0] * len(class_names)
    for label in labels:
        for group in label_groups:
            if label == group:
                label_means[group] += 1
    width = 0.35

    fig, ax = plt.subplots()
    #rects = ax.bar(label_locations - width / 2, label_means, width)
    ax.set_ylabel('#Samples')
    ax.set_title(f"{title} Distribution")
    ax.set_xticks(label_locations)
    ax.legend()
    ax.bar(label_locations, label_means)
    fig.tight_layout()
    plt.savefig(f"logs/{title}_dist.png")

def encode_one_hot_vector(labels, num_classes):
    """
    @description:
    """
    encoder = numpy.eye(num_classes, dtype=numpy.int)
    one_hot_vectors = encoder[labels]
    return one_hot_vectors

def visualize_roc(labels, preds, model_name):
    """
    @description:
        +) drawing ROC curves for multiple classes classification
        +) compute ROC curve and ROC area for each class
    @parameters:
        +) labels:
                -) one-hot vector of the labels
                -) examples: [[0, 0, 1], [0, 1, 0]]
        +) preds:
                -) prediction scores
                -) not necessary to be in probability range [0 -> 1]
    """
    lw = 4 #line width
    labels = encode_one_hot_vector(labels, 2)
    preds = numpy.asanyarray(preds)
    labels = numpy.asanyarray(labels)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[0], tpr[0], _ = roc_curve(labels[:, 1], preds[:, 1])
    roc_auc[0] = auc(fpr[0], tpr[0])

    # Plot all ROC curves
    plt.figure()

    colors = ['aqua', 'darkorange']
    plt.plot(fpr[0], tpr[0], color=colors[0], lw=lw,
             label='ROC curve of positive class (area = {1:0.2f})'
             ''.format(0, roc_auc[0]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC of {model_name}'s performances")
    plt.legend(loc="lower right")
    plt.savefig(f'logs/{model_name}_roc.png')

def visualize_conf(labels, preds, num_classes, model_name, acc=None):
    """
    @description:
        +) visualize confusion matrix
    """
    seaborn.set(font_scale=3.5)
    confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=num_classes)
    with torch.no_grad():
        labels = torch.tensor(labels)
        preds = torch.tensor(preds)
        cf = confusion_matrix(preds, labels).numpy()
        cf = cf.astype(numpy.int)
    figure = seaborn.heatmap(cf, annot=True, cmap='Blues', fmt='g')
    if acc:
        title = f"{model_name} ACC: {acc}"
    else:
        title = f"{model_name}"
    plt.title(title, fontsize=20)
    figure.figure.savefig(f"logs/{title}_conf_matrix.png")

def plot_two_images(left_image, right_image, title):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    plt.title(title)
    _ = ax1.imshow(left_image)
    _ = ax2.imshow(right_image)
    plt.savefig(f"logs/figures/{title}.png")
    plt.close()

def convert_dcm_to_png(dcm_file_name, saved_png_file_name):
    dcm_image = read_dcm_image(dcm_file_name)
    write_png(dcm_image, saved_png_file_name)

def find_contours(mask):
    contours, _ = cv2.findContours(mask.astype(dtype=numpy.uint8),\
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def extract_bbox(contour):
    """
    @desc:
        - extract bounding box from contour
    """
    (x_min, y_min, width, height) = cv2.boundingRect(contour)
    return (x_min, y_min, width, height)

def crop_image(image, bbox):
    """
    @desc:
        - crop image based on bbox
    @paras:
        - bbox: x_min, y_min, width, height
    """
    x_min, y_min, width, height = bbox
    crop_image = image[y_min : y_min + height, x_min : x_min + width, :]
    return crop_image

def crop_mask_image(image, mask):
    """
    @desc:
        - crop image based on mask
    """
    mask_rgb = mask.copy()
    mask_rgb = mask_rgb.astype(dtype=numpy.uint8)
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_GRAY2BGR)
    crop = image * mask_rgb
    return crop

def get_time_now():
    now = datetime.datetime.now()
    return (now.year, now.month, now.day, now.hour, now.minute, now.second)

def cal_sensi_speci(labels, preds):
    """
    @desc:
        - Calculate specificity and sensitivity
    """
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp + fn)
    return (sensitivity, specificity)

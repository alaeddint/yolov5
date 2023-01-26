import numpy as np
import torch
from .model import Model
from torchvision.models import vgg
from utils import ClassAverages
import cv2
from utils.Math import calc_location
from utils.Plotting import plot_2d_box, plot_3d_box

def generate_bins(bins):
    angle_bins = np.zeros(bins)
    interval = 2 * np.pi / bins
    for i in range(1, bins):
        angle_bins[i] = i * interval
    angle_bins += interval / 2  # center of the bin

    return angle_bins

def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    # if img_2d is not None:
    #     plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location


def annotate_3D(detectedObject, box_2d, label):
    theta_ray = detectedObject.theta_ray
    input_img = detectedObject.img
    proj_matrix = detectedObject.proj_matrix

    truth_img = detectedObject.img
    img = np.copy(truth_img)

    input_tensor = torch.zeros([1, 3, 224, 224])
    input_tensor[0, :, :, :] = input_img

    my_vgg = vgg.vgg19_bn(pretrained=True)
    model = Model(features=my_vgg.features, bins=2)

    [orient, conf, dim] = model(input_tensor)
    orient = orient.cpu().data.numpy()[0, :, :]
    conf = conf.cpu().data.numpy()[0, :]
    dim = dim.cpu().data.numpy()[0, :]

    averages = ClassAverages.ClassAverages()

    dim += averages.get_item(label)

    argmax = np.argmax(conf)
    orient = orient[argmax, :]
    cos = orient[0]
    sin = orient[1]
    alpha = np.arctan2(sin, cos)

    angle_bins = generate_bins(2)

    alpha += angle_bins[argmax]
    alpha -= np.pi

    location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray, truth_img)

    numpy_vertical = np.concatenate((truth_img, img), axis=0)
    cv2.imshow('SPACE for next image, any other key to exit', numpy_vertical)
    cv2.imshow('3D detections', img)

    cv2.waitKey(1)


import numpy as np
import pandas as pd
import os.path
import scipy
import cv2
import sys
import re
from lxml import etree
import caffe

def getImage(imageInput, input_shape=[1, 3, 360, 480]):
    frameRaw = cv2.imread(imageInput)
    rawSize = frameRaw.shape
    frame = cv2.resize(frameRaw, (input_shape[3], input_shape[2]))
    input_image = frame.transpose((2, 0, 1))
    input_image = np.asarray([input_image])
    return input_image,frameRaw,rawSize

def loadNet(model, weights):
    net = caffe.Net(model, weights, caffe.TEST)
    return net

def roadLaneFromSegNet(imageInput, net):
    caffe.set_mode_gpu()
    input_shape = net.blobs['data'].data.shape
    output_shape = net.blobs['argmax'].data.shape

    input_image,raw_image,rawSize = getImage(imageInput, input_shape)
    out = net.forward_all(data=input_image)
    segmentation_ind = np.squeeze(net.blobs['argmax'].data)
    segmentation_ind_3ch = np.resize(segmentation_ind, (3, input_shape[2], input_shape[3]))
    #segmentation_ind
    segmentation_ind_3ch = segmentation_ind_3ch.transpose(1, 2, 0).astype(np.uint8)
    out = cv2.resize(((segmentation_ind_3ch[:,:,0]==3)*255).astype(np.uint8), (rawSize[1],rawSize[0]))
    return out,raw_image

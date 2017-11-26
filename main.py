# coding: utf-8

import numpy as np
import pandas as pd
import os.path
import scipy
import cv2
import sys
from lxml import etree
from utils import *
from laneDetection import LaneProcessor
from caffeutil import *
from xmlutil import *


def extractBinarizedImg(imgout,rawimg):
    rawimg = rawimg[:,:,::-1]
    imgBinary = (imgout>100).astype(np.float)
    for i in range(3):
        rawimg[:,:,i] = rawimg[:,:,i]*imgBinary

    return rawimg


def generateAllFrame(sample_dir, xml_INFO, net, BEV_coord, initParams=None, predLinFit=None):
    l_image =  os.listdir("%s" % (sample_dir))
    l_image = sorted(l_image)
    sample = filter(lambda x: len(x)>0, sample_dir.split("/"))[-1]
    outName = "./testset/%s-Result.xml" % (sample)

    coord_3d = BEV_coord['coord_3d']
    coord_6m = BEV_coord['coord_600cm']

    M_RTK = parseInfo(xml_INFO)
    src,jac = cv2.projectPoints(coord_3d, M_RTK['R'],  M_RTK['T'], M_RTK['K'], M_RTK['D'] )
    src = src[src[:,0,:].argsort(axis=0)[:,0],0,:]
    src_6m = cv2.projectPoints(coord_6m, M_RTK['R'],  M_RTK['T'], M_RTK['K'], M_RTK['D'] )[0][0][0]

    dst = np.array([[[100,1000], [100,0], [1100,0], [1100,1000]]]).astype(np.float32)
    M    = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    mm = np.dot(M, np.array(list(src_6m) + [1.]))
    xy_trans = (mm / mm[2])[0:2]

    refPos = xy_trans
    l_name = []
    l_nameRaw = []
    l_t_x = []
    l_t_y = []
    l_errors = []

    seedParams = []
    l_ret =  []
    for i in range(len(l_image)):
        inName  = "%s/%s" % (sample_dir, l_image[i])
        caffeOutImg,rawImg = roadLaneFromSegNet(inName, net)
        rawImg2 = rawImg.copy()
        caffeOutImg_color = extractBinarizedImg(caffeOutImg,rawImg2)
    
    
        binarizedImg = caffeOutImg
        binarizedImg_t = cv2.warpPerspective(binarizedImg, M, (rawImg2.shape[1],rawImg2.shape[0]), cv2.WARP_INVERSE_MAP)
    
        #rawImg3 = rawImg.copy()
        binarizedImg2 = caffeOutImg_color
        binarizedImg2_t = cv2.warpPerspective(binarizedImg2, M, (rawImg2.shape[1],rawImg2.shape[0]), cv2.WARP_INVERSE_MAP)
        try:
            lp = LaneProcessor(binarizedImg_t, binarizedImg2_t, Minv)
            l_ret.append(lp.full_process())
        except:
            l_ret.append([])
        
    print(outName)
    writeXML(l_ret, outName)


dirs = sys.argv[1]#"/home/xufq/Downloads/hubq/datasets/TSD-Lane/TSD-Lane-00056"
info = sys.argv[2]#"/home/xufq/Downloads/hubq/SegNet-Tutorial/xml/TSD-Lane-Info/TSD-Lane-00056-Info.xml"


model = sys.argv[3]#"../SegNet-Tutorial/Example_Models/segnet_model_driving_webdemo.prototxt"
weights = sys.argv[4]#"../SegNet-Tutorial/Example_Models/segnet_weights_driving_webdemo.caffemodel"

width   = 500
x_start = 200
x_end   = 5000
coord_3d = np.float32([
        [x_start,-width,0], [x_start, width,0],
        [x_end,  -width,0], [x_end,   width,0]
]).reshape(-1,3)
coord_6m = np.array([600., 0., 0.]).reshape(-1,3)
BEV_coords = {
        "coord_3d":coord_3d,
        "coord_600cm" : coord_6m
}
initParams = [
        np.linspace(0.001, 0.02, 10),
        np.array(list(np.linspace(1.5, 50, 10)) + list(np.linspace(-50, 1.5, 10))),
        np.arange( 23, 28, 2),
        np.arange(-15, 15, 2)
]
predLinFit = np.array([ 6.66317499,  1.35541496])
net = loadNet(model, weights)

generateAllFrame(dirs,info, net, BEV_coords)

# -*- coding: gbk -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import scipy
import cv2
import sys
import re
import os
from lxml import etree



def parseChildren(dataSub):
    dim_x = int(dataSub.getchildren()[0].text)
    dim_y = int(dataSub.getchildren()[1].text)
    return np.array(
        filter(
            lambda x: len(x)>0, re.split("\n|\s+", dataSub.getchildren()[-1].text)
        )).astype(np.float64).reshape([dim_x, dim_y])

def parseInfo(xml_INFO):
    tree = etree.parse(xml_INFO)
    data = tree.getroot()
    M_RTK = {}
    for node in data:
        if node.tag == "rotation_matrix":
            M_RTK['R'] = parseChildren(node)
        if node.tag == "translation_vector":
            M_RTK['T'] = parseChildren(node)
        if node.tag == "camera_matrix":
            M_RTK['K'] = parseChildren(node)
        if node.tag == "distortion_coefficients":
            M_RTK['D'] = parseChildren(node)

    return M_RTK



def parseXML(frame):
    #print(frame['Position'])
    out  = '  <Type>"%s"</Type>\n' % (frame['Type'])
    out += '  <Position type_id="opencv-matrix">\n'
    for pos in frame['Position']:
        
        out += '    <rows>%d</rows>\n' % (pos['rows'])
        out += '    <cols>%d</cols>\n' % (pos['cols'])
        out += '    <dt>"%s"</dt>\n'   % (pos['dt'])
        out += '    <data>\n      %s\n    </data>\n' % (" ".join(map(str, pos['data'])))
    
    out += '  </Position>\n'
    return out

    

def getType(intype, ll_points):
    #print(ll_points)
    frame = {
        "Type" : intype,
        "Position" : [{
            "data" : l_points,
            "cols" : 1,
            "rows" : int(len(l_points)/2),
            "dt"   : "2i",
        } for l_points in [ll_points]]
    }
    return parseXML(frame)



def FrameTargetsInfo(ll_frames):
    out  = '<?xml version="1.0" encoding="gbk"?>\n'
    out += '<opencv_storage>'

    for idxFrame,ll_targets in enumerate(ll_frames):
        out += '<Frame%0*dTargetNumber>%d</Frame%0*dTargetNumber>\n' % \
                (5, idxFrame, len(ll_targets), 5, idxFrame)
        for idxTarget,l_targets in enumerate(ll_targets):
            #print(l_targets)
            out += '<Frame%0*dTarget%0*d>\n' % (5, idxFrame, 5, idxTarget)        
            out += getType(intype=l_targets[0], ll_points=l_targets[1])
            out += '</Frame%0*dTarget%0*d>\n' % (5, idxFrame, 5, idxTarget)
    
    return out


def writeXML(FrameTargets, outName):
    out = FrameTargetsInfo(FrameTargets)
    out += '</opencv_storage>\n'
    with open(outName, "w") as fout:
        fout.write(out)

    os.system("iconv -f utf-8 -t gbk %s >%s.1 && mv %s.1 %s" % \
		(outName,outName,outName,outName))





'''
l_points1 = [6,584,407,418]
l_points2 = [7,591,415,419]
l_points3 = [3,1007,157,841,332,654,481,486,526,434,581,363]
FrameTargets = [
  [
    ['黄色实线', [l_points1]],
    
    ['白色实线', [l_points3]]
  ],
  [
    ['黄色实线', [l_points2]]
  ]
]
out = FrameTargetsInfo(FrameTargets)
out += '</opencv_storage>\n'
print(out)

with open("hubqLane.xml", "w") as fout:
    fout.write(out)

import os 
os.system("iconv -f utf-8 -t gbk hubqLane.xml >hubqLane.icv.xml")
'''

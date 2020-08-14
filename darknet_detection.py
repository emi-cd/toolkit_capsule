from ctypes import *
import math
import random
import os, glob, sys
import cv2
import numpy as np
import time
import darknet
import pandas as pd
import codecs

SIZE = 608
OVER = 50

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


def get_latlon(file, path):
    basename = os.path.splitext(os.path.basename(file))[0]
    file_name = basename.split('_')[0]
    with codecs.open(path, "r", "Shift_JIS", "ignore") as file:
        df = pd.read_csv(file)
        for low in df.values.tolist():
            if file_name == low[1]:
                return low[4], low[5]
    return -1, -1
        


netMain = None
metaMain = None
altNames = None


def YOLO(configPath, weightPath, metaPath, inputPath, csvPath):

    global metaMain, netMain, altNames
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    prev_time = time.time()
    inputPath = os.path.join(inputPath, '*.JPG')
    filePaths = glob.glob(inputPath)
    for file in filePaths:
        image_origin = cv2.imread(file)

        height, width, channels = image_origin.shape
        image_origin = cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB)
        lat, lon = get_latlon(file, csvPath)
        print('\r{}'.format(file), end='')

        for y in range(0, height, SIZE-OVER):
            for x in range(0, width, SIZE-OVER):
                clp = image_origin[y:y+SIZE, x:x+SIZE]
                image_resized = cv2.resize(clp,
                                           (darknet.network_width(netMain),
                                            darknet.network_height(netMain)),
                                           interpolation=cv2.INTER_LINEAR)
                darknet.copy_image_from_bytes(darknet_image,image_resized.tobytes())

                detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.70)
                # print(detections)
                if len(detections) > 0:
                    image = cvDrawBoxes(detections, image_resized)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    cv2.imshow('Demo', image)
                    name, ext = os.path.splitext(os.path.basename(file))
                    cv2.imwrite('result/{}_{}_{}result.jpg'.format(name, x, y), image)
                    if lat == -1 and lon == -1:
                        print('\nThere are no information about {} in {}'.format(file, csvPath))
                    else:
                        print('\nlat: {}, lon: {}'.format(lat, lon))
                    cv2.waitKey()
    print(time.time()-prev_time)


if __name__ == "__main__":
    configPath = "./task_capsule/yolov4-custom.cfg"
    weightPath = "./task_capsule/backup/yolov4-custom_best.weights"
    metaPath = "./task_capsule/capsule.data"
    inputPath = '../sampling_data/high'
    if len(sys.argv[1:]) == 2:
        YOLO(configPath, weightPath, metaPath, sys.argv[1], sys.argv[2])
    elif len(sys.argv[1:]) == 1:
        YOLO(configPath, weightPath, metaPath, inputPath, sys.argv[1])
    else:
        print('Wrong argument')

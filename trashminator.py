import numpy as np
import cv2
#import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from sklearn import svm
from sklearn.externals import joblib
import features as ft
import os
import sys

if(__name__ == "__main__"):
    if(len(sys.argv) > 1):
        path_home = sys.argv[1]
    else:
        path_home = "."
    path_model = path_home + "/model.pkl"
    path_testeo = path_home + "/testeo/"
    path_output = path_home + "/output/"
    np.seterr(divide='ignore', invalid='ignore')
    model = joblib.load(path_model)
    for image_name in os.listdir(path_testeo):
    #for image_name in ["1.jpg"]:
    #for image_name in ["IMG_20180414_112742690.jpg"]:
        #print(image_name)
        image = cv2.resize(cv2.imread(path_testeo + image_name), (400,400))
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        segments = slic(image, n_segments = 100, sigma = 5)
        for (i, segVal) in enumerate(np.unique(segments)):
            if(i == i):
                #print(segVal.shape)
                features = ft.retrieve_features(image[segments==segVal], image_hsv[segments==segVal], image_hls[segments==segVal])
                output = model.predict(np.asarray(features).reshape(1, len(features)))
                output2 = model.predict_proba(np.asarray(features).reshape(1, len(features)))
                # mask = np.zeros(image.shape[:2], dtype = "uint8")
                # mask[segments == segVal] = 255
                # cv2.imshow(str(i), cv2.bitwise_and(image, image, mask = mask))
                # cv2.waitKey()
                #print(i, output2)
                #image[segments==segVal]=255
                if(output2[0][0]>=0.65):
                    #print("is not plant")
                    image[segments==segVal]=255
                #else:
                    #print("is plant")
                # mask = np.zeros(image.shape[:2], dtype = "uint8")
                # mask[segments == segVal] = 255
                # cv2.imshow(str(i), cv2.bitwise_and(image, image, mask = mask))
                # cv2.waitKey()
                ### numero de pixeles en cada superpixel ###
                #print(len(image[segments==segVal]))
        cv2.imwrite(path_output + image_name, image)
        #try:
        #    os.remove(path_testeo+image_name)
        #except: pass

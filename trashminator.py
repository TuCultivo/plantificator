import numpy as np
import cv2
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
        image = cv2.resize(cv2.imread(path_testeo + image_name), (400,400))
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        segments = slic(image, n_segments = 100, sigma = 5)
        for (i, segVal) in enumerate(np.unique(segments)):
            superPixel_complete = cv2.bitwise_and(image, image, mask = mask)
            features = ft.retrieve_features(image[segments==segVal], image_hsv[segments==segVal], image_hls[segments==segVal], superPixel_complete)
            output = model.predict_proba(np.asarray(features).reshape(1, len(features)))

            if(output[0][0]>=0.7):
                image[segments==segVal]=255

        cv2.imwrite(path_output + image_name, image)
        #try:
        #    os.remove(path_testeo+image_name)
        #except: pass

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from sklearn import svm
from sklearn.externals import joblib
import features as tf
import os

if(__name__ == "__main__"):
    np.seterr(divide='ignore', invalid='ignore')
    model = joblib.load('model.pkl')
    for image_name in os.listdir("testeo"):
        image = cv2.resize(cv2.imread("testeo/" + image_name), (400,400))
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        segments = slic(image, n_segments = 100, sigma = 5)
        for (i, segVal) in enumerate(np.unique(segments)):
            features = tf.retrieve_features(image[segments==segVal], image_hsv[segments==segVal])
            output = model.predict(np.asarray(features).reshape(1, len(features)))
            if(output==0):
                #print("is not plant")
                image[segments==segVal]=255
            #else:
                #print("is plant")
            # mask = np.zeros(image.shape[:2], dtype = "uint8")
            # mask[segments == segVal] = 255
            # cv2.imshow("superPixel", cv2.bitwise_and(image, image, mask = mask))
            # cv2.waitKey()
            ### numero de pixeles en cada superpixel ###
            #print(len(image[segments==segVal]))
        cv2.imwrite("output/" + image_name, image)

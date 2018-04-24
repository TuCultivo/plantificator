import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from sklearn import svm
from sklearn.externals import joblib
import features as ft

if (__name__ == "__main__"):
    np.seterr(divide='ignore', invalid='ignore')

    x = []
    y = []
    for i in range(1,251):
        if(i < 111 or i > 192):
            image = cv2.resize(cv2.imread("datasetSantaElena/" + str(i) + ".jpg"),(400,400))
            imageBYW = cv2.resize(cv2.imread("datasetSantaElenaBlanco/" + str(i) + ".jpg"),(400,400))
            #imageBYW = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            segments = slic(image, n_segments = 100, sigma = 5)
            for (j, segVal) in enumerate(np.unique(segments)):
                features = ft.retrieve_features(image[segments==segVal], image_hsv[segments==segVal])
                x.append(features)
                if(np.mean(imageBYW[segments==segVal]) > 200):
                    y.append(0)
                else:
                    y.append(1)

                # mask = np.zeros(image.shape[:2], dtype = "uint8")
                # mask[segments == segVal] = 255
                # cv2.imshow("superPixel", cv2.bitwise_and(image, image, mask = mask))
                # cv2.waitKey()
                #if(np.mean(imageBYW[segments==segVal]) > 188 and np.mean(imageBYW[segments==segVal] < 190):
                #    print(np.mean(imageBYW[segments==segVal])
    x = np.asarray(x)
    y = np.asarray(y)
    print(np.sum(y))
    print(len(y))
    #model = svm.SVC(kernel='poly', degree=2, coef0=1)
    model = svm.SVC()
    model.fit(x, y.reshape((x.shape[0])))
    joblib.dump(model, 'model.pkl')

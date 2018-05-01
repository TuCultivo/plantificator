import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from sklearn import svm
from sklearn.externals import joblib
import features as ft
import sys

if (__name__ == "__main__"):
    np.seterr(divide='ignore', invalid='ignore')
    if(len(sys.argv) > 1):
        path_home = sys.argv[1]
    else:
        path_home = "."
    path_model = path_home + "/model.pkl"
    path_dataset = path_home + "/datasetSantaElena/"
    path_datasetBlanco = path_home + "/datasetSantaElenaBlanco/"
    x = []
    y = []
    for i in range(1,251):
        if(i < 111 or i > 192):
            image = cv2.resize(cv2.imread(path_dataset + str(i) + ".jpg"),(400,400))
            imageBYW = cv2.resize(cv2.imread(path_datasetBlanco + str(i) + ".jpg"),(400,400))
            #imageBYW = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            segments = slic(image, n_segments = 100, sigma = 5)
            for (j, segVal) in enumerate(np.unique(segments)):
                features = ft.retrieve_features(image[segments==segVal], image_hsv[segments==segVal], image_hls[segments==segVal])
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

    # with open('juan.pkl', 'wb') as f:
    #     pickle.dump(x, f)
    #
    # with open('juan.pkl', 'rb') as f:
    #     x = pickle.load(f)
    y = np.asarray(y)
    print(np.sum(y))
    print(len(y))
    model = svm.SVC(kernel='poly', degree=3, coef0=1, probability = True)
    #model = svm.SVC(kernel = 'rbf', probability = True)
    model.fit(x, y.reshape((x.shape[0])))
    joblib.dump(model, path_model)

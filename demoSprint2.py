import numpy as np
import cv2
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from sklearn import svm
from sklearn.externals import joblib
import features as ft
import os
import sys
import json
import requests
from watson_developer_cloud import VisualRecognitionV3
import tempfile


# load selected images.
def load_images(path):
    images = []
    for i in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]:
        images.append(cv2.resize(cv2.imread(path + "/datasetSantaElena/" + str(i) + ".jpg"), (400,400)))
    #for i in [198,199,200,201,202]:
    #    images.append(cv2.resize(cv2.imread(path + "/datasetSantaElena/" + str(i) + ".jpg"), (400,400)))
    return images


# preprocess images for deleting trash in them.
def preprocess_images(images, path):
    preprocessed_images = []
    path_model = path + "/model.pkl"
    np.seterr(divide='ignore', invalid='ignore')
    model = joblib.load(path_model)

    for image in images:
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        segments = slic(image, n_segments = 100, sigma = 5)

        for (i, segVal) in enumerate(np.unique(segments)):
            features = ft.retrieve_features(image[segments==segVal], image_hsv[segments==segVal])
            output = model.predict(np.asarray(features).reshape(1, len(features)))
            if(output==0):
                image[segments==segVal]=255
        preprocessed_images.append(image)

    return preprocessed_images

# clasify the preprocessed images into the categories.
def clasify_images(preprocessed_images):
    visual_recognition = VisualRecognitionV3(
    	'2016-05-20',
    	api_key="4893812447dc238483d5c01a41dfc798057baaeb")
    for image in preprocessed_images:
        tmp = tempfile.NamedTemporaryFile()
        cv2.imwrite(tmp.name + ".jpg", image)
        with open(tmp.name + ".jpg", 'rb') as images_file:
            classes = visual_recognition.classify(
    			images_file,
    			parameters=json.dumps({
    				'classifier_ids': ['pestsClasificator_501793644'],
    				'threshold': 0.6
    			}))
    	#print(classes)
        predict = classes["images"][0]["classifiers"][0]["classes"][0]["class"]
        print(predict)
        image = cv2.imread(tmp.name + ".jpg")
        cv2.imshow(predict, image)
        cv2.waitKey()
        cv2.destroyAllWindows()


if( __name__ == "__main__"):
    if(len(sys.argv) > 1):
        path_home = sys.argv[1]
    else:
        path_home = "."
    images = load_images(path_home)
    preprocessed_images = preprocess_images(images, path_home)
    result = clasify_images(preprocessed_images)
    #mostrar_resultado()

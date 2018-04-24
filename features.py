import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import slic

def retrieve_features(image, image_hsv):
    features = []
    #features.append(get_deviation(image))
    #features.append(get_deviation(image_hsv))
    means = get_means(image)
    means[1]=means[1]*2
    features.extend(means)
    mean_combinated = get_mean_combinated(image)
    features.append(mean_combinated)
    #features.append(means[1]*255)
    #means_hsv = get_means(image_hsv)
    #features.extend(means_hsv)
    #normalized_means = get_normalized_means(image)
    for i in range(0):
        if(type(normalized_means[i]) != np.uint8):
            features.append(0)
        else:
            features.append(normalized_means[0])
    #normalized_means_hsv = get_normalized_means(image_hsv)
    for i in range(0):
        if(type(normalized_means_hsv[i]) != np.uint8):
            features.append(0)
        else:
            features.append(normalized_means_hsv[0])
    #ranges = get_ranges(image)
    #features.extend(ranges)
    #ranges_hsv = get_ranges(image_hsv)
    #features.extend(ranges_hsv)


    return features

def get_deviation(image):
    return np.std(image)/100

def get_mean_combinated(image):
    return np.mean((image[:,0] + image[:,1] + image[:,2]) / 765)

def get_means(image):
    R = np.mean(image[:,2])/255
    G = np.mean(image[:,1])/255
    B = np.mean(image[:,0])/255
    return [R, G, B]

def get_normalized_means(image):
    R = image[:,2]
    G = image[:,1]
    B = image[:,0]
    R = np.mean(R / (R + G + B))
    G = np.mean(G / (R + G + B))
    B = np.mean(B / (R + G + B))
    return [R, G, B]

def get_ranges(image):
    Ri = image[:,2]
    Gi = image[:,1]
    Bi = image[:,0]
    R = (Ri.max() - Ri.min())/255.0
    G = (Gi.max() - Gi.min())/255.0
    B = (Bi.max() - Bi.min())/255.0
    return [R,G,B]

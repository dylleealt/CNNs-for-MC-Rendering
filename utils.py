import tensorflow as tf
import numpy as np
import cv2
import pyexr

def read_image(filepath):
    image = cv2.imread(filepath)
    image = np.array(image) / 255
    return image

def write_image(filepath, image):
	image = np.clip(image * 255, 0, 255)
    image = image.astype(np.uint8)
    cv2.imwrite(filepath, image)

def read_exr(filepath):
	file = pyexr.open(filepath)
    data = file.get_all()
    for k, v in data.items():
        data[k] = np.nan_to_num(v)

    return data

def write_exr(filepath, data):
	pass

def preprocess_diffuse(diffuse, albedo, eps):
	return diffuse / (albedo + eps)

def postprocess_diffuse(diffuse, albedo, eps):
    return diffuse * (albedo + eps)

def preprocess_specular(specular):
	return np.log(specular + 1)

def postprocess_specular(specular):
    return np.exp(specular) - 1

def preprocess_diff_variance(variance, albedo, eps):
    return variance / (albedo + eps)**2

def preprocess_spec_variance(variance, specular):
    return variance / (specular)**2

def calulate_variance(data):
	pass

def calculate_gradient(data):
    h, w, c = data.shape
    dX = data[:, 1:, :] - data[:, :w - 1, :]
    dY = data[1:, :, :] - data[:h - 1, :, :]
    dX = np.concatenate((np.zeros([h, 1, c]), dX), axis=1)
    dY = np.concatenate((np.zeros([1, w, c]), dY), axis=0)
    return np.concatenate((dX, dY), axis=2)
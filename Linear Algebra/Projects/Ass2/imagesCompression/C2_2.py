#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 17:08:36 2022

@author: flaviaferrusmarimon
"""

import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
from skimage import img_as_ubyte,img_as_float
import time
from imageio import imwrite
from skimage import img_as_ubyte


color_images = {
    "fez": img_as_float(plt.imread("IMG_1.jpeg")), #.astype(float)
    "marroc": img_as_float(plt.imread("IMG_2.jpeg")), #.astype(float)
    "noruega": img_as_float(plt.imread("IMG_3.jpeg")),
    "cova": img_as_float(plt.imread("IMG_4.jpeg")),
    "mallorca": img_as_float(plt.imread("IMG_5.jpeg")),
    "piano": img_as_float(plt.imread("IMG_6.jpeg")),
    "pompeia": img_as_float(plt.imread("IMG_7.jpeg")),
    "pirineus": img_as_float(plt.imread("IMG_8.jpeg")),
    "aurora": img_as_float(plt.imread("IMG_9.jpeg")),
    "bb": img_as_float(plt.imread("IMG_10.jpeg"))
}

def compress_svd(image,k):
    """
    Perform svd decomposition and truncated (using k singular values/vectors) reconstruction
    returns
    --------
      reconstructed matrix reconst_matrix, array of singular values s
    """
    U,s,V = svd(image,full_matrices=False)
    reconst_matrix = np.dot(U[:,:k],np.dot(np.diag(s[:k]),V[:k,:]))
   
    return reconst_matrix,s


def compress_show_color_images_layer(img_name,k):
    """
     compress and display the reconstructed color image using the layer method 
    """
    image = color_images[img_name]
    original_shape = image.shape
    image_reconst_layers = [compress_svd(image[:,:,i],k)[0] for i in range(3)]
    image_reconst = np.zeros(image.shape)
    for i in range(3):
        image_reconst[:,:,i] = image_reconst_layers[i]
    
    compression_ratio =100.0*3* (k*(original_shape[0] + original_shape[1])+k)/(original_shape[0]*original_shape[1]*original_shape[2])
    frob_percentage =  100*(np.linalg.norm(image) /np.linalg.norm(image_reconst))
    
    plt.title("compression ratio={:.2f}".format(compression_ratio)+"%")
    #plt.title("Frobenius ratio={:.2f}".format(frob_percentage)+"%")
    print("compression ratio={:.2f}".format(compression_ratio)+"%")
    
    plt.imshow(image_reconst)
    
    return image_reconst, compression_ratio


def compute_k_max_color_images_layers(img_name):
    image = color_images[img_name]
    original_shape = image.shape
    return (original_shape[0]*original_shape[1]*original_shape[2])// (3*(original_shape[0] + original_shape[1] + 1))

def importImages(image_name):
    k_max = int(compute_k_max_color_images_layers(image_name))
    
    print('Importing images:')
    print('---- k = 5')
    image, ratio = compress_show_color_images_layer(image_name, 5)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    name = ("./output_"+image_name+"/frobPerc_{:.2f}".format(ratio)+"_"+timestr+".jpeg")
    imwrite(name, image)
    
    for k in range(1, 100):
        if k %10 ==0:
            print('---- k =', k)
            image, ratio = compress_show_color_images_layer(image_name, k)
            timestr = time.strftime("%Y%m%d-%H%M%S")
            name = ("./output_"+image_name+"/frobPerc_{:.2f}".format(ratio)+"_"+timestr+".jpeg")
            #imwrite(name, img_as_ubyte(image))
            imwrite(name, image)
        
    interval = k_max - 100
    step = int(interval / 5)
    #print(step)

    for i in range(5):
        print('---- k =', 100 + step*i)
        image, ratio = compress_show_color_images_layer(image_name, 100 + step*i)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        name = ("./output_"+image_name+"/frobPerc_{:.2f}".format(ratio)+"_"+timestr+".jpeg")
        #imwrite(name, img_as_ubyte(image))        
        imwrite(name, image)



importImages('pirineus')     

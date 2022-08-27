# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 20:47:16 2022

@author: Semih
"""

#Image augmentation

import numpy as np
from skimage.morphology import diameter_closing
from skimage import io
from skimage.color import rgb2lab

import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import pandas as pd



def augmenter(directory,numImages,initial):
    
    for i in range(initial,numImages+1):
        fext = f'{i}' # generate file extension
        
        image = io.imread('Tomato/'+directory+'/image ('+fext+').jpg')

        img_lab = rgb2lab(image)
        a = img_lab[:,:,1]
        b = img_lab[:,:,2]
        
        # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
        pixel_vals_a = a.flatten()
        pixel_vals_b = b.flatten()


        # Convert to float type
        pixel_vals_a = np.float32(pixel_vals_a)
        pixel_vals_b = np.float32(pixel_vals_b)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
        # then perform k-means clustering wit h number of clusters defined as 3
        #also random centres are initially choosed for k-means clustering
        k = 2
        retval, labels_a, centers_a = cv2.kmeans(pixel_vals_a, k, None, criteria, 10,  cv2.KMEANS_RANDOM_CENTERS)
        retval, labels_b, centers_b = cv2.kmeans(pixel_vals_b, k, None, criteria, 10,  cv2.KMEANS_RANDOM_CENTERS)

        centers_a = np.uint8(centers_a)
        centers_b = np.uint8(centers_b)

        segmented_data_a = centers_a[labels_a.flatten()]
        segmented_data_b = centers_b[labels_b.flatten()]

        segmented_image_a = segmented_data_a.reshape((a.shape))
        segmented_image_b = segmented_data_b.reshape((b.shape))

        Threshold_a = threshold_otsu(segmented_image_a)
        Threshold_b = threshold_otsu(segmented_image_b)

        # displaying segmented image
        if segmented_data_a[0]>100:
            segmented_image_a = np.where(segmented_image_a<Threshold_a,255,0)
        else:
            segmented_image_a = np.where(segmented_image_a>Threshold_a,255,0)
        if segmented_data_b[0]>100:
            segmented_image_b = np.where(segmented_image_b<Threshold_b,255,0)
        else:  
            segmented_image_b = np.where(segmented_image_b>Threshold_b,255,0)
    
        segmented_image = np.where(segmented_image_a!=255,segmented_image_b,segmented_image_a)
        temp = np.dstack((segmented_image,segmented_image))
        segmented_image=np.dstack((temp,segmented_image))

        leaf_image = np.where(segmented_image==255,image,0)
    
        leaf_image = diameter_closing(leaf_image, 4, connectivity=1, parent=None, tree_traverser=None)
    
        plt.imsave('Tomato_Augmented/'+directory+'/image ('+fext+').jpg', leaf_image)  
        
    return print('Augmented')

def single_augmenter(num,directory):
    
    fext = f'{num}' # generate file extension
        
    image = io.imread('Tomato/'+directory+'/image ('+fext+').jpg')
    img_lab = rgb2lab(image)
    a = img_lab[:,:,1]

    b = img_lab[:,:,2]
    # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals_a = a.flatten()
    pixel_vals_b = b.flatten()


    # Convert to float type
    pixel_vals_a = np.float32(pixel_vals_a)
    pixel_vals_b = np.float32(pixel_vals_b)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    # then perform k-means clustering wit h number of clusters defined as 3
    #also random centres are initially choosed for k-means clustering
    k = 2
    retval, labels_a, centers_a = cv2.kmeans(pixel_vals_a, k, None, criteria, 10,  cv2.KMEANS_RANDOM_CENTERS)
    retval, labels_b, centers_b = cv2.kmeans(pixel_vals_b, k, None, criteria, 10,  cv2.KMEANS_RANDOM_CENTERS)

    centers_a = np.uint8(centers_a)
    centers_b = np.uint8(centers_b)

    segmented_data_a = centers_a[labels_a.flatten()]
    segmented_data_b = centers_b[labels_b.flatten()]

    segmented_image_a = segmented_data_a.reshape((a.shape))
    segmented_image_b = segmented_data_b.reshape((b.shape))
 
    Threshold_a = threshold_otsu(segmented_image_a)
    Threshold_b = threshold_otsu(segmented_image_b)
    print(Threshold_a, Threshold_b)
    # displaying segmented image
    if segmented_data_a[0]>100:
        segmented_image_a = np.where(segmented_image_a<Threshold_a,255,0)
    else:
        segmented_image_a = np.where(segmented_image_a>Threshold_a,255,0)
    if segmented_data_b[0]>100:
        segmented_image_b = np.where(segmented_image_b>Threshold_b,0,255)
    else:  
        segmented_image_b = np.where(segmented_image_b<Threshold_b,255,0)
 
    segmented_image = np.where(segmented_image_a!=255,segmented_image_b,segmented_image_a)
    temp = np.dstack((segmented_image,segmented_image))
    segmented_image=np.dstack((temp,segmented_image))

    leaf_image = np.where(segmented_image==255,image,0)

    plt.imsave('Tomato_Augmented/'+directory+'/image ('+fext+').jpg', leaf_image)  
    
    return leaf_image,segmented_image_b
data = pd.read_csv("disease_info_tomato.csv", encoding="cp437")

#Augmentation for all dataset
i = 0
'''
while i<10:
    if i == 1:
        i = 9
        initial = 1161
    
    directory = data["disease_name"][i]
    
    amount = len([f for f in os.listdir('Tomato/'+directory)])
    if directory == 'Tomato___Spider_mites Two-spotted_spider_mite':
        initial = 1161
    else:
        initial = 1
    augmenter(directory,amount,initial)
    
    i = i+1
'''
directory = 'Tomato___Septoria_leaf_spot'
num = 19
img, _ = single_augmenter(num,directory)
plt.title('augmented {num}')
#plt.imshow(img)
# Print the number of
# CPUs in the system
#print("Number of CPUs in the system:", cpuCount)

'''
img = io.imread('Tomato/'+directory+'/image (168).jpg')
img_lab = rgb2lab(img)
a = img_lab[:,:,1]

b = img_lab[:,:,2]
    # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
pixel_vals_a = a.flatten()
pixel_vals_b = b.flatten()
    #cluster_data = img_a.reshape((-1,2))

    # Convert to float type
pixel_vals_a = np.float32(pixel_vals_a)
pixel_vals_b = np.float32(pixel_vals_b)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
    # then perform k-means clustering wit h number of clusters defined as 3
    #also random centres are initially choosed for k-means clustering
k = 2
retval, labels_a, centers_a = cv2.kmeans(pixel_vals_a, k, None, criteria, 10,  cv2.KMEANS_RANDOM_CENTERS)
retval, labels_b, centers_b = cv2.kmeans(pixel_vals_b, k, None, criteria, 10,  cv2.KMEANS_RANDOM_CENTERS)

    # convert data into 8-bit values
centers_a = np.uint8(centers_a)
centers_b = np.uint8(centers_b)

segmented_data_a = centers_a[labels_a.flatten()]
segmented_data_b = centers_b[labels_b.flatten()]

segmented_image_a = segmented_data_a.reshape((a.shape))
segmented_image_b = segmented_data_b.reshape((b.shape))

Threshold_a = threshold_otsu(segmented_image_a)
Threshold_b = threshold_otsu(segmented_image_b)

pixel_labels_a = labels_a.reshape(img_lab.shape[0], img_lab.shape[1])
pixel_labels_b = labels_b.reshape(img_lab.shape[0], img_lab.shape[1])
# displaying segmented image
if segmented_data_a[0]>100:
    segmented_image_a = np.where(segmented_image_a<Threshold_a,255,0)
else:
    segmented_image_a = np.where(segmented_image_a>Threshold_a,255,0)
if segmented_data_b[0]>100:
    segmented_image_b = np.where(segmented_image_b<Threshold_b,255,0)
else:  
    segmented_image_b = np.where(segmented_image_b>Threshold_b,255,0)
    
print(Threshold_a,Threshold_b)

segmented_image = np.where(segmented_image_a!=255,segmented_image_b,segmented_image_a)
temp = np.dstack((segmented_image,segmented_image))
segmented_image=np.dstack((temp,segmented_image))

leaf_image = np.where(segmented_image==255,img,0)

####################
'''
'''
rgb_vals = img.reshape((-1,3))

cluster1 = np.where(0==labels, rgb_vals,0)
cluster_img1 = cluster1.reshape(img.shape)

cluster2 = np.where(1==labels, rgb_vals,0)
cluster_img2 = cluster2.reshape(img.shape)

cluster3 = np.where(2==labels, rgb_vals,0)
cluster_img3 = cluster3.reshape(img.shape)

    #getting rid of leafless cluster
v1 = np.sum(cluster_img1[:,:,0])/(256*256*3)
check1 = np.sum(cluster_img1[100:140,100:140,1])/1681

v2 = np.sum(cluster_img2[:,:,0])/(256*256*3)
v3 = np.sum(cluster_img3[:,:,0])/(256*256*3)
check = max(v1,v2,v3)
print(v1,v2,v3)
if check == v1:
    leaf_img = cluster_img2+cluster_img1
elif check == v2:
    leaf_img = cluster_img1+cluster_img3
else:
    leaf_img = cluster_img1+cluster_img2
    
leaf_img = diameter_closing(leaf_img, 4, connectivity=1, parent=None, tree_traverser=None)

plt.subplot(2,2,1)   
plt.title('1')
plt.imshow(cluster_img1)

plt.subplot(2,2,2)   
plt.title('2')
plt.imshow(cluster_img2)

plt.subplot(2,2,3)   
plt.title('3')
plt.imshow(cluster_img3)

plt.subplot(2,2,4)   
plt.title('3')
plt.imshow(leaf_img)
'''
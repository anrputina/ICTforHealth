#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 11:45:55 2017

@author: anr.putina
"""

import numpy as np
import scipy

from skimage import io, filters


mole_img = np.array(io.imread("../Data/images/low_risk_1.jpg"),
              dtype=np.float64)
#edges = filters.sobel(mole_img[:,:,2])
#io.imshow(edges)

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin
#from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time

n_colors = 4

# Load the Summer Palace photo

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1])
mole_img = np.array(mole_img, dtype=np.float64) / 255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(mole_img.shape)
assert d == 3
image_array = np.reshape(mole_img, (w * h, d))

image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
labels = kmeans.predict(image_array)

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


from sklearn.feature_extraction import image
mole_img = scipy.misc.imresize(recreate_image(kmeans.cluster_centers_, labels, w, h)[:,:,0], 0.99)

from skimage import measure
# Find contours at a constant value of 0.8
contours = measure.find_contours(mole_img, 0.8)

# Display all results, alongside original image
plt.figure(1)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Original image')
plt.imshow(mole_img)

plt.figure(2)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image kmeans')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))
ax.plot(contours[0][:, 1], contours[0][:, 0], linewidth=2)

fig, ax = plt.subplots(1, 2, figsize=(13,6))
ax[0].imshow(mole_img)
ax[1].imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))
ax[0].set_xlabel('original')
ax[1].set_xlabel('k-means')

fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))
ax.plot(contours[0][:, 1], contours[0][:, 0], linewidth=2)
ax.set_xlim(0,580)
ax.set_ylim(0,580)
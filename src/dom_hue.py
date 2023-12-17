import matplotlib.image as img
import matplotlib.pyplot as plt
from scipy.cluster.vq import whiten
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
import time
import cv2
import numpy as np

start = time.time()
image = img.imread('blob.jpg')
image = img.imread('dom_col.jpg')
image = cv2.imread('dom_col.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
 
hue = image[:, :, 0].flatten()
 
batman_df = pd.DataFrame({'hue': hue,
                          })
 

start = time.time()
batman_df['scaled_hue'] = whiten(batman_df['hue'])
 
n_clusters = 5
batch_size = 32
print(batman_df[['scaled_hue']].shape)
km = MiniBatchKMeans(n_clusters = n_clusters, batch_size=batch_size, n_init='auto').fit(batman_df[['scaled_hue']])
cluster_centers = km.cluster_centers_

hue_std = batman_df['hue'].std()

dominant_hues = []
for cluster_center in cluster_centers:
    hue_scaled = cluster_center[0]
 
    # Convert each standardized value to scaled value
    dominant_hues.append(
        hue_scaled * hue_std,
    )

hues = np.asarray(dominant_hues, dtype='uint8')

percentage = np.asarray(np.unique(km.labels_, return_counts = True)[1], dtype='float32')
percentage = percentage/(image.shape[0]*image.shape[1])


dom = [[percentage[ix], hues[ix]] for ix in range(km.n_clusters)]
dominance = sorted(dom, key=lambda x:x[0], reverse=True)
print(dominance)

image_cp = image.copy()

labels_mask = km.labels_.reshape(image.shape[0], image.shape[1])
for label in range(km.n_clusters):
    image_cp[labels_mask == label, 0] = hues[label]

img = image_cp.reshape(image.shape[0], -1, 3)
img = img[:, :, 0]
print(f"{time.time() - start=}")
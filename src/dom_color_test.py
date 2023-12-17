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
#image = cv2.imread('dom_col.jpg')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
 
r = image[:, :, 0].flatten()
g = image[:, :, 1].flatten()
b = image[:, :, 2].flatten()

 
batman_df = pd.DataFrame({'red': r,
                          'green': g,
                          'blue': b})
 
print(f"Time DF: {time.time() - start}")

start = time.time()
batman_df['scaled_color_red'] = whiten(batman_df['red'])
batman_df['scaled_color_blue'] = whiten(batman_df['blue'])
batman_df['scaled_color_green'] = whiten(batman_df['green'])

print(f"Time DF: {time.time() - start}")
 
start = time.time()
n_clusters = 10
batch_size = 32
km = MiniBatchKMeans(n_clusters = n_clusters, batch_size=batch_size).fit(batman_df[['scaled_color_red','scaled_color_blue','scaled_color_green']])
cluster_centers = km.cluster_centers_
print(f"Time kmeans: {time.time() - start}")

red_std, green_std, blue_std = batman_df[['red',
                                          'green',
                                          'blue']].std()

print(f"{red_std=}, {green_std=}, {blue_std=}")
dominant_colors = []
for cluster_center in cluster_centers:
    red_scaled, green_scaled, blue_scaled = cluster_center
 
    # Convert each standardized value to scaled value
    dominant_colors.append((
        red_scaled * red_std,
        green_scaled * green_std,
        blue_scaled * blue_std
    ))

print(f"{dominant_colors=}")
colors = np.asarray(dominant_colors, dtype='uint8')
print(colors.shape)
#colors = cv2.cvtColor(np.expand_dims(colors, 0), cv2.COLOR_HSV2RGB)
print(f"{colors=}")
#colors = colors.reshape(-1, 3)

percentage = np.asarray(np.unique(km.labels_, return_counts = True)[1], dtype='float32')
percentage = percentage/(image.shape[0]*image.shape[1])
print(percentage)
print(np.sum(percentage))



plt.figure(0)
for ix in range(colors.shape[0]):
    patch = np.ones((20, 20, 3), dtype=np.uint8)
    patch[:, :, :] = colors[ix]
    plt.subplot(1, colors.shape[0], ix+1)
    plt.axis('off')
    plt.imshow(patch)
plt.show()



dom = [[percentage[ix], colors[ix]] for ix in range(km.n_clusters)]
dominance = sorted(dom, key=lambda x:x[0], reverse=True)

plt.figure(0)
plt.axis('off')

patch = np.zeros((50, 500, 3), dtype=np.uint8)

start = 0
for cx in range(km.n_clusters):
    width = int(dominance[cx][0] * patch.shape[1])
    end = start + width
    patch[:, start:end, :] = dominance[cx][1]
    start = end
plt.imshow(patch)
plt.show()


print(f"{km.labels_.shape=}")
image_cp = image.copy()
print(np.sum(image_cp != image))
labels_mask = km.labels_.reshape(image.shape[0], image.shape[1])
for label in range(km.n_clusters):
    image_cp[labels_mask == label] = colors[label]

img = image_cp.reshape(image.shape[0], -1, 3)

plt.imshow(img)
plt.show()


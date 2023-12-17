import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage.color import label2rgb# settings for LBP

radius = 5
n_points = 8 * radius
METHOD = 'uniform'


def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))


def match(refs, img):
    best_score = 10
    best_name = None
    lbp = local_binary_pattern(img, n_points, radius, METHOD)
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    for name, ref in refs.items():
        ref_hist, _ = np.histogram(ref, density=True, bins=n_bins,
                                   range=(0, n_bins))
        score = kullback_leibler_divergence(hist, ref_hist)
        print(f"Score for {name}: {score}")
        if score < best_score:
            best_score = score
            best_name = name
    return best_name

def hist(ax, lbp):
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')

balcony = cv2.cvtColor(cv2.imread('balcony.jpg'), cv2.COLOR_BGR2GRAY)
balcony_test = cv2.cvtColor(cv2.imread('balcony_test.jpg'), cv2.COLOR_BGR2GRAY)
kermit = cv2.cvtColor(cv2.imread('blob.jpg'), cv2.COLOR_BGR2GRAY)

refs = {
    'balcony': local_binary_pattern(balcony, n_points, radius, METHOD),
    'kermit': local_binary_pattern(kermit, n_points, radius, METHOD)
}

# classify rotated textures
print('Rotated images matched against references using LBP:')
print('original: balcony, rotated: 30deg, match result: ',
      match(refs, rotate(balcony_test, angle=30, resize=False)))
print('original: balcony, rotated: 70deg, match result: ',
      match(refs, rotate(balcony_test, angle=70, resize=False)))
print('original: balcony, rotated: 145deg, match result: ',
      match(refs, rotate(balcony_test, angle=145, resize=False)))

# plot histograms of LBP of textures
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3,
                                                       figsize=(9, 6))
plt.gray()

ax1.imshow(balcony)
ax1.axis('off')
hist(ax4, refs['balcony'])
ax4.set_ylabel('Percentage')

ax2.imshow(kermit)
ax2.axis('off')
hist(ax5, refs['kermit'])
ax5.set_xlabel('Uniform LBP values')

ax3.imshow(balcony_test)
ax3.axis('off')
hist(ax6, local_binary_pattern(balcony_test, n_points, radius, METHOD))

plt.show()
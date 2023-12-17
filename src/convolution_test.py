import cv2
import numpy as np
import scipy


kermit = cv2.imread("blob.jpg")
balcony = cv2.imread("balcony.jpg")
balcony_test = cv2.imread("balcony_test.jpg")
print(f"kermit: {kermit.shape}")
print(f"balcony: {balcony.shape}")
print(f"balcony_test: {balcony_test.shape}")
balcony_test = cv2.resize(balcony_test, (kermit.shape[1], kermit.shape[0]))

kermit = kermit.astype(np.float32)
balcony = balcony.astype(np.float32)
balcony_test = balcony_test.astype(np.float32)



out1 = scipy.signal.correlate(balcony, kermit, mode='full')
print(np.max(out1))

out2 = scipy.signal.correlate(kermit, balcony_test, mode='full')
print(np.max(out2))

out3 = scipy.signal.correlate(balcony, balcony_test, mode='full')
print(np.max(out3))

out4 = scipy.signal.correlate(balcony_test, balcony_test, mode='full')
print(np.max(out4))


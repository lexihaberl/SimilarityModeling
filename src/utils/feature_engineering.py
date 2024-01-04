from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import cv2
import numpy as np
import time
import pickle
import librosa
from tqdm import tqdm
from utils import io
import matplotlib.pyplot as plt


def detect_blob(image, sigma, mu, gauss=True, debug=False):
    if gauss:
        if len(image.shape) == 3:
            image_h = image[:, :, 0]
        else:
            image_h = image
        gauss = 1/(2*np.pi*sigma**2) * np.exp(-(image_h - mu)**2/(2*sigma**2))
        image_gray = gauss/np.max(gauss)
        image = image_gray

    start = time.time()
    blobs_doh = blob_doh(image_gray, min_sigma=10, max_sigma=500, threshold=.01)
    end = time.time()
    # print("Blobdoh: ", end - start)

    blobs_list = [blobs_doh]
    colors = ['red']
    titles = ['Determinant of Hessian']
    sequence = zip(blobs_list, colors, titles)

    if debug:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)

        for (blobs, color, title) in sequence:
            ax.set_title(title)
            ax.imshow(image_gray)
            for blob in blobs:
                y, x, r = blob
                c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
                ax.add_patch(c)
            ax.set_axis_off()

        plt.tight_layout()
        plt.show()
    
    return blobs_doh
    

def detect_blob_cv(im, im_bgr, im_hsv, mu_v=225, sigma_v=35, mu_h=40, sigma_h=35, debug=False):
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    image_v = im_hsv[:, :, 2].astype(np.float32)
    image_h = im_hsv[:, :, 0].astype(np.float32)
    
    max_v = np.max(image_v)
    gauss = 1/(2*np.pi*sigma_h**2*sigma_v**2) * np.exp(-(image_h - mu_h)**2/(2*sigma_h**2) - (image_v - mu_v)**2/(2*sigma_v**2))
    im = (gauss/np.max(gauss)*255).astype(np.uint8)
    if debug:
        # print(max_v)
        plt.gray()
        ax1 = plt.subplot(1, 2, 1) 
        ax1.imshow(im)
        ax2 = plt.subplot(1, 2, 2)
        ax2.imshow(image_h)
        plt.show()
    # 150 - 230/240
    # 160-244
     
    # Change thresholds
    params.minThreshold = 100
    params.maxThreshold = 2000
     
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 150
     
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1
     
    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87
     
    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01
    params.blobColor = 255
     
    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else : 
        detector = cv2.SimpleBlobDetector_create(params)
     
     
    # Detect blobs.
    keypoints = detector.detect(im)
     
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob
    if debug:
        im_with_keypoints = cv2.drawKeypoints(im_bgr, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        im_with_keypoints = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2RGB)
        plt.imshow(im_with_keypoints)

    return keypoints


def get_dominant_color(episode_names, video_paths, path_to_save=None):
    feat_dict = {}
    for ep in episode_names:
        cap = io.load_video(video_paths[ep])
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        feat_list = []  
        
        for i in range(frame_count):
            _, frame = cap.read()
            dc = get_dominant_color(frame)[1]
            feat_list.append(dc)

        feat_dict[ep] = np.array(feat_list)[1:]

        del feat_list
        cap.release()
    
    if path_to_save is not None:
        pickle.dump(feat_dict, open(path_to_save, 'wb'))


def create_foreground_masks_video(episode_path):
    cap = io.load_video(episode_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    shape = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out_path = episode_path.replace('.avi', '_foreground.avi')
    out_path = out_path.replace('videos', 'features')
    out_video = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'H264'), 
        fps, 
        shape,
        isColor=False
    )

    prev_flow = None
    prev_frame = None
    for i in tqdm(range(frame_count)):
        _, frame = cap.read()
        this_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_flow is None:
            prev_frame = np.zeros_like(this_frame)
            prev_flow = np.zeros_like(this_frame, dtype=np.float32)

        flow = cv2.calcOpticalFlowFarneback(
                prev_frame, 
                this_frame, 
                prev_flow, 
                pyr_scale=0.5, 
                levels=2, 
                winsize=100, 
                iterations=1, 
                poly_n=7, 
                poly_sigma=1.7, 
                flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
            )
        flow_mag, flow_ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        threshold = max(np.mean(flow_mag) - 0.2 * np.std(flow_mag), 0.3)
        mask= ((flow_mag > threshold) * 255).astype(np.uint8)

        kernel = np.ones((7,7),np.uint8)
        mask = cv2.erode(mask, kernel=kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=kernel, iterations=1)
        out_video.write(mask)

        prev_frame = this_frame
        prev_flow = flow

    cap.release()
    out_video.release()



def get_biggest_stft_peaks(rec, sr, frame_length, hop_length, top_k=5):
    stft = librosa.stft(y=rec, n_fft=frame_length, hop_length=hop_length)
    
    stft_mag = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)

    highest_mags_feature = np.zeros((stft_mag.shape[1], top_k))
    freqs_feature = np.zeros((stft_mag.shape[1], top_k))

    for sample_idx in range(stft_mag.shape[1]):
        peaks = librosa.util.peak_pick(
                    stft_mag[:, sample_idx], 
                    pre_max=30, 
                    post_max=30, 
                    pre_avg = 30, 
                    post_avg = 30, 
                    delta=0.4, 
                    wait=30
                    )
        if len(peaks) <= 0:
            continue

        peak_freqs = freqs[peaks]
        peak_mags = stft_mag[peaks, sample_idx]
        sorted_idx = np.argsort(peak_mags)[::-1]
        peak_freqs = peak_freqs[sorted_idx]
        peak_mags = peak_mags[sorted_idx]

        n_peaks = min(top_k, len(peaks))
        highest_mags_feature[sample_idx, :n_peaks] = peak_mags[:n_peaks]
        freqs_feature[sample_idx, :n_peaks] = peak_freqs[:n_peaks]
        
    return highest_mags_feature, freqs_feature
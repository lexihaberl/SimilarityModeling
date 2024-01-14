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
from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.vq import whiten
from hmmlearn.hmm import GaussianHMM
import mahotas as mt


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
    blobs_doh = blobs_doh[blobs_doh[:, 2] > 20]
    end = time.time()
    # print("Blobdoh: ", end - start)

    blobs_list = [blobs_doh]
    colors = ['red']
    titles = ['Determinant of Hessian']
    sequence = zip(blobs_list, colors, titles)

    list_blobs = []
    if debug:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)

        for (blobs, color, title) in sequence:
            ax.set_title(title)
            ax.imshow(image_gray)
            for blob in blobs:
                y, x, r = blob
                list_blobs.append(r)
                c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
                ax.add_patch(c)
            ax.set_axis_off()
            print(sorted(list_blobs, reverse=True))

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


def get_dominant_colors(episode_path, episode_name, n_clusters = 5, batch_size = 2048, type='full'):
    '''
    Computes the dominant hues in each frame of the video using KMeans clustering.
    The returned array is of shape (n_frames, n_clusters, 2) where the second dimension is the percentage of pixels in the frame that belong to the cluster and the hue of the cluster.

    type: full, foreground, background: Full uses the entire image, foreground only uses pixels inside the foreground mask, background uses pixels outside the foreground mask
    '''
    cap = io.load_video(episode_path)
    if type != 'full':
        foreground_video = io.load_video("../data/features/{}_foreground.avi".format(episode_name))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hue_list = []

    for i in tqdm(range(frame_count)):
        _, image = cap.read()
        
        if type != 'full':
            _, fg = foreground_video.read()
            if type == 'foreground':
                image[fg == 0] = 0
            elif type == 'background':
                image[fg != 0] = 0

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue = image[:, :, 0].flatten()
        hue_std = hue.std()
        if hue_std == 0:
            hue_list.append([[0, 0] for _ in range(n_clusters)])
            continue
        scaled_hue = np.expand_dims(whiten(hue), -1)

        km = MiniBatchKMeans(n_clusters = n_clusters, batch_size=batch_size, n_init='auto').fit(scaled_hue)
        cluster_centers = km.cluster_centers_

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

        if len(percentage) < n_clusters:
            percentage = np.append(percentage, np.zeros(n_clusters - len(percentage)))

        dom = [[percentage[ix], hues[ix]] for ix in range(km.n_clusters)]
        dominance = sorted(dom, key=lambda x:x[0], reverse=True)

        hue_list.append(dominance)

        del km

    cap.release()
    if type != 'full':
        foreground_video.release()

    return np.array(hue_list)[1:]


def create_foreground_masks_video(episode_path, write_video=True):
    '''
    Creates a video of the foreground masks for each frame in the video and saves it as a video to the features folder.
    The foreground mask is created by using the Farneback optical flow algorithm to detect motion in the video. 
    The magnitude of the flow is thresholded to create a mask. The threshold is set to be the mean flow magnitude minus 0.2 times the standard deviation of the flow magnitude.
    '''
    cap = io.load_video(episode_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mean_flow = np.zeros(frame_count)
    std_dev_flow = np.zeros(frame_count)
    fps = cap.get(cv2.CAP_PROP_FPS)
    shape = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out_path = episode_path.replace('.avi', '_foreground.avi')
    out_path = out_path.replace('videos', 'features')
    if write_video:
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
        mean_flow[i] = np.mean(flow_mag)
        std_dev_flow[i] = np.std(flow_mag)
        threshold = max(np.mean(flow_mag) - 0.2 * np.std(flow_mag), 0.3)
        mask= ((flow_mag > threshold) * 255).astype(np.uint8)

        kernel = np.ones((7,7),np.uint8)
        mask = cv2.erode(mask, kernel=kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=kernel, iterations=1)
        if write_video:
            out_video.write(mask)

        prev_frame = this_frame
        prev_flow = flow
    feat_dict = {}
    feat_dict['mean_flow'] = mean_flow
    feat_dict['std_dev_flow'] = std_dev_flow
    pickle.dump(feat_dict, open(out_path.replace('.avi', '.pkl'), 'wb'))
    cap.release()
    if write_video:
        out_video.release()



def get_biggest_stft_peaks(rec, sr, frame_length, hop_length, top_k=5):
    '''
    Computes the STFT of the audio signal and returns the top_k peaks (value and frequency) in each frame of the STFT.
    '''
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


def get_kermitian_pixels(episode_path, episode_name):
    '''
    Computes the number of pixels in each frame inside the foreground and background masks that are within the HSV range of Kermit's color.
    '''
    cap = io.load_video(episode_path)
    foreground_video_path = f"../data/features/{episode_name}_foreground.avi"
    foreground_video = io.load_video(foreground_video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    kermit_pixels_fg_arr = np.zeros(frame_count)
    kermit_pixels_bg_arr = np.zeros(frame_count)
    kermit_pixels_ratio = np.zeros(frame_count)
    for i in tqdm(range(frame_count)):
        _, image = cap.read()
        _, foreground_mask = foreground_video.read()

        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        foreground_mask = cv2.cvtColor(foreground_mask, cv2.COLOR_BGR2GRAY)
        foreground = image_hsv[foreground_mask != 0]
        background = image_hsv[foreground_mask == 0]
        if foreground.shape[0] == 0 or background.shape[0] == 0:
            continue
        kermit_pixels_fg = ((foreground[:, 0] >= 25) & (foreground[:, 0] <= 50) & 
                            #(foreground[:, :, 1] >= 100) & (foreground[:, :, 1] <= 255) &
                            (foreground[:, 2] >= 80) & (foreground[:, 2] <= 180)).sum()
        kermit_pixels_bg = ((background[:, 0] >= 25) & (background[:, 0] <= 50) & 
                            #(background[:, :, 1] >= 100) & (background[:, :, 1] <= 255) &
                            (background[:, 2] >= 80) & (background[:, 2] <= 180)).sum()
        if kermit_pixels_fg == 0 and kermit_pixels_bg == 0:
            continue
        kermit_pixels_ratio[i] = kermit_pixels_fg / (kermit_pixels_fg + kermit_pixels_bg)
        kermit_pixels_fg_arr[i] = kermit_pixels_fg
        kermit_pixels_bg_arr[i] = kermit_pixels_bg
    feat = {}
    feat['num_kermit_pixels_foreground'] = kermit_pixels_fg_arr[1:]
    feat['num_kermit_pixels_background'] = kermit_pixels_bg_arr[1:]
    feat['num_kermit_pixels'] = kermit_pixels_fg_arr[1:] + kermit_pixels_bg_arr[1:]
    feat['kermit_pixels_ratio'] = kermit_pixels_ratio[1:]
    return feat

def extract_foreground(orig_image):
    '''
    Extracts the foreground from an image using GrabCut.
    
    Args:
        image: A 3D numpy array representing the image.
        
    Returns:
        A 3D numpy array representing the foreground.
    '''
    resize_factor = 8
    image = cv2.resize(orig_image, (orig_image.shape[1]//resize_factor, orig_image.shape[0]//resize_factor))
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    rect = (5,5,image.shape[1]-5,image.shape[0]-5)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    mask2 = cv2.resize(mask2, (orig_image.shape[1], orig_image.shape[0]))
    foreground = orig_image*mask2[:,:,np.newaxis]
    return foreground, mask2


def get_pig_pixels(episode_path):
    '''
    Computes the number of pixels in each frame inside the foreground and background masks that are within the HSV range of a Pigs's color.
    Uses masks form GrabCut to extract the foreground.
    '''
    print(episode_path)
    cap = io.load_video(episode_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    pig_pixels_fg_arr = np.zeros(frame_count)
    pig_pixels_bg_arr = np.zeros(frame_count)
    pig_pixels_ratio = np.zeros(frame_count)
    for i in tqdm(range(frame_count)):
        _, image = cap.read()
        image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
        
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        foreground_mask = extract_foreground(image)[1]
        foreground = image_hsv[foreground_mask != 0]
        background = image_hsv[foreground_mask == 0]
        if foreground.shape[0] == 0 or background.shape[0] == 0:
            continue
        
        pig_pixels_fg = (((foreground[:, 0] < 15) | (foreground[:, 0] > 175)) & 
                            (foreground[:, 1] > 80) & (foreground[:, 1] < 150) &
                            (foreground[:, 2] > 100) & (foreground[:, 2] < 170)).sum()
        pig_pixels_bg = (((background[:, 0] < 15) | (background[:, 0] > 175)) &
                            (background[:, 1] > 80) & (background[:, 1] < 150) &
                            (background[:, 2] > 100) & (background[:, 2] < 170)).sum()

        if pig_pixels_fg == 0 and pig_pixels_bg == 0:
            continue
        pig_pixels_ratio[i] = pig_pixels_fg / (pig_pixels_fg + pig_pixels_bg)
        pig_pixels_fg_arr[i] = pig_pixels_fg
        pig_pixels_bg_arr[i] = pig_pixels_bg
    feat = {}
    feat['num_pig_pixels_foreground'] = pig_pixels_fg_arr[1:]
    feat['num_pig_pixels_background'] = pig_pixels_bg_arr[1:]
    feat['num_pig_pixels'] = pig_pixels_fg_arr[1:] + pig_pixels_bg_arr[1:]
    feat['pig_pixels_ratio'] = pig_pixels_ratio[1:]
    return feat

    

def get_sift(video_paths, episode_names):
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                table_number = 6, # 12
                key_size = 12,     # 20
                multi_probe_level = 1) #2
    search_params = dict(checks=50)   
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    cap = cv2.VideoCapture(video_paths[episode_names[0]])
    cap.set(1, 19500)
    ret, reference = cap.read()
    reference = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
    plt.imshow(reference)

    cap.set(1, 19600)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()

    reference = extract_foreground(reference)
    frame = extract_foreground(frame)
    plt.imshow(reference)
    plt.imshow(frame)

    # Initiate ORB detector
    orb = cv2.ORB_create()
    kp_ref, des_ref = orb.detectAndCompute(reference, None)
    kp, des = orb.detectAndCompute(frame, None)


    matches = flann.knnMatch(des_ref,des,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(reference,kp_ref,frame,kp,matches,None,**draw_params)
    plt.figure(figsize=(20,10))
    plt.imshow(img3,),plt.show()

def extract_haralick_texture(video_paths, episode_names):
    texture_feat = {}
    for episode in episode_names:
        cap = cv2.VideoCapture(video_paths[episode])
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        textures = np.zeros((frame_count, 13))
        for i in tqdm(range(frame_count)):
            cap.set(1, i)
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_fg = extract_foreground(frame)
            frame_fg_resized = cv2.resize(frame_fg, (frame_fg.shape[1]//4, frame_fg.shape[0]//4))
            haralick = mt.features.haralick(frame_fg_resized)
            texture = haralick.mean(axis=0)
            textures[i] = texture
        cap.release()
        texture_feat[episode] = {'texture': textures[1:]}

    return texture_feat


def cmvnw(vec, win_size=301, variance_normalization=False):
    """ This function is aimed to perform local cepstral mean and
    variance normalization on a sliding window. The code assumes that
    there is one observation per row.

    Args:
        vec (array): input feature matrix
            (size:(num_observation,num_features))
        win_size (int): The size of sliding window for local normalization.
            Default=301 which is around 3s if 100 Hz rate is
            considered(== 10ms frame stide)
        variance_normalization (bool): If the variance normilization should
            be performed or not.

    Return:
          array: The mean(or mean+variance) normalized feature vector.
    """
    # Get the shapes
    eps = 2**-30
    rows, cols = vec.shape

    # Windows size must be odd.
    assert isinstance(win_size, int), "Size must be of type 'int'!"
    assert win_size % 2 == 1, "Windows size must be odd!"

    # Padding and initial definitions
    pad_size = int((win_size - 1) / 2)
    vec_pad = np.lib.pad(vec, ((pad_size, pad_size), (0, 0)), 'symmetric')
    mean_subtracted = np.zeros(np.shape(vec), dtype=np.float32)

    for i in range(rows):
        window = vec_pad[i:i + win_size, :]
        window_mean = np.mean(window, axis=0)
        mean_subtracted[i, :] = vec[i, :] - window_mean

    # Variance normalization
    if variance_normalization:

        # Initial definitions.
        variance_normalized = np.zeros(np.shape(vec), dtype=np.float32)
        vec_pad_variance = np.lib.pad(
            mean_subtracted, ((pad_size, pad_size), (0, 0)), 'symmetric')

        # Looping over all observations.
        for i in range(rows):
            window = vec_pad_variance[i:i + win_size, :]
            window_variance = np.std(window, axis=0)
            variance_normalized[i, :] \
            = mean_subtracted[i, :] / (window_variance + eps)
        output = variance_normalized
    else:
        output = mean_subtracted

    return output


def get_audio_librosa(video_paths, ep, gt_df):
    rec, sr = librosa.load(video_paths[ep], sr=None)

    frame_size_ms = 100
    hop_length = int(1/25 * sr)
    frame_length = int(frame_size_ms / 1000 * sr)
    
    desired_len = len(gt_df[gt_df.episode==ep])

    return rec, sr, frame_length, hop_length, desired_len


def get_mfcc(episode_names, video_paths, gt_df):
    ep_dfs = {}
    for ep in episode_names:
        rec, sr, frame_length, hop_length, desired_len = get_audio_librosa(video_paths, ep, gt_df)

        mfcc = librosa.feature.mfcc(y=rec, sr=sr, n_fft=frame_length, hop_length=hop_length)
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, desired_len - mfcc.shape[1])))
        normalized_mfcc = cmvnw(mfcc.T, win_size=frame_length+1, variance_normalization=True)
        delta_mfccs = librosa.feature.delta(mfcc).T
        delta_2_mfccs = librosa.feature.delta(mfcc, order=2).T
    
        feats = {}

        mfcc = mfcc.T
        for i in range(mfcc.shape[1]):
            feats[f'mfcc_{i}'] = mfcc[:, i]

        for i in range(delta_mfccs.shape[1]):
            feats[f'delta_mfccs_{i}'] = delta_mfccs[:, i]

        for i in range(delta_2_mfccs.shape[1]):
            feats[f'delta_2_mfccs_{i}'] = delta_2_mfccs[:, i]

        for i in range(normalized_mfcc.shape[1]):
            feats[f'normalized_mfcc_{i}'] = normalized_mfcc[:, i]

        ep_dfs[ep] = feats
    return ep_dfs


def get_chroma(episode_names, video_paths, gt_df):
    ep_dfs = {}
    for ep in episode_names:
        rec, sr, frame_length, hop_length, desired_len = get_audio_librosa(video_paths, ep, gt_df)
        
        chroma = librosa.feature.chroma_stft(y=rec, sr=sr, n_fft=frame_length, hop_length=hop_length, )
        chroma = np.pad(chroma, pad_width=((0, 0), (0, desired_len - chroma.shape[1])))

        chroma = chroma.T 
        ep_dfs[ep] = {}
        for i in range(chroma.shape[1]):
            ep_dfs[ep][f'chroma_{i}'] = chroma[:, i]

    return ep_dfs


def get_spectral_contrast(episode_names, video_paths, gt_df):
    ep_dfs = {}
    for ep in episode_names:
        rec, sr, frame_length, hop_length, desired_len = get_audio_librosa(video_paths, ep, gt_df)
        
        contrast = librosa.feature.spectral_contrast(y=rec, sr=sr, n_fft=frame_length, hop_length=hop_length, )
        contrast = np.pad(contrast, pad_width=((0, 0), (0, desired_len - contrast.shape[1])))

        contrast = contrast.T
        ep_dfs[ep] = {}
        for i in range(contrast.shape[1]):
            ep_dfs[ep][f'chroma_{i}'] = contrast[:, i]
    return ep_dfs


def get_sequences_for_hmm_mfcc(episode_names, video_paths, gt_df, target_cols=['Audio_Pigs', 'Audio_Cook']):
    # feature for real classifier: scorepig, score_nonpig, score_cook, score_noncook, then probably also boolean flags for cook and pig
    n_mfcc = 13

    recordings = {}
    for ep in episode_names:
        rec, sr = librosa.load(video_paths[ep], sr=None)
        recordings[ep] = ((rec, sr))

    sequences_dict = {}
    for target_col in target_cols:
        background_noise_mask = (gt_df[target_col]==0)
        gt_df[f'no_{target_col}'] = 0
        gt_df.loc[background_noise_mask, f'no_{target_col}'] = 1

        #find sequences in gt where target is 1
        sequences = {}
        sequences_dict[target_col] = sequences
        for target in [target_col, f'no_{target_col}']:
            sequences[target] = []
            for ep in episode_names:
                mask = gt_df[gt_df.episode==ep][target]
                sequence = []
                for i in range(0,len(mask)):
                    if mask[i] == 1 and len(sequence) < 30:
                        sequence.append(i)
                    else:
                        if len(sequence) > 5:
                            sequence_start = sequence[0]
                            sequence_end = sequence[-1]
                            rec, sr = recordings[ep]
                            frame_size_ms = 100
                            hop_length = int(1/25 * sr)
                            frame_length = int(frame_size_ms / 1000 * sr)
                            start_audio = int(sequence_start*hop_length)
                            end_audio = int(sequence_end*hop_length)
                            mfcc = librosa.feature.mfcc(y=rec[start_audio:end_audio], sr=sr, n_fft=frame_length, hop_length=hop_length, n_mfcc=n_mfcc)
                            mfcc = np.swapaxes(mfcc, 1,0)
                            sequences[target].append((ep, sequence_start, sequence_end, mfcc))
                            sequence = []
    return sequences_dict

import cv2
import os
import pandas as pd

def load_video(path):
    cap = cv2.VideoCapture(path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = chr(h&0xff) + chr((h>>8)&0xff) + chr((h>>16)&0xff) + chr((h>>24)&0xff) 
    print(f"Loaded video with resolution {frame_width}x{frame_height}")
    print(f"FPS = {fps}, Frame Count = {frame_count}, Codec = {codec}")
    return cap

#TODO: properly release video etc.

def write_frames_with_frame_number(cv2_video, episode_name):
    frame_count = int(cv2_video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in range(frame_count):
        ret, frame = cv2_video.read()
        if ret == True:
            cv2.imwrite(f"../data/frames/{episode_name}/frame_{i:05d}.jpg", frame)
        else:
            print(f"Should never happen, but frame {i} not read properly")
            break 


def get_init_vars(gt_folder, videos_folder):
    episode_names, video_paths = get_episode_names_video_paths(videos_folder)
    gt_df = get_gt_df(episode_names, gt_folder)
    return episode_names, video_paths, gt_df


def get_episode_names_video_paths(videos_folder):
    episode_names = []
    video_paths = {}
    for file in os.listdir(videos_folder):
        if file.endswith(".avi"):
            file_path = os.path.join(videos_folder, file)
            episode_name = file.split('.')[0]
            episode_names.append(episode_name)
            video_paths[episode_name] = file_path
    return episode_names, video_paths


def get_gt_df(episode_names, gt_folder, episode_splits = {
        'Muppets-02-01-01': 19718,
        'Muppets-02-04-04': 19466,
        'Muppets-03-04-03': 19435,
    }):
    gt_dfs = []
    split_counter = 0
    for file in os.listdir(gt_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(gt_folder, file)
            df = pd.read_csv(file_path, sep=";")
            episode_name = file.split('.')[0]

            assert episode_name in episode_names

            df['episode'] = episode_name
            df['episode_split'] = split_counter
            df.loc[df.Frame_number < episode_splits[episode_name], 'episode_split'] = split_counter + 1

            split_counter += 2

            gt_dfs.append(df)
    gt_df = pd.concat(gt_dfs)

    return gt_df
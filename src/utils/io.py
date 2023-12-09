import cv2

def load_video(path):
    cap = cv2.VideoCapture(path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
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


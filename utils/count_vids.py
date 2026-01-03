import glob
import os

VIDEO_DIR = "D:/APCS/2025-2026/Computer_Vision/PROJECT/CS412-CV-FinalProject/sutd-traffic-video-qa/videos_obj_tracking/videos_obj_tracking"

if __name__ == "__main__":
    video_files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    print(f"Total number of videos: {len(video_files)}")
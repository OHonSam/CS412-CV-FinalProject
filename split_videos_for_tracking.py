from pathlib import Path
from loguru import logger
import glob
import os
import shutil

DATA_ROOT = Path("D:/APCS/2025-2026/Computer_Vision/PROJECT/CS412-CV-FinalProject/sutd-traffic-video-qa")
# VIDEO_DIR = Path(f"{DATA_ROOT}/videos/SUTD/videos")
VIDEO_DIR = Path(f"{DATA_ROOT}/videos_partitioned/videos_1")
VIDEO_OBJ_TRACKING_DIR = Path(f"{DATA_ROOT}/videos_obj_tracking/videos_obj_tracking")
NUM_PARTITIONS = 4


if __name__ == "__main__":
    # load videos from VIDEO_DIR
    total_video_files = glob.glob(Path(VIDEO_DIR, "*.mp4").as_posix())
    unprocessed_video_files = []

    for i in range(NUM_PARTITIONS):
        os.makedirs(DATA_ROOT / "videos_partitioned" / f"videos_{i+1}", exist_ok=True)

    for video_file in total_video_files:

        if (VIDEO_OBJ_TRACKING_DIR / Path(video_file).name).exists():
            logger.info(f"Skip processed videos: {video_file}")
            continue
        
        unprocessed_video_files.append(video_file)

    # divide unprocessed videos into three parts
    for i in range(len(unprocessed_video_files)):
        for mod_res in range(0, NUM_PARTITIONS):
            if i % NUM_PARTITIONS == mod_res:
                part_dir = DATA_ROOT / "videos_partitioned" / f"videos_{mod_res+1}"
                break
            
        dest_path = part_dir / Path(unprocessed_video_files[i]).name
        os.rename(unprocessed_video_files[i], dest_path)
        logger.info(f"Moved {unprocessed_video_files[i]} to {dest_path}")
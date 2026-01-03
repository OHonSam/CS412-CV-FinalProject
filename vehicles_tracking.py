import os
import yaml
import cv2
import numpy as np
import glob
import subprocess
from collections import defaultdict
from ultralytics import YOLO
from argparse import ArgumentParser
from pathlib import Path
from loguru import logger

DATA_ROOT = "D:/APCS/2025-2026/Computer_Vision/PROJECT/CS412-CV-FinalProject/sutd-traffic-video-qa"
VIDEO_DIR = f"{DATA_ROOT}/videos"
QUESTIONS_DIR = f"{DATA_ROOT}/questions/questions"
VIDEO_OBJ_TRACKING_DIR = f"{DATA_ROOT}/videos_obj_tracking/videos_obj_tracking"
FFMPEG_PATH = r"C:\Users\Admin\miniconda3\envs\zalo_aic\Library\bin\ffmpeg.exe"

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def load_model_weights(weights_path, device):
    model = YOLO(weights_path).to(device)
    return model


def object_tracking(model, video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    # Get video properties for VideoWriter
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Define colors for different classes (you can customize)
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)

    frame_count = 0

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO11 tracking on the frame, persisting tracks between frames
            result = model.track(frame, persist=True, tracker="botsort.yaml")[0]

            # Get the boxes and track IDs
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
                classes = result.boxes.cls.cpu().numpy().astype(int)
                track_ids = (
                    result.boxes.id.int().cpu().tolist()
                    if result.boxes.id is not None
                    else [None] * len(boxes)
                )

                # Draw boxes and class names manually (no ID, no confidence)
                for box, cls, track_id in zip(boxes, classes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = model.names[cls]
                    color = tuple(map(int, colors[cls]))

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw class name only (no ID, no confidence)
                    label = class_name
                    (label_w, label_h), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                    # Track history for drawing lines
                    if track_id is not None:
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        track = track_history[track_id]
                        track.append((cx, cy))
                        if len(track) > 30:
                            track.pop(0)

                        # Draw tracking lines
                        points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(
                            frame,
                            [points],
                            isClosed=False,
                            color=(230, 230, 230),
                            thickness=2,
                        )

            # Write frame to output video
            out.write(frame)
            frame_count += 1
        else:
            # Break the loop if the end of the video is reached
            break

    # Release resources
    cap.release()
    out.release()
    print(f"Processed {frame_count} frames. Output saved to: {output_path}")


def convert_into_h264(input_path, output_path, bitrate="5M"):
    """
    Convert video to H.264 using Windows Media Foundation (h264_mf).
    Compatible with Conda FFmpeg builds on Windows.
    Returns True if successful.
    """
    try:
        subprocess.run(
            [
                FFMPEG_PATH,
                "-y",
                "-i", input_path,
                "-c:v", "h264_mf",
                "-b:v", bitrate,
                output_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"Converted {input_path} → {output_path} (H.264, h264_mf)")
        return True

    except subprocess.CalledProcessError as e:
        logger.error("FFmpeg conversion failed")
        logger.error(e.stderr)
        return False

    except FileNotFoundError:
        logger.error("FFmpeg not found. Ensure FFmpeg is installed and in PATH.")
        return False


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="D:/APCS/2025-2026/Computer_Vision/PROJECT/CS412-CV-FinalProject/sutd-traffic-video-qa/configs/configs/config.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="D:/APCS/2025-2026/Computer_Vision/PROJECT/CS412-CV-FinalProject/sutd-traffic-video-qa/videos_obj_tracking/videos_obj_tracking",
        help="Directory to save output videos",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    model_weights = config["model"]["yolo_weights"]
    device = config["model"]["device"]
    model = load_model_weights(model_weights, device)

    videos_dir = VIDEO_DIR
    videos_obj_tracking = VIDEO_OBJ_TRACKING_DIR
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    video_files = glob.glob(Path(videos_dir, "*.mp4").as_posix())

    for video_file in video_files:
        video_filename = Path(video_file).name

        if (Path(videos_obj_tracking) / video_filename).exists():
            logger.info(f"Skipping already processed video: {video_filename}")
            continue

        output_path = Path(output_dir) / video_filename
        logger.info(f"Processing video: {video_filename}")

        object_tracking(model, video_file, output_path.as_posix())

        # Convert to H.264 format
        # Keep the same filename and delete the output_path after conversion
        h264_output_path = output_path.with_name(output_path.stem + "_h264.mp4")
        if convert_into_h264(output_path.as_posix(), h264_output_path.as_posix()):
            os.remove(output_path.as_posix())
            logger.info(f"Removed intermediate file: {output_path.as_posix()}")

            # Rename H.264 file to original output path
            os.rename(h264_output_path.as_posix(), output_path.as_posix())
            logger.info(f"Renamed H.264 video to: {output_path.as_posix()}")
        else:
            logger.warning(
                f"H.264 conversion failed. Keeping original mp4v file: {output_path.as_posix()}"
            )
            os.remove(output_path.as_posix())
            break

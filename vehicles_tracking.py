import os
import yaml
import cv2
import numpy as np
import glob
from collections import defaultdict
from ultralytics import YOLO
from argparse import ArgumentParser
from pathlib import Path
from loguru import logger


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_model_weights(weights_path, device):
    model = YOLO(weights_path).to(device)
    return model

def object_tracking(model, video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    logger.info(f"Opened video file: {video_path}")

    # Get video properties for VideoWriter
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Store the track history
    track_history = defaultdict(lambda: [])

    frame_count = 0

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO11 tracking on the frame, persisting tracks between frames
            result = model.track(frame, persist=True)[0]

            # Get the boxes and track IDs
            if result.boxes and result.boxes.is_track:
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist()

                # Visualize the result on the frame
                frame = result.plot()

                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

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

def convert_into_h264(input_path, output_path):
    command = f"ffmpeg -i {input_path} -c:v libx264 -preset fast -crf 22 -c:a copy {output_path}"
    os.system(command)
    logger.info(f"Converted {input_path} to H.264 format at {output_path}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the config file')
    parser.add_argument('--output_dir', type=str, default='/datastore/clc_hcmus/ZaAIC/CS412-CV-FinalProject/SUTD/videos_obj_tracking', help='Directory to save output videos')
    args = parser.parse_args()

    config = load_config(args.config)

    model_weights = config['model']['yolo_weights']
    device = config['model']['device']
    model = load_model_weights(model_weights, device)

    videos_dir = config['dataset']['SUTD']['videos_dir']
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    video_files = glob.glob(Path(videos_dir, '*.mp4').as_posix())

    for video_file in video_files:
        video_filename = Path(video_file).name
        output_path = Path(output_dir) / video_filename
        logger.info(f"Processing video: {video_filename}")
        object_tracking(model, video_file, output_path.as_posix())
        
        # Convert to H.264 format
        # Keep the same filename and delete the output_path after conversion
        h264_output_path = output_path.with_name(output_path.stem + '_h264.mp4')
        convert_into_h264(output_path.as_posix(), h264_output_path.as_posix())
        os.remove(output_path.as_posix())
        logger.info(f"Removed intermediate file: {output_path.as_posix()}")

        # Rename H.264 file to original output path
        os.rename(h264_output_path.as_posix(), output_path.as_posix())
        logger.info(f"Saved H.264 video to: {output_path.as_posix()}")


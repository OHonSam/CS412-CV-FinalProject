from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
from loguru import logger
from pathlib import Path
import os

VIDEO_FOLDER = "videos"
OUTPUT_FOLDER = "keyframes"
NUM_KEYFRAMES = 5


def extract_keyframes(video_path, output_dir, num_keyframes):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_path = Path(output_dir) / Path(video_path).stem
    output_path = output_path.as_posix()
    os.makedirs(output_path, exist_ok=True)

    vd = Video()
    diskwriter = KeyFrameDiskWriter(location=output_path)
    vd.extract_video_keyframes(
        no_of_frames=num_keyframes,
        file_path=video_path,
        writer=diskwriter,
    )

    print(f"Keyframes saved in: {output_path}")

    keyframe_paths = []

    for filename in os.listdir(output_path):
        keyframe_paths.append(os.path.join(output_path, filename))

    return keyframe_paths


if __name__ == "__main__":
    for video_file in os.listdir(VIDEO_FOLDER):
        video_path = os.path.join(VIDEO_FOLDER, video_file)
        output_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(video_file)[0])
        extract_keyframes(video_path, output_path, num_keyframes=NUM_KEYFRAMES)

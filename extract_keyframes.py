from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
import os

VIDEO_FOLDER = "videos"
OUTPUT_FOLDER = "keyframes"
NUM_KEYFRAMES = 5

def extract_keyframes(video_path, output_path, num_keyframes):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    vd = Video()
    diskwriter = KeyFrameDiskWriter(location=output_path)
    vd.extract_video_keyframes(
        no_of_frames=num_keyframes,
        file_path=video_path,
        writer=diskwriter,
    )

    print(f"Keyframes saved in: {output_path}")
    
    if os.path.exists("clipped"):
        os.rmdir("clipped")

    for filename in os.listdir(output_path):
        new_name = filename.split("_", 1)[-1]
        os.rename(
            os.path.join(output_path, filename),
            os.path.join(output_path, new_name)
        )

if __name__ == "__main__":
    for video_file in os.listdir(VIDEO_FOLDER):
        video_path = os.path.join(VIDEO_FOLDER, video_file)
        output_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(video_file)[0])
        extract_keyframes(video_path, output_path, num_keyframes=NUM_KEYFRAMES)

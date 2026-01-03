import os
import subprocess
import glob
from pathlib import Path
from loguru import logger

# Configuration
VIDEO_DIR = "D:/APCS/2025-2026/Computer_Vision/PROJECT/CS412-CV-FinalProject/sutd-traffic-video-qa/videos_obj_tracking/videos_obj_tracking"
FFMPEG_PATH = r"C:\Users\Admin\miniconda3\envs\zalo_aic\Library\bin\ffmpeg.exe"
FFPROBE_PATH = r"C:\Users\Admin\miniconda3\envs\zalo_aic\Library\bin\ffprobe.exe"


def check_video_health(video_path):
    """
    Check if a video file is valid and playable.
    Returns: (is_valid, codec, error_message)
    """
    try:
        # Use ffprobe to check video metadata
        result = subprocess.run(
            [
                FFPROBE_PATH,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,width,height,duration",
                "-of", "csv=p=0",
                video_path
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return False, None, result.stderr.strip()
        
        output = result.stdout.strip()
        if not output:
            return False, None, "No video stream found"
        
        parts = output.split(",")
        codec = parts[0] if parts else "unknown"
        
        # Additional check: try to decode a few frames
        decode_result = subprocess.run(
            [
                FFMPEG_PATH,
                "-v", "error",
                "-i", video_path,
                "-vframes", "5",
                "-f", "null",
                "-"
            ],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if decode_result.returncode != 0 or decode_result.stderr:
            return False, codec, decode_result.stderr.strip() or "Decode error"
        
        return True, codec, None
        
    except subprocess.TimeoutExpired:
        return False, None, "Timeout while checking video"
    except FileNotFoundError:
        return False, None, "ffprobe/ffmpeg not found"
    except Exception as e:
        return False, None, str(e)


def convert_to_h264(input_path, output_path=None):
    """
    Convert video to H.264 format.
    If output_path is None, creates a temp file and replaces original.
    Returns True if successful.
    """
    if output_path is None:
        output_path = str(Path(input_path).with_suffix(".h264_temp.mp4"))
        replace_original = True
    else:
        replace_original = False
    
    try:
        # Use openh264 encoder (available in non-GPL FFmpeg builds)
        # Or use h264_mf (Windows Media Foundation) as fallback
        result = subprocess.run(
            [
                FFMPEG_PATH,
                "-y",
                "-i", input_path,
                "-c:v", "libopenh264",  # Changed from libx264
                "-b:v", "2M",           # Use bitrate instead of CRF
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                output_path
            ],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        # If libopenh264 fails, try h264_mf (Windows Media Foundation)
        if result.returncode != 0 and "libopenh264" in result.stderr:
            logger.warning("libopenh264 not available, trying h264_mf...")
            result = subprocess.run(
                [
                    FFMPEG_PATH,
                    "-y",
                    "-i", input_path,
                    "-c:v", "h264_mf",
                    "-b:v", "2M",
                    "-c:a", "aac",
                    "-b:a", "128k",
                    output_path
                ],
                capture_output=True,
                text=True,
                timeout=600
            )
        
        if result.returncode != 0:
            logger.error(f"Conversion failed: {result.stderr}")
            if os.path.exists(output_path):
                os.remove(output_path)
            return False
        
        # Verify the converted file
        is_valid, codec, error = check_video_health(output_path)
        if not is_valid:
            logger.error(f"Converted file is invalid: {error}")
            if os.path.exists(output_path):
                os.remove(output_path)
            return False
        
        # Replace original if needed
        if replace_original:
            os.remove(input_path)
            os.rename(output_path, input_path)
            logger.info(f"Replaced original with H.264 version: {input_path}")
        
        return True
        
    except subprocess.TimeoutExpired:
        logger.error(f"Conversion timeout: {input_path}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False
    except Exception as e:
        logger.error(f"Conversion error: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def process_video_directory(video_dir, fix_corrupted=True):
    """
    Check all videos in directory and optionally fix corrupted ones.
    """
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    
    results = {
        "healthy": [],
        "corrupted": [],
        "fixed": [],
        "failed": []
    }
    
    logger.info(f"Found {len(video_files)} video files to check")
    
    for i, video_path in enumerate(video_files, 1):
        filename = os.path.basename(video_path)
        logger.info(f"[{i}/{len(video_files)}] Checking: {filename}")
        
        is_valid, codec, error = check_video_health(video_path)
        
        if is_valid:
            status = "✓ Healthy"
            if codec != "h264":
                status += f" (codec: {codec})"
            logger.info(f"  {status}")
            results["healthy"].append(video_path)
        else:
            logger.warning(f"  ✗ Corrupted/Unplayable: {error}")
            results["corrupted"].append(video_path)
            
            if fix_corrupted:
                logger.info(f"  → Attempting to fix...")
                if convert_to_h264(video_path):
                    logger.info(f"  ✓ Fixed successfully")
                    results["fixed"].append(video_path)
                else:
                    logger.error(f"  ✗ Could not fix")
                    logger.info(f"  → Deleting corrupted file")
                    os.remove(video_path)
                    results["failed"].append(video_path)
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total videos checked: {len(video_files)}")
    print(f"Healthy: {len(results['healthy'])}")
    print(f"Corrupted found: {len(results['corrupted'])}")
    if fix_corrupted:
        print(f"Successfully fixed: {len(results['fixed'])}")
        print(f"Failed to fix: {len(results['failed'])}")
        
        if results["failed"]:
            print("\nFailed files:")
            for f in results["failed"]:
                print(f"  - {os.path.basename(f)}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check and fix video files")
    parser.add_argument(
        "--video_dir",
        type=str,
        default=VIDEO_DIR,
        help="Directory containing video files"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check videos, don't fix them"
    )
    parser.add_argument(
        "--single",
        type=str,
        help="Check/fix a single video file"
    )
    
    args = parser.parse_args()
    
    if args.single:
        # Process single file
        is_valid, codec, error = check_video_health(args.single)
        if is_valid:
            print(f"✓ Video is healthy (codec: {codec})")
        else:
            print(f"✗ Video is corrupted: {error}")
            if not args.check_only:
                print("Attempting to fix...")
                if convert_to_h264(args.single):
                    print("✓ Fixed successfully")
                else:
                    print("✗ Could not fix")
    else:
        # Process directory
        process_video_directory(args.video_dir, fix_corrupted=not args.check_only)

# python video_health_check.py --video_dir "D:/APCS/2025-2026/Computer_Vision/PROJECT/CS412-CV-FinalProject/sutd-traffic-video-qa/videos_obj_tracking/videos_obj_tracking"
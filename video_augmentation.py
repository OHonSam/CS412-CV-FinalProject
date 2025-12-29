import cv2
import numpy as np
from pathlib import Path
from typing import Literal, Optional, Tuple
import os

class VideoEnvironmentalDegradation:
    """
    Apply environmental degradation effects to videos with gradual intensity changes.
    Supports: rain, snow, fog, darkness, motion_blur, and combined effects.
    """
    
    def __init__(self):
        self.supported_effects = ['rain', 'snow', 'fog', 'darkness', 'motion_blur', 'combined']
    
    def _add_rain(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """Add rain streaks with given intensity (0.0 to 1.0)"""
        if intensity <= 0:
            return frame
        
        frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Number of rain drops proportional to intensity
        num_drops = int(intensity * 1000)
        
        for _ in range(num_drops):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            length = np.random.randint(10, 30)
            thickness = 1
            
            # Rain streak angle (slightly angled)
            angle = np.random.uniform(-15, -5)
            end_x = int(x + length * np.sin(np.radians(angle)))
            end_y = int(y + length * np.cos(np.radians(angle)))
            
            # Clip to frame bounds
            end_x = np.clip(end_x, 0, w-1)
            end_y = np.clip(end_y, 0, h-1)
            
            # Draw rain streak
            color = (200, 200, 200)  # Light gray
            cv2.line(frame, (x, y), (end_x, end_y), color, thickness)
        
        # Add slight blur to simulate rain atmosphere
        if intensity > 0.3:
            frame = cv2.GaussianBlur(frame, (3, 3), 0)
        
        return frame
    
    def _add_snow(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """Add snow particles with given intensity (0.0 to 1.0)"""
        if intensity <= 0:
            return frame
        
        frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Create snow mask
        num_flakes = int(intensity * 2000)
        
        for _ in range(num_flakes):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            size = np.random.randint(1, 4)
            
            # Draw snowflake (small circle)
            color = (255, 255, 255)
            cv2.circle(frame, (x, y), size, color, -1)
        
        # Add slight brightness to simulate snow reflection
        if intensity > 0.5:
            frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
        
        return frame
    
    def _add_fog(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """Add fog/haze effect with given intensity (0.0 to 1.0)"""
        if intensity <= 0:
            return frame
        
        # Create white fog overlay
        fog = np.ones_like(frame) * 255
        
        # Blend original frame with fog
        blended = cv2.addWeighted(frame, 1 - intensity * 0.7, fog, intensity * 0.7, 0)
        
        return blended.astype(np.uint8)
    
    def _add_darkness(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """Reduce brightness with given intensity (0.0 to 1.0)"""
        if intensity <= 0:
            return frame
        
        # Reduce brightness: intensity 0 = no change, intensity 1 = very dark
        brightness_factor = 1.0 - (intensity * 0.8)  # Keep at least 20% brightness
        darkened = (frame * brightness_factor).astype(np.uint8)
        
        return darkened
    
    def _add_motion_blur(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """Add motion blur with given intensity (0.0 to 1.0)"""
        if intensity <= 0:
            return frame
        
        # Kernel size based on intensity
        kernel_size = int(5 + intensity * 20)  # 5 to 25
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create motion blur kernel (horizontal motion)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel /= kernel_size
        
        # Apply blur
        blurred = cv2.filter2D(frame, -1, kernel)
        
        return blurred
    
    def _add_combined(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """Add combined effects (darkness + rain + slight blur)"""
        if intensity <= 0:
            return frame
        
        # Apply multiple effects with scaled intensities
        frame = self._add_darkness(frame, intensity * 0.6)
        frame = self._add_rain(frame, intensity * 0.8)
        frame = self._add_motion_blur(frame, intensity * 0.3)
        
        return frame
    
    def process_video(
        self,
        input_path: str,
        output_path: str,
        effect_type: Literal['rain', 'snow', 'fog', 'darkness', 'motion_blur', 'combined'] = 'rain',
        mode: Literal['gradual', 'consistent', 'segment'] = 'gradual',
        max_intensity: float = 1.0,
        segment_config: Optional[Tuple[float, float, float]] = None
    ) -> bool:
        """
        Apply environmental degradation to a video.
        
        Args:
            input_path: Path to input video file
            output_path: Path to output video file
            effect_type: Type of degradation effect
            mode: Degradation mode:
                - 'gradual': Intensity increases from 0 to max_intensity over video
                - 'consistent': Same intensity throughout (uses max_intensity)
                - 'segment': Different intensities for 3 segments (use segment_config)
            max_intensity: Maximum intensity (0.0 to 1.0)
            segment_config: Tuple of 3 intensities for segment mode (start, middle, end)
        
        Returns:
            True if successful, False otherwise
        """
        
        if effect_type not in self.supported_effects:
            print(f"Error: Unsupported effect type '{effect_type}'")
            print(f"Supported effects: {self.supported_effects}")
            return False
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file '{input_path}'")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        print(f"Processing video: {input_path}")
        print(f"Properties: {width}x{height} @ {fps}fps, {total_frames} frames")
        print(f"Effect: {effect_type}, Mode: {mode}, Max intensity: {max_intensity}")
        
        # Create output video writer
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"Error: Cannot create output video file '{output_path}'")
            cap.release()
            return False
        
        # Get effect function
        effect_func = getattr(self, f'_add_{effect_type}')
        
        # Process frames
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate intensity based on mode
            if mode == 'gradual':
                # Linear progression from 0 to max_intensity
                intensity = (frame_idx / total_frames) * max_intensity
            
            elif mode == 'consistent':
                # Same intensity throughout
                intensity = max_intensity
            
            elif mode == 'segment':
                # Three segments with different intensities
                if segment_config is None:
                    segment_config = (0.0, 0.7, 0.3)  # Default: clean, heavy, light
                
                segment_size = total_frames // 3
                if frame_idx < segment_size:
                    intensity = segment_config[0]
                elif frame_idx < 2 * segment_size:
                    intensity = segment_config[1]
                else:
                    intensity = segment_config[2]
            
            # Apply effect
            degraded_frame = effect_func(frame, intensity)
            
            # Write frame
            out.write(degraded_frame)
            
            frame_idx += 1
            
            # Progress indicator
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames} frames)", end='\r')
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"\nVideo processing complete! Output saved to: {output_path}")
        return True


# Example usage functions
def example_gradual_rain(input_video: str, output_video: str):
    """Apply gradually increasing rain effect"""
    degrader = VideoEnvironmentalDegradation()
    degrader.process_video(
        input_path=input_video,
        output_path=output_video,
        effect_type='rain',
        mode='gradual',
        max_intensity=0.8
    )

def example_consistent_darkness(input_video: str, output_video: str):
    """Apply consistent darkness throughout video"""
    degrader = VideoEnvironmentalDegradation()
    degrader.process_video(
        input_path=input_video,
        output_path=output_video,
        effect_type='darkness',
        mode='consistent',
        max_intensity=0.6
    )

def example_segment_effects(input_video: str, output_video: str):
    """Apply different intensities in 3 segments: clean -> foggy -> light fog"""
    degrader = VideoEnvironmentalDegradation()
    degrader.process_video(
        input_path=input_video,
        output_path=output_video,
        effect_type='fog',
        mode='segment',
        segment_config=(0.0, 0.8, 0.3)  # (first 1/3, middle 1/3, last 1/3)
    )

def example_combined_effects(input_video: str, output_video: str):
    """Apply combined effects (darkness + rain + blur)"""
    degrader = VideoEnvironmentalDegradation()
    degrader.process_video(
        input_path=input_video,
        output_path=output_video,
        effect_type='combined',
        mode='gradual',
        max_intensity=0.7
    )


# Batch processing helper
def batch_process_videos(
    input_dir: str,
    output_dir: str,
    effect_type: str = 'rain',
    mode: str = 'gradual'
):
    """
    Process all videos in a directory with the same effect.
    
    Args:
        input_dir: Directory containing input videos
        output_dir: Directory for output videos
        effect_type: Type of effect to apply
        mode: Degradation mode
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_path.glob(f'*{ext}'))
    
    print(f"Found {len(video_files)} videos to process")
    
    degrader = VideoEnvironmentalDegradation()
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_file.name}")
        
        output_file = output_path / f"{video_file.stem}_{effect_type}{video_file.suffix}"
        
        degrader.process_video(
            input_path=str(video_file),
            output_path=str(output_file),
            effect_type=effect_type,
            mode=mode,
            max_intensity=0.7
        )


if __name__ == "__main__":
    # Example usage
    print("Video Environmental Degradation Tool")
    print("=" * 50)
    print("\nExample 1: Gradual rain effect")
    # example_gradual_rain('SUTD/videos/b_1a4411B7sb_clip_005.mp4', 'output_rain.mp4')
    
    print("\nExample 2: Consistent darkness")
    example_consistent_darkness('SUTD/videos/b_1A4411H7RU_clip_009.mp4', 'output_dark.mp4')
    
    print("\nExample 3: Segment-based fog")
    print("example_segment_effects('input.mp4', 'output_fog.mp4')")
    
    print("\nExample 4: Combined effects")
    print("example_combined_effects('input.mp4', 'output_combined.mp4')")
    
    print("\nExample 5: Batch processing")
    print("batch_process_videos('input_videos/', 'output_videos/', 'snow', 'gradual')")
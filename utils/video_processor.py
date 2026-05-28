"""
Video processing module.
Handles video upload, frame extraction, timeline image generation.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import uuid


class VideoProcessor:
    """Process video files: extract frames, generate timeline images, get metadata."""

    def __init__(self):
        self.ffmpeg_available = self._check_ffmpeg()

    @staticmethod
    def _check_ffmpeg() -> bool:
        """Check if ffmpeg is available."""
        import shutil
        return shutil.which("ffmpeg") is not None

    def get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds."""
        try:
            import ffmpeg
            probe = ffmpeg.probe(video_path)
            duration = float(probe["format"]["duration"])
            return duration
        except Exception as e:
            print(f"[VideoProcessor] Error getting duration: {e}")
            return 0.0

    def get_video_resolution(self, video_path: str) -> Tuple[int, int]:
        """Get video resolution (width, height)."""
        try:
            import ffmpeg
            probe = ffmpeg.probe(video_path)
            video_stream = next(
                s for s in probe["streams"] if s["codec_type"] == "video"
            )
            return (int(video_stream["width"]), int(video_stream["height"]))
        except Exception as e:
            print(f"[VideoProcessor] Error getting resolution: {e}")
            return (1920, 1080)

    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        interval: float = 5.0,
        max_frames: Optional[int] = None,
    ) -> List[str]:
        """
        Extract frames from video at regular intervals.
        
        Returns:
            List of paths to extracted frame images.
        """
        os.makedirs(output_dir, exist_ok=True)

        try:
            import ffmpeg
        except ImportError:
            raise ImportError("ffmpeg-python not installed. Run: pip install ffmpeg-python")

        duration = self.get_video_duration(video_path)
        if duration <= 0:
            return []

        # Calculate timestamps
        timestamps = []
        t = 0.0
        while t < duration:
            timestamps.append(t)
            t += interval

        if max_frames and len(timestamps) > max_frames:
            step = len(timestamps) // max_frames
            timestamps = timestamps[::step]

        frame_paths = []
        for i, ts in enumerate(timestamps):
            output_path = os.path.join(output_dir, f"frame_{i:06d}_{ts:.2f}s.jpg")
            try:
                (
                    ffmpeg
                    .input(video_path, ss=ts)
                    .output(
                        output_path,
                        vframes=1,
                        qscale=2,
                        format="image2",
                    )
                    .overwrite_output()
                    .run(quiet=True, capture_stdout=True, capture_stderr=True)
                )
                frame_paths.append(output_path)
            except Exception as e:
                print(f"[VideoProcessor] Error extracting frame at {ts}s: {e}")

        print(f"[VideoProcessor] Extracted {len(frame_paths)} frames from video.")
        return frame_paths

    def extract_frame_at_time(self, video_path: str, timestamp: float, output_path: str) -> str:
        """Extract a single frame at a specific timestamp."""
        try:
            import ffmpeg
            (
                ffmpeg
                .input(video_path, ss=timestamp)
                .output(output_path, vframes=1, qscale=2)
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
            return output_path
        except Exception as e:
            print(f"[VideoProcessor] Error extracting frame at {timestamp}s: {e}")
            return ""

    def extract_clip(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        output_path: str,
    ) -> str:
        """
        Extract a video clip from start_time to end_time.
        
        Returns:
            Path to the output clip.
        """
        try:
            import ffmpeg
            duration = end_time - start_time
            (
                ffmpeg
                .input(video_path, ss=start_time, t=duration)
                .output(output_path, c="copy")
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
            return output_path
        except Exception as e:
            print(f"[VideoProcessor] Error extracting clip: {e}")
            return ""

    def extract_audio(self, video_path: str, output_path: str) -> str:
        """Extract audio from video."""
        try:
            import ffmpeg
            (
                ffmpeg
                .input(video_path)
                .output(output_path, acodec="pcm_s16le", ac=1, ar=16000)
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
            return output_path
        except Exception as e:
            print(f"[VideoProcessor] Error extracting audio: {e}")
            return ""

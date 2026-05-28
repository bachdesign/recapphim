"""
Scene detection module.
Splits video into timeline scenes using content-aware detection.
Supports:
- PySceneDetect (content-aware scene transitions)
- Histogram-based detection (fallback)
- Fixed-interval splitting (simple mode)
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np


class SceneDetector:
    """
    Detect scene boundaries in a video file.
    Returns a list of scenes with start/end timestamps.
    """

    def __init__(self, method: str = "auto", threshold: float = 30.0):
        """
        Args:
            method: "pyscenedetect", "histogram", "fixed", or "auto" (prefers pyscenedetect)
            threshold: Sensitivity threshold (lower = more scenes detected)
        """
        self.method = method
        self.threshold = threshold

    def detect_scenes(
        self, video_path: str, min_scene_duration: float = 3.0, max_scenes: int = 60
    ) -> List[Dict]:
        """
        Detect all scene boundaries in the video.

        Args:
            video_path: Path to the video file
            min_scene_duration: Minimum scene duration in seconds
            max_scenes: Maximum number of scenes to return

        Returns:
            List of dicts with keys: scene_id, start, end, duration
        """
        method = self.method

        # Try PySceneDetect first if auto
        if method in ("auto", "pyscenedetect"):
            scenes = self._detect_pyscenedetect(video_path)
            if scenes:
                print(f"[SceneDetector] PySceneDetect found {len(scenes)} scenes")
                scenes = self._filter_scenes(scenes, min_scene_duration, max_scenes)
                return scenes

        # Fallback to histogram-based detection
        if method in ("auto", "histogram"):
            print("[SceneDetector] Falling back to histogram-based detection")
            scenes = self._detect_histogram(video_path)
            scenes = self._filter_scenes(scenes, min_scene_duration, max_scenes)
            if scenes:
                return scenes

        # Final fallback: fixed-interval splitting
        print(f"[SceneDetector] Falling back to fixed-interval splitting")
        scenes = self._detect_fixed(video_path, max_scenes)
        return scenes

    def _detect_pyscenedetect(self, video_path: str) -> List[Dict]:
        """Use PySceneDetect library for content-aware scene detection."""
        try:
            from scenedetect import detect, ContentDetector
        except ImportError:
            print("[SceneDetector] PySceneDetect not installed. Install with: pip install scenedetect")
            return []

        try:
            scene_list = detect(video_path, ContentDetector(threshold=self.threshold))

            scenes = []
            for i, (start, end) in enumerate(scene_list):
                start_sec = start.get_seconds()
                end_sec = end.get_seconds()
                scenes.append({
                    "scene_id": i + 1,
                    "start": round(start_sec, 2),
                    "end": round(end_sec, 2),
                    "duration": round(end_sec - start_sec, 2),
                })

            return scenes

        except Exception as e:
            print(f"[SceneDetector] PySceneDetect error: {e}")
            return []

    def _detect_histogram(self, video_path: str) -> List[Dict]:
        """
        Detect scene changes using histogram comparison between consecutive frames.
        Uses OpenCV to compute color histogram differences.
        """
        try:
            import cv2
        except ImportError:
            print("[SceneDetector] OpenCV not available for histogram detection")
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("[SceneDetector] Cannot open video file")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        if duration <= 0:
            cap.release()
            return []

        # Sample frames at ~1 fps for histogram comparison
        sample_interval = max(1, int(fps))
        prev_hist = None
        scene_boundaries = [0.0]  # Always start at 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                # Compute HSV histogram
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

                if prev_hist is not None:
                    # Compare histograms using correlation
                    score = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    # Low correlation = scene change
                    if score < 0.4:  # Threshold for scene change detection
                        timestamp = frame_idx / fps
                        scene_boundaries.append(round(timestamp, 2))

                prev_hist = hist

            frame_idx += 1

        cap.release()

        # Add the end of video
        if scene_boundaries[-1] < duration:
            scene_boundaries.append(round(duration, 2))

        # Build scene list
        scenes = []
        for i in range(len(scene_boundaries) - 1):
            start = scene_boundaries[i]
            end = scene_boundaries[i + 1]
            scenes.append({
                "scene_id": i + 1,
                "start": start,
                "end": end,
                "duration": round(end - start, 2),
            })

        print(f"[SceneDetector] Histogram detection found {len(scenes)} scenes")
        return scenes

    def _detect_fixed(self, video_path: str, max_scenes: int = 60) -> List[Dict]:
        """Simple fixed-interval splitting as last resort."""
        try:
            import ffmpeg
        except ImportError:
            raise ImportError("ffmpeg-python not installed")

        probe = ffmpeg.probe(video_path)
        duration = float(probe["format"]["duration"])

        # Target ~8 second scenes
        target_duration = 8.0
        num_scenes = min(max_scenes, max(1, int(duration / target_duration)))
        actual_duration = duration / num_scenes

        scenes = []
        for i in range(num_scenes):
            start = i * actual_duration
            end = min((i + 1) * actual_duration, duration)
            scenes.append({
                "scene_id": i + 1,
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(end - start, 2),
            })

        return scenes

    @staticmethod
    def _filter_scenes(
        scenes: List[Dict], min_duration: float = 3.0, max_scenes: int = 60
    ) -> List[Dict]:
        """Filter and merge scenes that are too short."""
        if not scenes:
            return []

        # Merge very short scenes with neighbors
        filtered = [scenes[0]]
        for scene in scenes[1:]:
            if scene["duration"] < min_duration and filtered:
                # Merge with previous scene
                filtered[-1]["end"] = scene["end"]
                filtered[-1]["duration"] = round(
                    filtered[-1]["end"] - filtered[-1]["start"], 2
                )
            else:
                filtered.append(scene)

        # Re-number scenes
        for i, scene in enumerate(filtered):
            scene["scene_id"] = i + 1

        return filtered[:max_scenes]

    @staticmethod
    def get_scene_subtitles(
        scenes: List[Dict], subtitles: List[Dict]
    ) -> List[Dict]:
        """
        Match subtitles to scenes.
        Returns scenes with subtitles grouped under each scene.

        Each scene dict gets a 'subtitles' key with the subtitle entries
        that fall within that scene's timeframe.
        """
        for scene in scenes:
            scene_subtitles = []
            for sub in subtitles:
                # Check if subtitle overlaps with scene
                if sub["start"] < scene["end"] and sub["end"] > scene["start"]:
                    scene_subtitles.append({
                        "text": sub["text"],
                        "start": round(max(sub["start"] - scene["start"], 0), 2),
                        "end": round(min(sub["end"], scene["end"]) - scene["start"], 2),
                        "original_start": sub["start"],
                        "original_end": sub["end"],
                    })
            scene["subtitles"] = scene_subtitles
            scene["subtitle_text"] = " ".join([s["text"] for s in scene_subtitles])

            # Ensure subtitle_text is not too long to avoid overly long narrations
            if len(scene["subtitle_text"]) > 200:  # Limit to 200 characters
                scene["subtitle_text"] = scene["subtitle_text"][:200] + "..."

        return scenes

    @staticmethod
    def extract_scene_frames(
        video_path: str,
        scenes: List[Dict],
        output_dir: str,
        frames_per_scene: int = 3,
    ) -> List[Dict]:
        """
        Extract representative frames for each scene.
        
        Args:
            video_path: Path to original video
            scenes: List of scene dicts with start/end
            output_dir: Directory to save frame images
            frames_per_scene: Number of frames per scene (1-5)

        Returns:
            scenes with 'frames' key containing frame file paths
        """
        try:
            import ffmpeg
        except ImportError:
            raise ImportError("ffmpeg-python not installed")

        os.makedirs(output_dir, exist_ok=True)

        for scene in scenes:
            scene_frames = []
            start = scene["start"]
            end = scene["end"]
            duration = end - start

            # Distribute frames evenly across the scene
            if frames_per_scene <= 1:
                timestamps = [(start + end) / 2]
            else:
                step = duration / (frames_per_scene + 1)
                timestamps = [start + step * (i + 1) for i in range(frames_per_scene)]

            for j, ts in enumerate(timestamps):
                output_path = os.path.join(
                    output_dir,
                    f"scene_{scene['scene_id']:04d}_frame_{j+1}_{ts:.2f}s.jpg",
                )
                try:
                    (
                        ffmpeg.input(video_path, ss=ts)
                        .output(output_path, vframes=1, qscale=2, format="image2")
                        .overwrite_output()
                        .run(quiet=True, capture_stdout=True, capture_stderr=True)
                    )
                    scene_frames.append(output_path)
                except Exception as e:
                    print(f"[SceneDetector] Error extracting frame at {ts}s: {e}")

            scene["frames"] = scene_frames
            scene["frame_count"] = len(scene_frames)

        return scenes

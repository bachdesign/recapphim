"""
Video editing module.
Combines video clips, audio narration, and subtitles into final commentary video.
Auto-extends clips when narration is longer than the available footage.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from config import CHARS_PER_SECOND
import subprocess


class VideoEditor:
    """Assemble the final commentary video from clips, narration, and subtitles."""

    def __init__(self):
        self._check_ffmpeg()

    @staticmethod
    def _check_ffmpeg():
        """Verify ffmpeg is available."""
        import shutil
        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg and add it to PATH."
            )

    def compose_final_video(
        self,
        original_video: str,
        narration_audio: str,
        script_segments: List[Dict],
        subtitle_data: List[Dict],
        output_path: str,
        highlight_clips: Optional[List[Dict]] = None,
        language: str = "zh-CN",
        enable_blur_mask: bool = False,
        enable_mirror_reflect: bool = False,
        mirror_interval: int = 2,
        speech_rate: int = -10,
    ) -> str:
        """
        Compose the final commentary video by matching narration with video scenes.

        Args:
            original_video: Path to original video
            narration_audio: Path to the full narration audio
            script_segments: List of script segments with keywords
            subtitle_data: Original subtitle data with timestamps
            output_path: Path for the final output video
            highlight_clips: Pre-identified highlight clips
            language: Language code for subtitles
            enable_blur_mask: Apply blur vignette mask
            enable_mirror_reflect: Apply mirror reflection to alternating clips
            mirror_interval: Apply mirror every N clips (2 = every other clip)

        Returns:
            Path to the final video
        """
        # Step 1: Create a temporary directory for processing
        local_temp = Path(__file__).parent.parent / "temp" / "video_edit"
        os.makedirs(local_temp, exist_ok=True)
        work_dir = tempfile.mkdtemp(prefix="video_edit_", dir=str(local_temp))
        self._speech_rate = speech_rate
        # Store the original video path for later use in _adjust_clips_to_duration
        self._original_video_path = original_video

        try:
            # Step 2: Match narration segments with video scenes
            scene_matches = self._match_scenes(
                script_segments, subtitle_data, highlight_clips
            )

            # Step 3: Extract clips, auto-extending if narration is too long
            clip_paths = []
            total_dur = self._get_video_duration(original_video)
            for i, match in enumerate(scene_matches):
                clip_path = os.path.join(work_dir, f"clip_{i:04d}.mp4")
                clip_duration = match["end"] - match["start"]

                # Estimate required duration from the narration text
                seg = script_segments[i] if i < len(script_segments) else {}
                narration = seg.get("narration", "")
                speech_rate = self._speech_rate if hasattr(self, '_speech_rate') else -10
                needed_duration = self._estimate_narration_duration(narration, language, speech_rate)

                if needed_duration > clip_duration:
                    # Try to extend by pushing end forward
                    extra = needed_duration - clip_duration
                    new_end = min(match["end"] + extra * 0.6, total_dur)
                    # Also pull start back
                    new_start = max(0, match["start"] - extra * 0.4)
                    match["start"] = new_start
                    match["end"] = new_end
                    new_dur = match["end"] - match["start"]

                    if new_dur < needed_duration:
                        # Still not enough — will loop in post-processing
                        print(f"[VideoEditor] Clip {i} extended to max: {new_dur:.1f}s (needed {needed_duration:.1f}s), will loop")
                    else:
                        print(f"[VideoEditor] Clip {i} extended: {clip_duration:.1f}s → {new_dur:.1f}s (needed {needed_duration:.1f}s)")

                # Validate scene timestamps before extraction
                if match["start"] >= match["end"] or match["start"] >= total_dur:
                    print(f"[VideoEditor] Scene {i} has invalid timestamps ({match['start']:.1f}s - {match['end']:.1f}s), creating placeholder")
                    # Create a placeholder clip directly
                    self._run_ffmpeg([
                        "ffmpeg", "-f", "lavfi", "-i", "color=c=black:s=1920x1080:d=3:r=30",
                        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                        "-shortest", "-y", clip_path,
                    ], f"placeholder for invalid scene {i}")
                    actual_dur = 3.0
                    clip_paths.append(clip_path)
                    continue

                self._extract_clip(original_video, match["start"], match["end"], clip_path)

                # Validate extracted clip
                actual_dur = self._get_clip_duration(clip_path)
                if actual_dur <= 0:
                    print(f"[VideoEditor] Clip {i} has zero duration ({match['start']:.1f}s-{match['end']:.1f}s), creating placeholder")
                    # Generate a short placeholder
                    placeholder_path = os.path.join(work_dir, f"placeholder_clip_{i:04d}.mp4")
                    self._run_ffmpeg([
                        "ffmpeg", "-f", "lavfi", "-i", "color=c=black:s=1920x1080:d=3:r=30",
                        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                        "-shortest", "-y", placeholder_path,
                    ], f"placeholder for broken clip {i}")
                    clip_path = placeholder_path
                    actual_dur = 3.0
                if actual_dur < needed_duration - 0.5:
                    # Limit the loop count to prevent extremely long clips
                    max_reasonable_loop = 3  # Don't loop more than 3 times
                    loop_count = min(int(needed_duration / actual_dur) + 1, max_reasonable_loop)
                    
                    if loop_count > max_reasonable_loop:
                        print(f"[VideoEditor] Limiting loop count from {int(needed_duration / actual_dur) + 1} to {max_reasonable_loop} to prevent excessively long clip")
                        # Also reduce needed_duration proportionally
                        needed_duration = actual_dur * max_reasonable_loop
                    
                    loop_path = os.path.join(work_dir, f"looped_{i:04d}.mp4")
                    import subprocess as _sp
                    cmd = [
                        "ffmpeg", "-stream_loop", str(loop_count),
                        "-i", clip_path,
                        "-t", str(needed_duration),
                        "-c:v", "libx264", "-preset", "ultrafast",
                        "-c:a", "aac",
                        "-y", loop_path,
                    ]
                    try:
                        _sp.run(cmd, capture_output=True, check=True)
                        clip_path = loop_path
                        print(f"[VideoEditor] Clip {i} looped {loop_count}x to reach {needed_duration:.1f}s")
                    except Exception as e:
                        print(f"[VideoEditor] Loop failed for clip {i}: {e}")
                        pass

                clip_paths.append(clip_path)

            # Step 3.5: Apply mirror reflection to alternating clips
            if enable_mirror_reflect and clip_paths:
                clip_paths = self._apply_mirror_reflection(
                    clip_paths, work_dir, mirror_interval
                )

            # Step 4: If narration audio is provided, calculate its duration
            # and adjust clips accordingly
            if narration_audio and os.path.exists(narration_audio):
                narration_duration = self._get_audio_duration(narration_audio)
                
                # Validate narration duration against original video length
                original_video_duration = self._get_video_duration(original_video)
                max_reasonable_narration = original_video_duration * 1.5  # Allow up to 50% longer
                
                if narration_duration > max_reasonable_narration:
                    print(f"[VideoEditor] Narration duration ({narration_duration:.1f}s) is too long vs original video ({original_video_duration:.1f}s)")
                    print(f"[VideoEditor] Limiting narration duration to {max_reasonable_narration:.1f}s")
                    narration_duration = max_reasonable_narration
                
                self._adjust_clips_to_duration(
                    clip_paths, scene_matches, narration_duration, work_dir
                )

            # Step 5: Concatenate all clips
            concat_path = os.path.join(work_dir, "concatenated.mp4")
            self._concat_clips(clip_paths, concat_path)

            # Step 5.5: Apply blur mask overlay
            if enable_blur_mask:
                blurred_path = os.path.join(work_dir, "blurred.mp4")
                self._apply_blur_mask(concat_path, blurred_path)
                concat_path = blurred_path

            # Step 6: Add narration audio (if provided)
            if narration_audio and os.path.exists(narration_audio):
                temp_output = os.path.join(work_dir, "with_audio.mp4")
                self._merge_audio_video(concat_path, narration_audio, temp_output)

                # Step 7: Add subtitles
                self._add_subtitles(temp_output, script_segments, output_path, language)
            else:
                # Step 7: Add subtitles without narration
                self._add_subtitles(concat_path, script_segments, output_path, language)

            print(f"[VideoEditor] Final video saved to: {output_path}")
            return output_path

        except Exception as e:
            print(f"[VideoEditor] Error composing video: {e}")
            raise
        finally:
            # Clean up temp files
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)
            # Clean up the original video path attribute
            if hasattr(self, '_original_video_path'):
                delattr(self, '_original_video_path')

    @staticmethod
    def _estimate_narration_duration(narration: str, language: str = "zh-CN", rate: int = -10) -> float:
        """
        Estimate how long the narration will take to speak at the given rate.
        rate: TTS speed percentage (-50 to +50, e.g. -10 = 10% slower).
        Considers Chinese chars, English words, and sentence pauses.
        """
        if not narration:
            return 3.0
        # Resolve language code from display name
        lang_code = language
        if language.startswith("中文") or language == "zh-CN":
            lang_code = "zh-CN"
        elif language.startswith("越南") or language == "vi-VN":
            lang_code = "vi-VN"
        elif language.startswith("英语") or language == "en-US":
            lang_code = "en-US"

        base_cps = CHARS_PER_SECOND.get(lang_code, 3.5)
        # Adjust CPS for the rate: -10% rate → speak slower by ~9%
        # Empirical formula: rate% change → ~rate/1.1% change in duration
        rate_factor = 1.0 + (rate / 110.0)
        cps = base_cps / rate_factor

        import re
        chinese = sum(1 for c in narration if '\u4e00' <= c <= '\u9fff')
        eng_words = len(re.findall(r'[a-zA-Z]+', narration))
        duration = (chinese / cps) + (eng_words / (cps * 0.8))
        pauses = sum(1 for c in narration if c in '。！？.!?')
        duration += pauses * 0.3
        
        # Limit the duration to prevent extremely long estimates
        max_reasonable_duration = len(narration) * 2  # Assume max 2 seconds per character as upper bound
        return max(3.0, min(duration, max_reasonable_duration))

    @staticmethod
    def _get_video_duration(video_path: str) -> float:
        """Get video duration in seconds using ffprobe."""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return float(result.stdout.strip())
        except (ValueError, TypeError):
            return 600.0

    @staticmethod
    def _text_similarity(text_a: str, text_b: str) -> float:
        """
        Compute similarity between two Chinese/English texts using character bigrams.
        Returns a score 0-1. Handles both Chinese and English content.
        """
        if not text_a or not text_b:
            return 0.0
        a_lower = text_a.lower()
        b_lower = text_b.lower()

        # Extract meaningful Chinese character bigrams
        def bigrams(t: str):
            chars = [c for c in t if c.strip()]
            return set(chars[i] + chars[i+1] for i in range(len(chars) - 1))

        # Extract English words
        import re
        words_a = set(re.findall(r'[a-zA-Z\u4e00-\u9fff]+', a_lower))
        words_b = set(re.findall(r'[a-zA-Z\u4e00-\u9fff]+', b_lower))

        # Combine: character bigrams for Chinese + words for English
        grams_a = bigrams(a_lower) | words_a
        grams_b = bigrams(b_lower) | words_b

        if not grams_a or not grams_b:
            return 0.0

        intersection = grams_a & grams_b
        # Jaccard-like similarity weighted by intersection size
        return len(intersection) / max(len(grams_a | grams_b), 1)

    def _match_scenes(
        self,
        script_segments: List[Dict],
        subtitle_data: List[Dict],
        highlight_clips: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        Match each script segment with the best video scene.
        Uses BOTH keywords AND full narration text similarity against subtitles.
        """
        if not subtitle_data:
            # No subtitles: distribute segments evenly across video duration
            total_dur = subtitle_data[-1]["end"] if subtitle_data else 600
            seg_dur = total_dur / max(len(script_segments), 1)
            return [
                {"start": i * seg_dur, "end": (i + 1) * seg_dur,
                 "keywords": s.get("keywords", []), "score": 3}
                for i, s in enumerate(script_segments)
            ]

        scene_matches = []

        if highlight_clips:
            # Use highlights as improved candidate pool, then match by similarity
            # Build a pool of available time slots from highlights
            highlight_pool = list(highlight_clips)

        for i, seg in enumerate(script_segments):
            narration = seg.get("narration", "")
            keywords = seg.get("keywords", [])

            if highlight_clips:
                # Find the BEST highlight clip for this segment by similarity
                best_h = None
                best_score = -1
                best_idx = -1
                for j, h in enumerate(highlight_pool):
                    # Score = keyword match + narration similarity
                    kw_score = sum(2 for kw in keywords if kw.lower() in str(h.get("reason", "")).lower())
                    # Also compare narration with the highlight's original subtitle context
                    sub_context = " ".join(
                        s["text"] for s in subtitle_data
                        if s["start"] >= h["start"] and s["end"] <= h["end"]
                    )
                    sim_score = self._text_similarity(narration, sub_context) * 5
                    total = kw_score + sim_score + h.get("score", 3)
                    if total > best_score:
                        best_score = total
                        best_h = h
                        best_idx = j

                if best_h is not None:
                    highlight_pool.pop(best_idx)
                    scene_matches.append({
                        "start": best_h["start"],
                        "end": best_h["end"],
                        "keywords": keywords,
                        "score": best_score,
                        "narration": narration,
                    })
                else:
                    # Fallback: find best match from subtitles
                    match = self._find_best_match(narration, keywords, subtitle_data)
                    scene_matches.append(match)
            else:
                # Match based on narration text AND keywords against subtitles
                match = self._find_best_match(narration, keywords, subtitle_data)
                scene_matches.append(match)

        return scene_matches

    def _find_best_match(
        self, narration: str, keywords: List[str], subtitle_data: List[Dict]
    ) -> Dict:
        """
        Find the best matching subtitle segment for a given narration + keywords.
        Uses both text similarity and keyword matching.
        """
        best_score = 0
        best_segment = {"start": 0, "end": 15, "keywords": keywords, "score": 0}

        # Sliding window over subtitles
        window_size = 5
        for i in range(len(subtitle_data) - window_size + 1):
            window = subtitle_data[i : i + window_size]
            window_text = " ".join([s.get("text", "") for s in window])

            # Score 1: keyword exact match
            kw_score = sum(2 for kw in keywords if kw.lower() in window_text.lower())

            # Score 2: text similarity between narration and subtitle window
            sim_score = self._text_similarity(narration, window_text) * 10

            # Score 3: character/name overlap bonus — important for donghua
            # Extract named entities (Chinese names are usually 2-4 chars)
            import re
            names_narration = set(re.findall(r'[\u4e00-\u9fff]{2,4}', narration))
            names_subtitle = set(re.findall(r'[\u4e00-\u9fff]{2,4}', window_text))
            name_overlap = len(names_narration & names_subtitle) * 3

            score = kw_score + sim_score + name_overlap

            if score > best_score:
                best_score = score
                best_segment = {
                    "start": window[0]["start"],
                    "end": window[-1]["end"],
                    "keywords": keywords,
                    "score": score,
                }

        return best_segment

    @staticmethod
    def _run_ffmpeg(cmd: List[str], description: str = "ffmpeg"):
        """Run an ffmpeg command with proper error handling."""
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"{description} failed (code {result.returncode}):\n"
                f"command: {' '.join(cmd)}\n"
                f"stdout: {result.stdout[-500:]}\n"
                f"stderr: {result.stderr[-500:]}"
            )
        return result

    def _extract_clip(
        self, video_path: str, start: float, end: float, output_path: str
    ):
        """Extract a video clip. Retries with re-encode if stream copy fails or produces broken file."""
        duration = end - start
        # First try with stream copy (fastest)
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-ss", str(start),
            "-t", str(duration),
            "-c", "copy",
            "-avoid_negative_ts", "1",
            "-y",
            output_path,
        ]
        try:
            self._run_ffmpeg(cmd, "clip extraction (copy)")
            # Validate: check if clip has a valid duration
            if self._get_clip_duration(output_path) <= 0.5:
                raise RuntimeError(f"Clip has no duration ({self._get_clip_duration(output_path)}s), retrying with re-encode")
        except RuntimeError:
            # Fallback to re-encode if stream copy fails or produces broken clip
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-ss", str(start),
                "-t", str(duration),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-c:a", "aac",
                "-y",
                output_path,
            ]
            self._run_ffmpeg(cmd, "clip extraction (re-encode)")

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio file duration in seconds."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return float(result.stdout.strip())
        except (ValueError, AttributeError):
            print(f"[VideoEditor] Warning: Could not get audio duration from {audio_path}")
            return 30.0  # fallback duration

    def _adjust_clips_to_duration(
        self,
        clip_paths: List[str],
        scene_matches: List[Dict],
        target_duration: float,
        work_dir: str,
    ):
        """
        Adjust clip durations to match the narration audio length.
        Uses speed adjustment or clip extension as needed.
        """
        total_clip_duration = sum(m["end"] - m["start"] for m in scene_matches)
        if abs(total_clip_duration - target_duration) < 1.0:
            return

        # Get the original video duration to prevent excessive adjustments
        original_video_path = getattr(self, '_original_video_path', None)
        if original_video_path:
            original_duration = self._get_video_duration(original_video_path)
            # Don't allow target duration to be more than 1.5x the original video duration
            max_allowed_duration = min(target_duration, original_duration * 1.5)
            if target_duration > max_allowed_duration:
                print(f"[VideoEditor] Limiting target duration from {target_duration:.1f}s to {max_allowed_duration:.1f}s")
                target_duration = max_allowed_duration

        ratio = target_duration / total_clip_duration if total_clip_duration > 0 else 1.0
        import subprocess as _sp

        # Cap the ratio to prevent extreme speed changes
        ratio = max(0.3, min(3.0, ratio))  # Limit to 30% to 300% of original speed

        for i, clip_path in enumerate(clip_paths):
            adjusted_path = os.path.join(work_dir, f"adjusted_{i:04d}.mp4")

            if 0.7 <= ratio <= 1.5:
                # Gentle speed adjustment — use setpts
                speed = 1.0 / ratio
                cmd = [
                    "ffmpeg", "-i", clip_path,
                    "-filter:v", f"setpts={speed}*PTS",
                    "-filter:a", f"atempo={1.0/speed}",
                    "-y", adjusted_path,
                ]
                try:
                    _sp.run(cmd, capture_output=True, check=True)
                    os.replace(adjusted_path, clip_path)
                except Exception as e:
                    print(f"[VideoEditor] Speed adjustment failed for clip {i}: {e}")
                    pass
            elif ratio > 1.5:
                # Narration much longer than clip → loop clip with crossfade
                # But limit the looping to prevent extremely long output
                if ratio > 2.0:
                    print(f"[VideoEditor] Ratio {ratio:.2f} is too high, limiting to 2.0 to prevent excessively long video")
                    ratio = 2.0
                
                loop_count = int(ratio) + 1
                cmd = [
                    "ffmpeg", "-stream_loop", str(loop_count),
                    "-i", clip_path,
                    "-filter_complex",
                    f"[0:v]setpts=PTS/{ratio}[v];[0:a]atempo={min(2.0, 1.0/ratio)}[a]",
                    "-map", "[v]", "-map", "[a]",
                    "-y", adjusted_path,
                ]
                try:
                    _sp.run(cmd, capture_output=True, check=True)
                    os.replace(adjusted_path, clip_path)
                except Exception as e:
                    print(f"[VideoEditor] Loop adjustment failed for clip {i}: {e}")
                    pass
            # ratio < 0.7: just let it play faster, still looks acceptable

    def _concat_clips(self, clip_paths: List[str], output_path: str):
        """Concatenate multiple video clips with smooth crossfade transitions."""
        if not clip_paths:
            raise ValueError("No clips to concatenate.")

        if len(clip_paths) == 1:
            # Single clip — just copy
            self._run_ffmpeg([
                "ffmpeg", "-i", clip_paths[0],
                "-c", "copy", "-y", output_path,
            ], "single clip copy")
            return

        # Use crossfade transitions between clips
        work_dir = os.path.dirname(output_path)
        transition_duration = 0.4  # seconds of crossfade

        # Step 1: prepare each clip with a small fade-out at the end
        prepped = []
        for i, cp in enumerate(clip_paths):
            prepped_path = os.path.join(work_dir, f"prepped_{i:04d}.mp4")
            clip_dur = self._get_clip_duration(cp)

            # Skip fade if clip has no duration (broken/empty clip)
            if clip_dur <= 0.5 or clip_dur > 86400:
                print(f"[VideoEditor] Clip {i} has invalid duration ({clip_dur}s), generating placeholder")
                try:
                    self._run_ffmpeg([
                        "ffmpeg", "-i", cp,
                        "-c", "copy",
                        "-y", prepped_path,
                    ], f"copy clip {i} (no fade)")
                except RuntimeError:
                    # Clip is completely broken — generate a blank 1s frame as placeholder
                    print(f"[VideoEditor] Clip {i} is corrupt, creating blank placeholder")
                    placeholder = os.path.join(work_dir, f"placeholder_{i:04d}.mp4")
                    self._run_ffmpeg([
                        "ffmpeg",
                        "-f", "lavfi", "-i", f"color=c=black:s=1920x1080:d=1:r=30",
                        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                        "-shortest",
                        "-y", placeholder,
                    ], f"create placeholder for clip {i}")
                    # Use placeholder as the prepped version
                    import shutil
                    shutil.copy(placeholder, prepped_path)
                prepped.append(prepped_path)
                continue

            # Add fade-out at the end of each clip (except last)
            if i < len(clip_paths) - 1 and clip_dur > transition_duration:
                fade_out = f",fade=t=out:st={clip_dur-transition_duration}:d={transition_duration}"
            else:
                fade_out = ""
            cmd = [
                "ffmpeg", "-i", cp,
                "-vf", f"fade=t=in:st=0:d=0.2{fade_out}",
                "-c:a", "copy",
                "-y", prepped_path,
            ]
            self._run_ffmpeg(cmd, f"prepare clip {i}")
            prepped.append(prepped_path)

        # Step 2: build filter_complex for crossfade
        filter_parts = []
        # Input labels: 0, 1, 2, ...
        for i in range(len(prepped)):
            filter_parts.append(f"[{i}:v][{i}:a]")

        # Build crossfade chain
        # [0] → [1] crossfade → [c0] → [2] crossfade → [c1] → ...
        stream_spec = ""
        for i in range(1, len(prepped)):
            prev = f"c{i-2}v" if i > 1 else "0v"
            cur = f"{i}v"
            out = f"c{i-1}v"
            prev_a = f"c{i-2}a" if i > 1 else "0a"

            if i == 1:
                # First transition: [0:v][0:a][1:v][1:a]
                stream_spec += (
                    f"[0:v][0:a][1:v][1:a]"
                    f"xfade=transition=fade:duration={transition_duration}:offset="
                    f"{self._get_clip_duration(prepped[0])-transition_duration}"
                    f"[c0v];"
                    f"[0:a][1:a]acrossfade=d={transition_duration}[c0a];"
                )
            else:
                # Nth transition
                offset = 0
                for j in range(i):
                    offset += self._get_clip_duration(prepped[j])
                offset -= transition_duration

                stream_spec += (
                    f"[c{i-2}v][c{i-2}a][{i}:v][{i}:a]"
                    f"xfade=transition=fade:duration={transition_duration}:offset={offset}"
                    f"[c{i-1}v];"
                    f"[c{i-1}a][{i}:a]acrossfade=d={transition_duration}[c{i-1}a];"
                )

        final_v_label = f"c{len(prepped)-2}v" if len(prepped) > 1 else "0:v"
        final_a_label = f"c{len(prepped)-2}a" if len(prepped) > 1 else "0:a"

        ffmpeg_cmd = [
            "ffmpeg",
        ]
        for pp in prepped:
            ffmpeg_cmd += ["-i", pp]
        ffmpeg_cmd += [
            "-filter_complex", stream_spec,
            "-map", f"[{final_v_label[0] if len(prepped)<=2 else 'c'+str(len(prepped)-2)+'v'}]",
            "-map", f"[{final_a_label[0] if len(prepped)<=2 else 'c'+str(len(prepped)-2)+'a'}]",
            "-c:v", "libx264", "-preset", "veryfast",
            "-c:a", "aac",
            "-y", output_path,
        ]

        # Simplify: if filter is too complex, fall back to simple concat
        import subprocess as _sp
        try:
            _sp.run(ffmpeg_cmd, capture_output=True, check=True, timeout=300)
        except Exception:
            print("[VideoEditor] Crossfade failed, falling back to simple concat")
            concat_file = os.path.join(work_dir, "concat_list.txt")
            with open(concat_file, "w") as f:
                for cp in clip_paths:
                    f.write(f"file '{cp}'\n")
            self._run_ffmpeg([
                "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_file,
                "-c", "copy", "-y", output_path,
            ], "simple concatenation")

    @staticmethod
    def _get_clip_duration(clip_path: str) -> float:
        """Get a clip's duration via ffprobe (quick). Returns 0 if undetermined."""
        import subprocess as _sp
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", clip_path]
        r = _sp.run(cmd, capture_output=True, text=True)
        dur_str = r.stdout.strip()
        if not dur_str or dur_str == "N/A":
            return 0.0
        try:
            return float(dur_str)
        except (ValueError, TypeError):
            return 0.0

    def _merge_audio_video(
        self, video_path: str, audio_path: str, output_path: str
    ):
        """Merge audio with video."""
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            "-y",
            output_path,
        ]
        self._run_ffmpeg(cmd, "audio-video merge")

    def _add_subtitles(
        self,
        video_path: str,
        script_segments: List[Dict],
        output_path: str,
        language: str,
    ):
        """Add hardcoded subtitles to the video."""
        # Create SRT subtitle file using estimated narration duration
        srt_path = os.path.join(os.path.dirname(output_path), "subtitles.srt")
        self._create_srt(script_segments, srt_path, language)

        # On Windows, ffmpeg's subtitles filter requires escaped backslashes
        # Convert backslashes to forward slashes, escape colons
        srt_path_ffmpeg = srt_path.replace("\\", "/").replace(":", "\\:")

        # Burn subtitles into video using the proper filter syntax
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"subtitles={srt_path_ffmpeg}",
            "-c:a", "copy",
            "-y",
            output_path,
        ]
        self._run_ffmpeg(cmd, "subtitle burn-in")

    def _create_srt(self, script_segments: List[Dict], output_path: str, language: str = "zh-CN"):
        """Create an SRT subtitle file from script segments using estimated narration duration."""
        current_time = 0.0
        with open(output_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(script_segments):
                narration = seg.get("narration", "")
                if not narration:
                    continue

                # Use the same duration estimation as clip extension
                duration = self._estimate_narration_duration(narration, language)

                start_h = int(current_time // 3600)
                start_m = int((current_time % 3600) // 60)
                start_s = int(current_time % 60)
                start_ms = int((current_time % 1) * 1000)

                end_time = current_time + duration
                end_h = int(end_time // 3600)
                end_m = int((end_time % 3600) // 60)
                end_s = int(end_time % 60)
                end_ms = int((end_time % 1) * 1000)

                f.write(f"{i+1}\n")
                f.write(f"{start_h:02d}:{start_m:02d}:{start_s:02d},{start_ms:03d} --> ")
                f.write(f"{end_h:02d}:{end_m:02d}:{end_s:02d},{end_ms:03d}\n")
                f.write(f"{narration}\n\n")

                current_time = end_time

    def _apply_mirror_reflection(
        self, clip_paths: List[str], work_dir: str, interval: int = 2
    ) -> List[str]:
        """
        Apply mirror reflection effect to alternating clips.
        Original clip on the left, mirrored reflection on the right (split-screen).

        Args:
            clip_paths: List of clip file paths
            work_dir: Working directory for temp files
            interval: Apply mirror to every Nth clip (2 = every other clip)

        Returns:
            Updated list of clip file paths
        """
        new_clip_paths = []
        for i, clip_path in enumerate(clip_paths):
            # Keep even-indexed (0-based) clips normal, mirror odd-indexed ones
            if (i + 1) % interval == 0:
                mirrored_path = os.path.join(work_dir, f"mirrored_{i:04d}.mp4")
                # Use ffmpeg to create split-screen: original left, mirrored right
                cmd = [
                    "ffmpeg",
                    "-i", clip_path,
                    "-vf",
                    "split=2[left][right];"
                    "[left]crop=iw/2:ih:0:0[left_crop];"
                    "[right]crop=iw/2:ih:iw/2:0,hflip[right_mirror];"
                    "[left_crop][right_mirror]hstack=inputs=2",
                    "-c:a", "copy",
                    "-y",
                    mirrored_path,
                ]
                self._run_ffmpeg(cmd, f"mirror reflection clip {i}")
                new_clip_paths.append(mirrored_path)
            else:
                new_clip_paths.append(clip_path)

        print(f"[VideoEditor] Applied mirror reflection to {len(new_clip_paths) - sum(1 for p in clip_paths if p != '')} clips")
        return new_clip_paths

    def _apply_blur_mask(self, input_path: str, output_path: str):
        """
        Apply a soft blur vignette mask overlay to the video.
        Creates a cinematic blur effect at the edges (top & bottom fade).

        The mask uses:
        - A blurred, darkened border at top and bottom
        - A subtle gaussian blur over the edges
        """
        # Complex filter: overlay a blurred vignette mask
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-vf",
            "split=2[original][blur];"
            "[blur]gblur=sigma=10:steps=4[blurred];"
            "[original][blurred]overlay=0:0:format=auto,"
            "drawbox=x=0:y=0:w=iw:h=ih/8:color=black@0.3:t=fill,"
            "drawbox=x=0:y=ih-ih/8:w=iw:h=ih/8:color=black@0.3:t=fill",
            "-c:a", "copy",
            "-y",
            output_path,
        ]
        self._run_ffmpeg(cmd, "blur mask overlay")

    def create_highlight_compilation(
        self,
        video_path: str,
        highlights: List[Dict],
        output_path: str,
        transition_duration: float = 0.5,
    ) -> str:
        """
        Create a compilation of highlight clips with crossfade transitions.
        
        Returns:
            Path to the compiled video.
        """
        if not highlights:
            raise ValueError("No highlights provided.")

        import random
        local_temp = Path(__file__).parent.parent / "temp" / "highlights"
        os.makedirs(local_temp, exist_ok=True)
        work_dir = tempfile.mkdtemp(prefix="highlights_", dir=str(local_temp))

        try:
            clip_paths = []
            for i, h in enumerate(highlights):
                clip_path = os.path.join(work_dir, f"highlight_{i:04d}.mp4")
                self._extract_clip(
                    video_path, h["start"], h["end"], clip_path
                )
                clip_paths.append(clip_path)

            # Create concat with transitions
            concat_file = os.path.join(work_dir, "concat_list.txt")
            with open(concat_file, "w") as f:
                for clip in clip_paths:
                    f.write(f"file '{clip}'\n")

            # First concatenate all clips
            temp_concat = os.path.join(work_dir, "temp_concat.mp4")
            cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                "-y",
                temp_concat,
            ]
            self._run_ffmpeg(cmd, "highlight concatenation")

            # Add fade transitions
            cmd = [
                "ffmpeg",
                "-i", temp_concat,
                "-vf", f"fade=t=in:st=0:d={transition_duration},fade=t=out:st={len(highlights)*15-transition_duration}:d={transition_duration}",
                "-c:a", "copy",
                "-y",
                output_path,
            ]
            self._run_ffmpeg(cmd, "fade transitions")

            return output_path

        finally:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)

    # ============================================================
    # FunClip-based video composition
    # ============================================================
    def compose_with_funclip(
        self,
        original_video: str,
        narration_audio: str,
        script_segments: List[Dict],
        subtitle_data: List[Dict],
        output_path: str,
        scene_matches: Optional[List[Dict]] = None,
        language: str = "zh-CN",
    ) -> str:
        """
        Compose final video using FunClip for scene-based clipping and splicing.

        FunClip uses ASR (FunASR) to align scenes and seamlessly concatenate clips.
        """
        try:
            from funasr import AutoModel
            import funclip
        except ImportError:
            raise ImportError("funasr/funclip not installed. Run: pip install funasr funclip")

        local_temp = Path(__file__).parent.parent / "temp" / "funclip"
        os.makedirs(local_temp, exist_ok=True)
        work_dir = tempfile.mkdtemp(prefix="funclip_", dir=str(local_temp))

        try:
            # Step 1: Load FunASR model for voice activity detection
            asr_model = AutoModel(
                model="paraformer-large",
                vad_model="fsmn-vad",
                punc_model="ct-punc",
                device="cuda",
            )

            # Step 2: Build clip list with timestamps
            if not scene_matches:
                scene_matches = self._match_scenes(script_segments, subtitle_data)

            # Step 3: Extract each scene clip
            clip_paths = []
            for i, match in enumerate(scene_matches):
                start, end = match["start"], match["end"]
                if start >= end:
                    continue
                clip_out = os.path.join(work_dir, f"scene_{i:04d}.mp4")
                self._extract_clip(original_video, start, end, clip_out)
                if self._get_clip_duration(clip_out) > 0.5:
                    clip_paths.append(clip_out)

            if not clip_paths:
                raise RuntimeError("No valid clips extracted for FunClip composition")

            # Step 4: Use funclip to concatenate with ASR-based alignment
            concat_path = os.path.join(work_dir, "funclip_concat.mp4")
            funclip.utils.concat_clips(
                clip_paths=clip_paths,
                output_path=concat_path,
                transition=0.3,
            )

            # Step 5: Merge narration audio
            if narration_audio and os.path.exists(narration_audio):
                temp_output = os.path.join(work_dir, "with_audio.mp4")
                self._merge_audio_video(concat_path, narration_audio, temp_output)
                concat_path = temp_output

            # Step 6: Add subtitles
            self._add_subtitles(concat_path, script_segments, output_path, language)

            print(f"[VideoEditor] FunClip composition complete: {output_path}")
            return output_path

        finally:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)

    # ============================================================
    # Narration Timing Check
    # ============================================================

    @staticmethod
    def check_narration_timing(
        script_segments: List[Dict],
        scene_data: Optional[List[Dict]] = None,
        language: str = "zh-CN",
        rate: int = -10,
        max_ratio: float = 1.3,
        min_ratio: float = 0.3,
    ) -> List[Dict]:
        """Check if each narration segment fits its scene duration."""
        results = []
        for i, seg in enumerate(script_segments):
            narration = seg.get("narration", "")
            estimated_dur = VideoEditor._estimate_narration_duration(narration, language, rate)

            if scene_data and i < len(scene_data):
                scene_dur = scene_data[i].get("duration", 10.0)
                scene_start = scene_data[i].get("start", 0)
                scene_end = scene_data[i].get("end", 10.0)
            else:
                scene_dur = 10.0
                scene_start = seg.get("start", 0)
                scene_end = seg.get("end", scene_start + 10.0)

            ratio = estimated_dur / max(scene_dur, 1.0)
            timing_ok = min_ratio <= ratio <= max_ratio

            if timing_ok:
                suggestion = "✅ 时长匹配"
            elif ratio > max_ratio:
                suggestion = f"⚠️ 解说过长 (ratio={ratio:.2f})：朗读需{estimated_dur:.1f}s，场景仅{scene_dur:.1f}s"
            else:
                suggestion = f"ℹ️ 解说偏短 (ratio={ratio:.2f})：朗读仅需{estimated_dur:.1f}s，场景有{scene_dur:.1f}s"

            result = dict(seg)
            result.update({
                "estimated_duration": round(estimated_dur, 2),
                "scene_duration": round(scene_dur, 2),
                "scene_start": scene_start,
                "scene_end": scene_end,
                "ratio": round(ratio, 2),
                "timing_ok": timing_ok,
                "suggestion": suggestion,
            })
            results.append(result)
        return results

    @staticmethod
    def estimate_narration_duration_public(narration: str, language: str = "zh-CN", rate: int = -10) -> float:
        return VideoEditor._estimate_narration_duration(narration, language, rate)

"""
Text-to-Speech module using Edge TTS.
Supports Chinese (zh-CN), Vietnamese (vi-VN), and English (en-US).
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

from config import TTS_VOICES, TTS_CONFIG, CHARS_PER_SECOND


class TTSEngine:
    """Text-to-Speech engine using Edge TTS."""

    def __init__(self):
        self.voice_map = TTS_VOICES  # display_name -> (voice_name, lang_code)
        self.config = TTS_CONFIG
        self.chars_per_sec = CHARS_PER_SECOND

    @staticmethod
    def estimate_duration(text: str, language: str = "zh-CN") -> float:
        """
        Estimate narration duration based on text length and speaking speed.
        Returns duration in seconds.
        """
        # Resolve language code from display name if needed
        lang_code = language
        if language in TTS_VOICES:
            lang_code = TTS_VOICES[language][1]

        cps = CHARS_PER_SECOND.get(lang_code, 3.5)
        # Count Chinese chars and English words
        chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        english_words = len([w for w in text.split() if w[0].isalpha()]) if any(c.isalpha() for c in text) else 0
        total_duration = (chinese / cps) + (english_words / (cps * 0.8))
        # Add small pause between sentences
        sentence_count = text.count('。') + text.count('！') + text.count('？') + text.count('.') + text.count('!') + text.count('?')
        total_duration += sentence_count * 0.3
        
        # Limit the duration to prevent extremely long estimates
        max_reasonable_duration = len(text) * 2  # Assume max 2 seconds per character as upper bound
        return max(2.0, min(total_duration, max_reasonable_duration))

    def _resolve_voice(self, language: str) -> str:
        """Resolve a display name or language code to an Edge TTS voice name."""
        # If it's a display name key in voice_map, use it directly
        if language in self.voice_map:
            return self.voice_map[language][0]
        # If it's a language code (zh-CN, vi-VN, en-US), pick first matching voice
        for disp, (voice, lc) in self.voice_map.items():
            if lc == language:
                return voice
        # Fallback to first voice
        first_key = next(iter(self.voice_map))
        return self.voice_map[first_key][0]

    def generate_speech(
        self,
        text: str,
        language: str = "zh-CN",
        output_path: Optional[str] = None,
        rate: str = None,
        volume: str = None,
        pitch: str = None,
    ) -> str:
        """
        Generate speech audio from text using Edge TTS.

        Args:
            text: Text to convert to speech
            language: Display name (e.g. "中文（女声 - Xiaoxiao）") or language code (zh-CN)
            output_path: Output audio file path (auto-generated if None)
            rate: Speech rate (e.g., "+0%", "-20%", "+50%")
            volume: Volume (e.g., "+0%")
            pitch: Pitch (e.g., "+0Hz")

        Returns:
            Path to the generated audio file
        """
        voice = self._resolve_voice(language)
        rate = rate or self.config["rate"]
        volume = volume or self.config["volume"]
        pitch = pitch or self.config["pitch"]

        if not output_path:
            output_path = tempfile.mktemp(suffix=".mp3")

        # Edge TTS uses asyncio, so we need to run it in an event loop
        try:
            asyncio.run(
                self._generate(text, voice, output_path, rate, volume, pitch)
            )
        except RuntimeError:
            # If there's already an event loop running
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                self._generate(text, voice, output_path, rate, volume, pitch)
            )
            loop.close()

        return output_path

    async def _generate(
        self, text: str, voice: str, output_path: str,
        rate: str, volume: str, pitch: str
    ):
        """Async method to generate speech using Edge TTS."""
        try:
            import edge_tts
        except ImportError:
            raise ImportError(
                "edge-tts not installed. Run: pip install edge-tts"
            )

        communicate = edge_tts.Communicate(
            text,
            voice,
            rate=rate,
            volume=volume,
            pitch=pitch,
        )
        await communicate.save(output_path)

    def generate_batch(
        self,
        segments: List[Dict],
        language: str = "zh-CN",
        output_dir: str = None,
    ) -> List[Dict]:
        """
        Generate speech for multiple script segments.

        Args:
            segments: List of dicts with 'narration' key
            language: Language code
            output_dir: Output directory for audio files

        Returns:
            List of dicts with narration, audio_path added
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        results = []
        for i, seg in enumerate(segments):
            if output_dir:
                audio_path = os.path.join(output_dir, f"narration_{i:04d}.mp3")
            else:
                audio_path = None

            narration = seg.get("narration", "")
            if not narration:
                continue

            print(f"[TTSEngine] Generating speech for segment {i+1}/{len(segments)}...")
            audio_file = self.generate_speech(
                text=narration,
                language=language,
                output_path=audio_path,
            )

            result = dict(seg)
            result["audio_path"] = audio_file
            results.append(result)

        return results

    def generate_full_script(
        self,
        full_text: str,
        language: str = "zh-CN",
        output_path: Optional[str] = None,
        rate: str = None,
        volume: str = None,
        pitch: str = None,
    ) -> str:
        """
        Generate speech for a complete script as a single audio file.

        Args:
            full_text: Complete script text
            language: Language code or display name
            output_path: Output audio file path
            rate: Speech rate (e.g., "+0%", "-10%")
            volume: Volume (e.g., "+0%")
            pitch: Pitch (e.g., "+0Hz")

        Returns:
            Path to the generated audio file
        """
        return self.generate_speech(full_text, language, output_path, rate=rate, volume=volume, pitch=pitch)

    @staticmethod
    def get_available_voices() -> Dict[str, str]:
        """Get available Edge TTS voices mapping."""
        return dict(TTS_VOICES)

    @staticmethod
    def list_all_voices() -> List[Dict]:
        """List all available Edge TTS voices."""
        try:
            import edge_tts
            voices = asyncio.run(edge_tts.list_voices())
            return [
                {"name": v["ShortName"], "locale": v["Locale"], "gender": v["Gender"]}
                for v in voices
            ]
        except Exception as e:
            print(f"[TTSEngine] Error listing voices: {e}")
            return []

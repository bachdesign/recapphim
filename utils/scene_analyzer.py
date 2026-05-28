"""
Scene analysis module.
Analyzes video content to find the best/exciting clips using:
- Frame difference analysis (histogram comparison)
- Audio analysis (volume spikes for action scenes)
- LLM-based analysis for understanding video content
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class SceneAnalyzer:
    """Analyze video scenes to find the most exciting/important clips."""

    def __init__(self, provider: str = "gemini", api_key: str = "", **kwargs):
        """
        Args:
            provider: LLM provider for content understanding ("gemini", "openai", "qwen", "deepseek")
            api_key: API key for the provider
        """
        self.provider = provider
        self.api_key = api_key
        self.config = kwargs
        self.mode = kwargs.get("mode", "donghua")

    def find_highlights(
        self,
        video_path: str,
        subtitles: List[Dict],
        num_clips: int = 10,
        clip_duration: float = 15.0,
    ) -> List[Dict]:
        """
        Find the most exciting clips from the video.
        
        Uses a combination of:
        1. Frame histogram analysis (detects scene changes)
        2. Subtitle emotional/action keyword analysis
        3. LLM-based content understanding
        
        Returns:
            List of dicts with: start, end, score, reason
        """
        # Method 1: Analyze subtitle content for action/emotional keywords
        subtitle_highlights = self._analyze_subtitle_highlights(subtitles, num_clips)

        # Method 2: Use LLM to understand video content and find highlights
        llm_highlights = self._analyze_with_llm(subtitles, video_path, num_clips)

        # Merge and rank results
        merged = self._merge_highlights(subtitle_highlights, llm_highlights, num_clips)

        # Extend to full clip duration
        for item in merged:
            mid_point = (item["start"] + item["end"]) / 2
            item["start"] = max(0, mid_point - clip_duration / 2)
            item["end"] = mid_point + clip_duration / 2

        return merged[:num_clips]

    def _analyze_subtitle_highlights(
        self, subtitles: List[Dict], num_clips: int
    ) -> List[Dict]:
        """Analyze subtitles to find action/emotional highlights."""
        # Mode-specific keywords
        if self.mode == "sports":
            action_keywords = [
                "score", "shot", "dunk", "block", "steal", "foul", "goal", "touchdown",
                "champion", "win", "loss", "defeat", "victory", "comeback", "upset",
                "crowd", "explosion", "amazing", "incredible", "clutch", "dominate",
                "break", "fast", "speed", "power", "fight", "battle", "intense",
                "emotional", "heart", "believe", "impossible", "history", "record",
                "final", "overtime", "buzzer", "ace", "serve", "match point",
                "highlight", "replay", "celebration", "anger", "shock", "reaction",
            ]
        else:
            action_keywords = [
                "战斗", "杀", "死", "爆炸", "危机", "危险", "突破", "升级",
                "功法", "法宝", "灵力", "爆发", "怒吼", "攻击", "防御",
                "决战", "大战", "激烈", "震撼", "恐怖", "强大", "逆天",
                "灭了", "斩杀", "轰", "秒杀", "偷袭", "围攻", "单挑",
                "受伤", "流血", "拼命", "绝招", "封印", "觉醒", "传承",
                "愤怒", "震惊", "没想到", "竟然", "突然", "原来",
            ]

        highlights = []
        for i, sub in enumerate(subtitles):
            text = sub.get("text", "")
            score = 0
            matched_keywords = []

            for kw in action_keywords:
                if kw in text:
                    score += 1
                    matched_keywords.append(kw)

            if score > 0:
                # Limit the text length to avoid overly long highlights
                limited_text = text[:100] + "..." if len(text) > 100 else text
                highlights.append({
                    "start": sub["start"],
                    "end": sub["end"],
                    "score": score,
                    "reason": f"包含激动人心关键词: {', '.join(matched_keywords)}",
                    "text": limited_text,
                })

        # Sort by score descending
        highlights.sort(key=lambda x: x["score"], reverse=True)

        # Merge nearby highlights
        merged = self._merge_nearby(highlights, gap=10.0)

        return merged[: num_clips * 2]

    def _analyze_with_llm(
        self, subtitles: List[Dict], video_path: str, num_clips: int
    ) -> List[Dict]:
        """Use an LLM to understand video content and find highlights."""
        if not self.api_key:
            print("[SceneAnalyzer] No API key provided for LLM analysis, using subtitle-only analysis.")
            return self._fallback_highlights(subtitles, num_clips)

        # Prepare subtitle summary for the LLM
        subtitle_summary = "\n".join(
            [
                f"[{s['start']:.1f}s-{s['end']:.1f}s] {s['text']}"
                for s in subtitles
            ]
        )

        if self.mode == "sports":
            prompt = f"""You are an elite ESPN sports video editor and analyst.

Analyze this sports match footage using the subtitle transcript with timestamps below.

{subtitle_summary}

TASKS:
1. Identify the MOST important, hype, and emotional moments.
2. Look for: clutch shots, blocks, dunks, steals, celebrations, crowd reactions, momentum shifts, player emotions, trash talk.
3. Rank scenes by hype level (1-10).
4. Keep the chronological order.

Output ONLY valid JSON:
{{
    "highlights": [
        {{
            "start": start_seconds,
            "end": end_seconds,
            "score": hype_score_1_to_10,
            "reason": "Why this moment matters (e.g. game-changing play, crowd eruption, superstar moment)"
        }}
    ]
}}

No markdown, no code fences, only JSON."""
        else:
            prompt = f"""你是一位专业的影视分析专家，特别擅长分析中国3D修仙、玄幻动画。

以下是某部动画片的字幕内容（含时间戳）：

{subtitle_summary}

请分析这个视频内容，找出其中最精彩、最有趣、最值得剪辑的10个片段，并且必须按照视频原本的逻辑顺序进行排列。

对每个片段，请提供：
1. 开始时间（秒）
2. 结束时间（秒）
3. 精彩程度评分（1-10分）
4. 为什么这个片段精彩
5. 必须保留人物名字、招式名称、法宝名称等不变。
请以JSON格式输出：
{{
    "highlights": [
        {{
            "start": 开始秒数,
            "end": 结束秒数,
            "score": 评分,
            "reason": "精彩原因"
        }}
    ]
}}

只输出JSON，不要包含```json标记。"""

        try:
            if self.provider == "gemini":
                return self._call_gemini(prompt, num_clips)
            elif self.provider == "openai":
                return self._call_openai(prompt, num_clips)
            elif self.provider == "deepseek":
                return self._call_deepseek(prompt, num_clips)
            elif self.provider == "qwen":
                return self._call_qwen(prompt, num_clips)
            else:
                return self._fallback_highlights(subtitles, num_clips)
        except Exception as e:
            print(f"[SceneAnalyzer] LLM analysis failed: {e}")
            return self._fallback_highlights(subtitles, num_clips)

    def _call_gemini(self, prompt: str, num_clips: int) -> List[Dict]:
        """Call Google Gemini API."""
        try:
            from google import genai
        except ImportError:
            raise ImportError("google-genai not installed. Run: pip install google-genai")

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.config.get("model", "gemini-2.5-flash"),
            contents=prompt,
        )
        return self._parse_llm_response(response.text, num_clips)

    def _call_openai(self, prompt: str, num_clips: int) -> List[Dict]:
        """Call OpenAI API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed.")

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.config.get("base_url", "https://api.openai.com/v1"),
        )
        response = client.chat.completions.create(
            model=self.config.get("model", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return self._parse_llm_response(response.choices[0].message.content, num_clips)

    def _call_deepseek(self, prompt: str, num_clips: int) -> List[Dict]:
        """Call DeepSeek API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed.")

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.config.get("base_url", "https://api.deepseek.com/v1"),
        )
        response = client.chat.completions.create(
            model=self.config.get("model", "deepseek-chat"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return self._parse_llm_response(response.choices[0].message.content, num_clips)

    def _call_qwen(self, prompt: str, num_clips: int) -> List[Dict]:
        """Call Qwen (Alibaba Cloud) API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed.")

        base_url = self.config.get("base_url", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        if "/api/v1" in base_url and "/compatible-mode" not in base_url:
            base_url = base_url.replace("/api/v1", "/compatible-mode/v1")
        client = OpenAI(api_key=self.api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=self.config.get("model", "qwen-max"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return self._parse_llm_response(response.choices[0].message.content, num_clips)

    def _parse_llm_response(self, text: str, num_clips: int) -> List[Dict]:
        """Parse LLM JSON response."""
        # Clean up markdown code blocks if present
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        if text.startswith("json"):
            text = text[4:].strip()

        try:
            data = json.loads(text)
            highlights = data.get("highlights", [])
            for h in highlights:
                h["score"] = float(h.get("score", 5))
            return highlights[:num_clips]
        except json.JSONDecodeError:
            print(f"[SceneAnalyzer] Failed to parse LLM response: {text[:200]}")
            return []

    def analyze_highlights_with_vision(
        self,
        scenes: List[Dict],
        subtitles: List[Dict],
        num_clips: int = 10,
    ) -> List[Dict]:
        """
        Analyze scene frames using Qwen/DashScope vision to select the best highlights.
        
        Each scene's representative frames are sent to the vision model for analysis.
        Returns a list of highlights with scores, reasons, and commentary suggestions.
        """
        if not self.api_key:
            print("[SceneAnalyzer] No API key, falling back to text-only highlight analysis")
            return self.find_highlights("", subtitles, num_clips)

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed")

        # Use compatible-mode base URL
        base_url = self.config.get("base_url", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        # Ensure compatible-mode URL
        if "/api/v1" in base_url and "/compatible-mode" not in base_url:
            base_url = base_url.replace("/api/v1", "/compatible-mode/v1")

        client = OpenAI(api_key=self.api_key, base_url=base_url)
        model = self.config.get("model", "qwen3.6-plus")

        highlights = []
        import base64 as _b64

        for scene in scenes[:num_clips * 2]:
            scene_id = scene.get("scene_id", 0)
            frames = scene.get("frames", [])
            subtitle_text = scene.get("subtitle_text", "(无字幕)")

            if not frames:
                continue

            # Build message with scene frames as images
            content_parts = []
            for fp in frames[:3]:  # Max 3 frames per scene
                if os.path.exists(fp):
                    with open(fp, "rb") as f:
                        b64 = _b64.b64encode(f.read()).decode("utf-8")
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })

            if self.mode == "sports":
                prompt_text = (
                    "You are an elite ESPN sports analyst. Analyze these scene frames.\n"
                    f"Scene {scene_id} subtitle: {subtitle_text}\n\n"
                    "Rate this scene's highlight potential (1-10) based on: action intensity, "
                    "emotional impact, crowd energy, player skill moments.\n\n"
                    "Output JSON only:\n"
                    '{"score": int, "reason": "why this is/isn\'t a highlight", '
                    '"commentary_hook": "suggested narration hook"}'
                )
            else:
                prompt_text = (
                    "你是一位拥有10年经验的中国3D动画电影解说编剧，擅长仙侠、玄幻题材。\n\n"
                    f"请分析以下场景 {scene_id} 的画面内容和字幕：\n"
                    f"【字幕】{subtitle_text}\n\n"
                    "1. 描述视频中人物所进行的一系列动作，保留人物名字、法宝名称、地名等专有名词不变。\n"
                    "2. 判断这个场景是否适合作为解说视频的精彩片段（评分1-10）。\n"
                    "3. 给出入选理由。\n"
                    "4. 给出一个吸引人的解说切入点（一句话钩子）。\n\n"
                    "只输出JSON格式：\n"
                    '{"score": 评分, "visual_actions": ["动作1", "动作2"], '
                    '"reason": "入选理由", "commentary_hook": "解说切入点"}'
                )

            content_parts.append({"type": "text", "text": prompt_text})

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content_parts}],
                    temperature=0.3,
                    max_tokens=1024,
                )
                text = response.choices[0].message.content.strip()
                # Parse JSON
                if text.startswith("```"):
                    text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                    text = text.rsplit("```", 1)[0] if "```" in text else text
                text = text.strip()
                if text.startswith("json"):
                    text = text[4:].strip()

                import json as _json
                data = _json.loads(text)
                highlights.append({
                    "start": scene.get("start", 0),
                    "end": scene.get("end", 0),
                    "score": float(data.get("score", 5)),
                    "reason": data.get("reason", "视觉分析结果"),
                    "visual_actions": data.get("visual_actions", []),
                    "commentary_hook": data.get("commentary_hook", ""),
                    "scene_id": scene_id,
                })
                print(f"[SceneAnalyzer] Scene {scene_id}: scored {data.get('score', 'N/A')}")
            except Exception as e:
                print(f"[SceneAnalyzer] Scene {scene_id} vision analysis failed: {e}")
                # Fallback: use keyword-based score
                score = self._score_subtitle_keywords(subtitle_text)
                highlights.append({
                    "start": scene.get("start", 0),
                    "end": scene.get("end", 0),
                    "score": score,
                    "reason": f"字幕关键词评分: {score}",
                    "visual_actions": [],
                    "commentary_hook": "",
                    "scene_id": scene_id,
                })

        # Sort by score descending, return top clips
        highlights.sort(key=lambda x: x.get("score", 0), reverse=True)
        return highlights[:num_clips]

    def _score_subtitle_keywords(self, text: str) -> float:
        """Score a subtitle text based on action/emotional keywords."""
        keywords = [
            "战斗", "杀", "死", "爆炸", "危机", "突破", "升级",
            "功法", "法宝", "灵力", "爆发", "怒吼", "攻击", "防御",
            "决战", "大战", "激烈", "震撼", "恐怖", "强大", "逆天",
            "灭了", "斩杀", "轰", "秒杀", "偷袭", "围攻", "单挑",
            "受伤", "流血", "拼命", "绝招", "封印", "觉醒", "传承",
            "愤怒", "震惊", "没想到", "竟然", "突然", "原来",
        ]
        score = 0
        for kw in keywords:
            if kw in text:
                score += 1
        return min(score * 1.5, 10.0)

    def _fallback_highlights(self, subtitles: List[Dict], num_clips: int) -> List[Dict]:
        """Fallback method if LLM is unavailable."""
        if not subtitles:
            return []

        # Evenly distribute clips throughout the video
        total_duration = subtitles[-1]["end"] if subtitles else 600
        interval = total_duration / num_clips

        highlights = []
        for i in range(num_clips):
            t = i * interval
            highlights.append({
                "start": t,
                "end": min(t + 15, total_duration),
                "score": 5.0,
                "reason": "均匀采样片段",
            })

        return highlights

    @staticmethod
    def _merge_nearby(highlights: List[Dict], gap: float = 10.0) -> List[Dict]:
        """Merge nearby highlights that are within gap seconds of each other."""
        if not highlights:
            return []

        merged = [highlights[0]]
        for h in highlights[1:]:
            if h["start"] - merged[-1]["end"] <= gap:
                # Merge
                merged[-1]["end"] = max(merged[-1]["end"], h["end"])
                merged[-1]["score"] = max(merged[-1]["score"], h["score"])
                merged[-1]["reason"] += "; " + h["reason"]
                if "text" in h and "text" in merged[-1]:
                    merged[-1]["text"] += " " + h["text"]
            else:
                merged.append(h)

        return merged

    @staticmethod
    def _merge_highlights(
        list1: List[Dict], list2: List[Dict], max_items: int
    ) -> List[Dict]:
        """Merge two lists of highlights, removing duplicates and sorting by score."""
        seen = set()
        merged = []

        for item in list1 + list2:
            key = (round(item.get("start", 0), 1), round(item.get("end", 0), 1))
            if key not in seen:
                seen.add(key)
                merged.append(item)

        merged.sort(key=lambda x: x.get("score", 0), reverse=True)
        return merged[:max_items]

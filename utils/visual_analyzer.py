"""
Visual Understanding Module.
Analyzes scene images via LLM vision APIs to understand visual semantics.
Supports Gemini (vision), OpenAI (GPT-4o vision), Qwen (qwen-vl-max), and DeepSeek (text-only).
"""

import base64
import json
import os
import sys
import traceback
from pathlib import Path
from typing import List, Dict, Optional

from utils.cancel_utils import raise_if_cancelled, TaskCancelledError


class VisualAnalyzer:
    """
    Analyze scene images to extract visual semantics.
    For each scene, takes the subtitle + representative frames
    and returns a description of what's happening visually.
    """

    def __init__(self, provider: str = "gemini", api_key: str = "", **kwargs):
        """
        Args:
            provider: "gemini", "openai", "qwen", or "deepseek"
            api_key: API key for the provider
        """
        self.provider = provider
        self.api_key = api_key
        self.config = kwargs
        self.mode = kwargs.get("mode", "donghua")

    def analyze_scene(
        self, scene: Dict, temperature: float = 0.3
    ) -> Dict:
        """
        Analyze a single scene: its subtitle + frames → visual understanding.

        Args:
            scene: Scene dict with scene_id, start, end, subtitle_text, frames (image paths)
            temperature: LLM temperature

        Returns:
            Scene dict enriched with 'visual' and 'emotion' fields
        """
        frames = scene.get("frames", [])
        subtitle_text = scene.get("subtitle_text", "")
        scene_id = scene.get("scene_id", 0)

        if not frames:
            print(f"[VisualAnalyzer] Scene {scene_id}: No frames to analyze")
            scene["visual"] = "（无画面信息）"
            scene["emotion"] = "中性"
            scene["visual_actions"] = []
            return scene

        if self.provider == "gemini":
            return self._analyze_with_gemini(scene, temperature)
        elif self.provider == "openai":
            return self._analyze_with_openai(scene, temperature)
        elif self.provider == "qwen":
            return self._analyze_with_qwen(scene, temperature)
        elif self.provider == "deepseek":
            return self._analyze_with_deepseek(scene, temperature)
        else:
            print(f"[VisualAnalyzer] Unsupported provider: {self.provider}, falling back")
            return self._analyze_with_gemini(scene, temperature)

    def _analyze_with_gemini(self, scene: Dict, temperature: float) -> Dict:
        """Send scene frames to Gemini Vision for visual understanding."""
        try:
            from google import genai
        except ImportError:
            raise ImportError("google-genai not installed. Run: pip install google-genai")

        if not self.api_key:
            print("[VisualAnalyzer] No API key for Gemini, skipping visual analysis")
            scene["visual"] = "（无API密钥，跳过视觉分析）"
            scene["emotion"] = "中性"
            scene["visual_actions"] = []
            return scene

        client = genai.Client(api_key=self.api_key)
        model = self.config.get("model", "gemini-2.5-flash")

        frames = scene.get("frames", [])
        scene_id = scene.get("scene_id", 0)
        subtitle_text = scene.get("subtitle_text", "(无字幕)")

        # Build prompt based on mode
        if self.mode == "sports":
            prompt = self._sports_prompt(scene_id, subtitle_text)
        else:
            prompt = self._donghua_prompt(scene_id, subtitle_text)

        # Upload images and build contents
        contents = []
        uploaded_files = []

        try:
            for fp in frames[:5]:  # Max 5 frames
                if os.path.exists(fp):
                    uploaded = client.files.upload(file=fp)
                    uploaded_files.append(uploaded)
                    contents.append(uploaded)

            contents.append(prompt)

            response = client.models.generate_content(
                model=model,
                contents=contents,
                config={"temperature": temperature},
            )

            result = self._parse_visual_response(response.text, scene)
            print(f"[VisualAnalyzer] Scene {scene_id}: Visual analysis complete")
            return result

        except Exception as e:
            print(f"[VisualAnalyzer] Gemini analysis failed for scene {scene_id}: {e}")
            # Try without images (text-only fallback)
            return self._text_fallback(scene, subtitle_text)
        finally:
            # Clean up uploaded files
            for f in uploaded_files:
                try:
                    client.files.delete(f.name)
                except Exception:
                    pass

    def _analyze_with_openai(self, scene: Dict, temperature: float) -> Dict:
        """Send scene frames to OpenAI GPT-4o Vision."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed")

        if not self.api_key:
            print("[VisualAnalyzer] No API key for OpenAI, skipping visual analysis")
            scene["visual"] = "（无API密钥，跳过视觉分析）"
            scene["emotion"] = "中性"
            scene["visual_actions"] = []
            return scene

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.config.get("base_url", "https://api.openai.com/v1"),
        )

        frames = scene.get("frames", [])
        scene_id = scene.get("scene_id", 0)
        subtitle_text = scene.get("subtitle_text", "(无字幕)")

        if self.mode == "sports":
            prompt_text = self._sports_prompt(scene_id, subtitle_text)
        else:
            prompt_text = self._donghua_prompt(scene_id, subtitle_text)

        # Build message content with images
        content_parts = [{"type": "text", "text": prompt_text}]
        for fp in frames[:5]:
            if os.path.exists(fp):
                with open(fp, "rb") as img_file:
                    b64 = base64.b64encode(img_file.read()).decode("utf-8")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })

        try:
            response = client.chat.completions.create(
                model=self.config.get("model", "gpt-4o"),
                messages=[{"role": "user", "content": content_parts}],
                temperature=temperature,
                max_tokens=1024,
            )

            result = self._parse_visual_response(
                response.choices[0].message.content, scene
            )
            print(f"[VisualAnalyzer] Scene {scene_id}: Visual analysis complete")
            return result

        except Exception as e:
            print(f"[VisualAnalyzer] OpenAI analysis failed for scene {scene_id}: {e}")
            return self._text_fallback(scene, subtitle_text)

    def _analyze_with_qwen(self, scene: Dict, temperature: float) -> Dict:
        """Analyze scene using Qwen Vision via OpenAI-compatible API (proven to work with DashScope)."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")

        # 确保传入的 key 是干净的
        api_key = self.api_key.strip() if self.api_key else ""
        if not api_key:
            print("[VisualAnalyzer] No API key for Qwen, skipping visual analysis")
            scene["visual"] = "（无API密钥，跳过视觉分析）"
            scene["emotion"] = "中性"
            scene["visual_actions"] = []
            return scene

        # 同时设置 dashscope 全局 key + 环境变量（兼容不同 SDK 版本）
        try:
            import dashscope
            dashscope.api_key = api_key
        except ImportError:
            pass
        os.environ["DASHSCOPE_API_KEY"] = api_key

        # Use compatible-mode base URL (OpenAI-compatible API)
        qwen_base_url = self.config.get("base_url", "")
        if not qwen_base_url:
            qwen_base_url = "https://dashscope-intl.aliyuncs.com/api/v1"
        # Ensure we use compatible-mode endpoint for OpenAI client
        if "/api/v1" in qwen_base_url and "/compatible-mode" not in qwen_base_url:
            qwen_base_url = qwen_base_url.replace("/api/v1", "/compatible-mode/v1")

        client = OpenAI(
            api_key=api_key,
            base_url=qwen_base_url,
        )

        frames = scene.get("frames", [])
        scene_id = scene.get("scene_id", 0)
        subtitle_text = scene.get("subtitle_text", "(无字幕)")

        if self.mode == "sports":
            prompt_text = self._sports_prompt(scene_id, subtitle_text)
        else:
            prompt_text = self._donghua_prompt(scene_id, subtitle_text)

        # Build message content with images as base64 (OpenAI-compatible format)
        content_parts = [{"type": "text", "text": prompt_text}]
        for fp in frames[:5]:
            if os.path.exists(fp):
                try:
                    with open(fp, "rb") as img_file:
                        b64 = base64.b64encode(img_file.read()).decode("utf-8")
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })
                except Exception as img_err:
                    print(f"[VisualAnalyzer] Failed to read image {fp}: {img_err}")

        try:
            model = self.config.get("model", "qwen-vl-max")
            masked_key = self.api_key[:6] + "****" + self.api_key[-4:] if len(self.api_key) > 12 else "****"
            print(f"[VisualAnalyzer] Calling Qwen {model} at {qwen_base_url} with key {masked_key}")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content_parts}],
                temperature=temperature,
                max_tokens=2048,
            )

            text = response.choices[0].message.content
            if text:
                result = self._parse_visual_response(text, scene)
                print(f"[VisualAnalyzer] Scene {scene_id}: Qwen analysis complete")
                return result
            else:
                print(f"[VisualAnalyzer] Qwen returned empty for scene {scene_id}")
                return self._text_fallback(scene, subtitle_text)

        except Exception as e:
            print(f"[VisualAnalyzer] Qwen vision failed for scene {scene_id}: {e}")
            import traceback
            traceback.print_exc()
            return self._text_fallback(scene, subtitle_text)

    def _qwen_text_fallback(self, scene: Dict, subtitle_text: str, temperature: float) -> Dict:
        """Fallback: call Qwen text-only via OpenAI-compatible API."""
        try:
            from openai import OpenAI
        except ImportError:
            return self._text_fallback(scene, subtitle_text)

        qwen_base_url = self.config.get("base_url", "")
        if not qwen_base_url:
            qwen_base_url = "https://dashscope-intl.aliyuncs.com/api/v1"
        # Ensure we use compatible-mode endpoint for OpenAI client
        if "/api/v1" in qwen_base_url and "/compatible-mode" not in qwen_base_url:
            qwen_base_url = qwen_base_url.replace("/api/v1", "/compatible-mode/v1")

        client = OpenAI(api_key=self.api_key, base_url=qwen_base_url)

        scene_id = scene.get("scene_id", 0)
        frame_names = [os.path.basename(f) for f in scene.get("frames", [])[:5] if os.path.exists(f)]
        frame_hint = f"本场景截图: {', '.join(frame_names)}" if frame_names else "（无截图）"

        prompt_text = (
            f"你是一位专业的中国3D修仙/玄幻动画视觉分析专家。\n\n"
            f"【场景 {scene_id}】\n"
            f"{frame_hint}\n"
            f"【字幕参考】\n{subtitle_text}\n\n"
            f"请根据字幕内容和场景时间推测画面中的视觉内容。\n"
            f"只输出JSON，格式如下：\n"
            f"{{\"visual_actions\": [\"动作1\", \"动作2\"], \"visual\": \"综合描述\", \"emotion\": \"情绪\"}}"
        )

        try:
            response = client.chat.completions.create(
                model=self.config.get("model", "qwen-max"),
                messages=[{"role": "user", "content": prompt_text}],
                temperature=temperature,
                max_tokens=1024,
            )
            text = response.choices[0].message.content
            if text:
                return self._parse_visual_response(text, scene)
        except Exception as e:
            print(f"[VisualAnalyzer] Qwen text fallback also failed: {e}")

        return self._text_fallback(scene, subtitle_text)

    def _analyze_with_deepseek(self, scene: Dict, temperature: float) -> Dict:
        """Analyze scene using DeepSeek (text-only — no vision support).
        Uses subtitle text to infer visual content."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed")

        if not self.api_key:
            print("[VisualAnalyzer] No API key for DeepSeek, skipping visual analysis")
            scene["visual"] = "（无API密钥，跳过视觉分析）"
            scene["emotion"] = "中性"
            scene["visual_actions"] = []
            return scene

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.config.get("base_url", "https://api.deepseek.com/v1"),
        )

        frames = scene.get("frames", [])
        scene_id = scene.get("scene_id", 0)
        subtitle_text = scene.get("subtitle_text", "(无字幕)")
        # Include frame filenames as weak visual hints
        frame_names = [os.path.basename(f) for f in frames[:5] if os.path.exists(f)]
        frame_hint = "本场景截图: " + ", ".join(frame_names) if frame_names else "（无截图）"

        if self.mode == "sports":
            prompt_text = (
                f"You are an elite sports visual analyst.\n\n"
                f"【Scene {scene_id}】\n"
                f"{frame_hint}\n"
                f"【Subtitle Reference】\n{subtitle_text}\n\n"
                f"Based on the subtitle text and scene timing, describe what is likely happening visually.\n"
                f"Output JSON only:\n"
                f"{{\"visual_actions\": [...], \"visual\": \"...\", \"emotion\": \"...\"}}"
            )
        else:
            prompt_text = (
                f"你是一位专业的中国3D修仙/玄幻动画视觉分析专家。\n\n"
                f"【场景 {scene_id}】\n"
                f"{frame_hint}\n"
                f"【字幕参考】\n{subtitle_text}\n\n"
                f"由于无法直接查看画面，请根据字幕内容和场景时间推测画面中的视觉内容。\n"
                f"只输出JSON，格式如下：\n"
                f"{{\"visual_actions\": [\"动作1\", \"动作2\"], \"visual\": \"综合描述\", \"emotion\": \"情绪\"}}"
            )

        try:
            model = self.config.get("model", "deepseek-chat")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=temperature,
                max_tokens=1024,
            )
            result = self._parse_visual_response(
                response.choices[0].message.content, scene
            )
            print(f"[VisualAnalyzer] Scene {scene_id}: DeepSeek analysis complete")
            return result
        except Exception as e:
            print(f"[VisualAnalyzer] DeepSeek analysis failed for scene {scene_id}: {e}")
            return self._text_fallback(scene, subtitle_text)

    def _donghua_prompt(self, scene_id: int, subtitle_text: str) -> str:
        """Build prompt for donghua/anime visual analysis."""
        return f"""你是一位专业的中国3D修仙/玄幻动画视觉分析专家。
请仔细观察这组截图画面，分析当前场景的视觉内容。

【场景 {scene_id}】
【字幕参考】
{subtitle_text}

请分析以下内容：

1. **画面中出现了什么？**（人物、场景、动作、特效等）
2. **角色在做什么？**（战斗、对话、修炼、逃亡等）
3. **画面色调和氛围**（明亮/阴暗/红色/蓝色等）
4. **情绪/气氛**（紧张、悲伤、激烈、平静、高潮等）

【输出格式 - 只输出JSON，不要用``标记】
{{
    "visual_actions": [
        "王林挥剑",
        "对手被击飞",
        "画面切换到红色天空"
    ],
    "visual": "综合描述：王林爆发杀气，剑阵展开，对面敌人被击飞，天空变为血红色，气氛紧张到极点。",
    "emotion": "高潮"
}}

注意：
- visual_actions 是3-5个简洁的动作/视觉元素列表
- visual 是1-2句综合描述
- emotion 是当前场景的情绪基调
- 必须保留人物名字、招式名称、法宝名称等不变。

只输出JSON。"""

    def _sports_prompt(self, scene_id: int, subtitle_text: str) -> str:
        """Build prompt for sports visual analysis."""
        return f"""You are an elite ESPN sports visual analyst.
Analyze these screenshots from a sports match.

【Scene {scene_id}】
【Subtitle Reference】
{subtitle_text}

Analyze:
1. What is happening visually? (players, plays, crowd, reactions)
2. Key actions (shots, blocks, dunks, celebrations)
3. Visual tone and energy level
4. Emotional intensity

【Output - JSON only, no markdown】
{{
    "visual_actions": [
        "LeBron drives to the basket",
        "Crowd on their feet",
        "Defender attempts block"
    ],
    "visual": "LeBron James drives hard to the rim with the crowd erupting. The defender leaps for a block attempt. High intensity playoff atmosphere.",
    "emotion": "high_intensity"
}}

JSON only."""

    def _parse_visual_response(self, text: str, scene: Dict) -> Dict:
        """Parse the LLM vision response."""
        text = text.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0] if "```" in text else text
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()

        try:
            data = json.loads(text)
            scene["visual_actions"] = data.get("visual_actions", [])
            scene["visual"] = data.get("visual", "（无描述）")
            scene["emotion"] = data.get("emotion", "中性")
        except json.JSONDecodeError:
            # Fallback: use raw text
            scene["visual_actions"] = [text[:100]]
            scene["visual"] = text[:200]
            scene["emotion"] = "中性"

        # Apply narration length control to ensure each sentence is 12-18 characters
        scene["visual"] = self._control_narration_length(scene["visual"])
        scene["visual_actions"] = [self._control_narration_length(action) for action in scene["visual_actions"]]

        return scene

    def _control_narration_length(self, text: str) -> str:
        """
        Control narration length to ensure each sentence is 12-18 characters.
        """
        import re

        if not text or len(text) < 12:
            return text

        # Split by Chinese punctuation marks
        sentences = re.split(r'([。！？!?])', text)

        # Reconstruct sentences with punctuation
        result = []
        i = 0
        while i < len(sentences) - 1:
            sentence = sentences[i] + sentences[i+1]  # Add the punctuation back
            i += 2

            # Clean up whitespace
            sentence = sentence.strip()
            if not sentence:
                continue

            # If sentence is too long, break it down further
            if len(sentence) > 18:
                # Split by clauses or phrases while respecting character boundaries
                temp_sentences = self._break_long_sentence(sentence, 18, 12)
                result.extend(temp_sentences)
            elif len(sentence) < 12 and result:
                # If sentence is too short, append to previous sentence if possible
                result[-1] += sentence
            else:
                result.append(sentence)

        # Join the sentences back together
        return "".join(result)

    def _break_long_sentence(self, sentence: str, max_length: int, min_length: int) -> List[str]:
        """
        Break a long sentence into smaller parts that fit length requirements.
        """
        import re

        # Try to split by common Chinese conjunctions/particles that indicate clause breaks
        clause_breakers = ['，', '、', '；', '而', '且', '但', '却', '虽', '然', '因', '所', '以']

        # First, try to split by punctuation
        parts = re.split(r'([，、；])', sentence)

        result = []
        current_part = ""

        for i in range(0, len(parts), 2):  # Process text and potential separator
            text_part = parts[i] if i < len(parts) else ""
            separator = parts[i+1] if i+1 < len(parts) else ""

            test_part = current_part + text_part + separator

            if len(test_part) <= max_length:
                current_part = test_part
            else:
                # If adding this part would exceed max length
                if len(current_part) >= min_length or current_part == "":
                    # Add current part if it meets minimum length or is empty
                    if current_part:
                        result.append(current_part)
                    # Start new part with current text
                    current_part = text_part + separator
                    # If this single part is still too long, force split by max_length
                    if len(current_part) > max_length:
                        forced_parts = [current_part[j:j+max_length] for j in range(0, len(current_part), max_length)]
                        # Add all but the last part to results
                        result.extend(forced_parts[:-1])
                        # Keep the last part as current
                        current_part = forced_parts[-1]
                else:
                    # Force split the current part to stay under max_length
                    if len(current_part) > max_length:
                        # Split the current part
                        result.append(current_part[:max_length])
                        current_part = current_part[max_length:] + text_part + separator
                    else:
                        # Add to result and start fresh
                        result.append(current_part)
                        current_part = text_part + separator

        # Add remaining part if it exists
        if current_part:
            if len(current_part) > max_length:
                # Further split if still too long
                forced_parts = [current_part[j:j+max_length] for j in range(0, len(current_part), max_length)]
                result.extend(forced_parts)
            else:
                result.append(current_part)

        return result

    def _text_fallback(self, scene: Dict, subtitle_text: str) -> Dict:
        """Fallback when vision API fails — use subtitle to infer."""
        scene["visual_actions"] = ["（视觉分析不可用，基于字幕推断）"]
        scene["visual"] = f"基于字幕推断：{subtitle_text[:100]}"
        scene["emotion"] = "中性"
        return scene

    def analyze_all_scenes(
        self, scenes: List[Dict], temperature: float = 0.3
    ) -> List[Dict]:
        """
        Analyze all scenes sequentially.
        Each scene's visual understanding is built independently.
        Checks cancel flag between scenes.
        """
        results = []
        for i, scene in enumerate(scenes):
            raise_if_cancelled()  # Allow user to cancel between scenes
            print(f"[VisualAnalyzer] Analyzing scene {scene['scene_id']} ({i+1}/{len(scenes)})...")
            result = self.analyze_scene(scene, temperature)
            results.append(result)

        return results

    @staticmethod
    def merge_scene_data(scenes: List[Dict]) -> List[Dict]:
        """
        Final merge: subtitle + visual + timeline → unified scene format.

        Input: scenes with subtitles, frames, visual_actions, visual, emotion
        Output: Unified format for narration generation
        """
        merged = []
        for scene in scenes:
            merged.append({
                "scene": scene.get("scene_id", 0),
                "start": scene.get("start", 0),
                "end": scene.get("end", 0),
                "duration": scene.get("duration", 0),
                "subtitle": scene.get("subtitle_text", ""),
                "visual": scene.get("visual", ""),
                "visual_actions": scene.get("visual_actions", []),
                "emotion": scene.get("emotion", "中性"),
                "frames": scene.get("frames", []),
            })
        return merged

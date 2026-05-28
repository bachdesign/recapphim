import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
import requests
import time

class ScriptGenerator:
    def __init__(self, provider: str = "gemini", api_key: str = None, model: str = None, base_url: str = "", mode: str = "donghua"):
        self.provider = provider
        self.api_key = api_key
        self.model = model or self._get_default_model(provider)
        self.base_url = base_url
        self.mode = mode
        self.temp_base_path = r"K:\New folder (22)\temp"
        
    def _get_default_model(self, provider: str) -> str:
        """Get default model based on provider"""
        defaults = {
            "gemini": "gemini-2.5-flash",
            "openai": "gpt-4o-mini",
            "qwen": "qwen-max",
            "deepseek": "deepseek-chat",
            "faster_whisper": "medium"
        }
        return defaults.get(provider, "gemini-2.5-flash")
    
    def find_project_dirs(self) -> List[Path]:
        """查找所有项目目录"""
        project_dirs = []
        temp_path = Path(self.temp_base_path)
        
        if not temp_path.exists():
            raise FileNotFoundError(f"临时目录不存在: {self.temp_base_path}")
        
        # 查找所有以 project_ 开头的目录
        for item in temp_path.iterdir():
            if item.is_dir() and item.name.startswith("project_"):
                project_dirs.append(item)
        
        return project_dirs
    
    def get_project_data(self, project_dir: Path) -> Dict:
        """获取单个项目的数据"""
        data = {
            'project_id': project_dir.name,
            'subtitles': [],
            'images': [],
            'image_descriptions': []
        }
        
        # 获取字幕文件
        subtitle_path = project_dir / "subtitles"
        if subtitle_path.exists():
            subtitle_files = list(subtitle_path.glob("*.json")) + list(subtitle_path.glob("*.srt"))
            if subtitle_files:
                # 读取字幕内容
                for sub_file in subtitle_files:
                    try:
                        if sub_file.suffix.lower() == '.json':
                            with open(sub_file, 'r', encoding='utf-8') as f:
                                data['subtitles'] = json.load(f)
                        elif sub_file.suffix.lower() == '.srt':
                            data['subtitles'] = self._parse_srt(sub_file)
                    except Exception as e:
                        print(f"读取字幕文件失败 {sub_file}: {e}")
        
        # 获取图片文件
        frames_path = project_dir / "frames"
        if frames_path.exists():
            image_files = [f for f in frames_path.iterdir() 
                          if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            data['images'] = sorted(image_files, key=lambda x: self._extract_frame_number(x.name))
        
        return data
    
    def _parse_srt(self, srt_file: Path) -> List[Dict]:
        """解析SRT字幕文件"""
        subtitles = []
        with open(srt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # SRT格式解析
        blocks = content.strip().split('\n\n')
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                try:
                    start_time, end_time = self._parse_time_range(lines[1])
                    text = ' '.join(lines[2:])
                    
                    subtitles.append({
                        'start': start_time,
                        'end': end_time,
                        'text': text.strip()
                    })
                except:
                    continue
        
        return subtitles
    
    def _parse_time_range(self, time_str: str) -> tuple:
        """解析时间范围字符串"""
        if ' --> ' in time_str:
            start_end = time_str.split(' --> ')
            return start_end[0].strip(), start_end[1].strip()
        return "00:00:00,000", "00:00:00,000"
    
    def _extract_frame_number(self, filename: str) -> int:
        """从文件名中提取帧号"""
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0
    
    def analyze_images_with_ai(self, image_paths: List[Path]) -> List[str]:
        """使用AI分析图片内容"""
        descriptions = []
        
        for img_path in image_paths:
            if self.provider == "qwen":
                desc = self._analyze_with_qwen_vl(img_path)
            else:
                desc = self._analyze_with_gemini_vl(img_path)
            
            descriptions.append(desc)
            time.sleep(1)  # 避免API调用过于频繁
        
        return descriptions
    
    def _analyze_with_qwen_vl(self, image_path: Path) -> str:
        """使用Qwen视觉模型分析图片"""
        # 这里需要配置Qwen视觉API
        # 示例实现（需要实际API接入）
        try:
            # 将图片转换为base64或其他格式
            import base64
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode()
            
            # 构建请求
            payload = {
                "model": "qwen-vl-max",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": image_data},
                            {"text": "请用一句话描述这张图片的主要内容"}
                        ]
                    }
                ],
                "max_tokens": 100
            }
            
            # 发送请求（这里需要真实的API配置）
            # response = requests.post(url, headers=headers, json=payload)
            # return response.json()['choices'][0]['message']['content']
            
            # 模拟返回
            return f"图片分析结果 - {image_path.name}"
            
        except Exception as e:
            print(f"图片分析失败 {image_path}: {e}")
            return f"图片内容分析 - {image_path.name}"
    
    def _analyze_with_gemini_vl(self, image_path: Path) -> str:
        """使用Gemini视觉模型分析图片"""
        # Gemini视觉API实现
        try:
            import google.generativeai as genai
            
            if self.api_key:
                genai.configure(api_key=self.api_key)
                
                model = genai.GenerativeModel('gemini-pro-vision')
                
                import PIL.Image
                img = PIL.Image.open(image_path)
                
                response = model.generate_content([
                    "请用一句话描述这张图片的主要内容",
                    img
                ])
                
                return response.text if response.text else "图片分析结果"
            
        except Exception as e:
            print(f"Gemini图片分析失败 {image_path}: {e}")
            return f"图片内容分析 - {image_path.name}"
    
    def generate_script(self, project_data: Dict) -> List[str]:
        """根据项目数据生成解说词"""
        # 准备上下文信息
        context = {
            'subtitles': project_data.get('subtitles', []),
            'image_descriptions': project_data.get('image_descriptions', []),
            'project_id': project_data.get('project_id', '')
        }
        
        # 调用LLM生成解说词
        script_text = self._call_text_model(context)
        
        # 格式化为1-3句，每句12-18字
        formatted_scripts = self._format_to_requirements(script_text)
        
        return formatted_scripts
    
    def _call_text_model(self, context: Dict) -> str:
        """调用文本生成模型"""
        # 构建提示词
        prompt = self._build_generation_prompt(context)
        
        try:
            if self.provider == "qwen":
                return self._call_qwen_text(prompt)
            elif self.provider == "gemini":
                return self._call_gemini_text(prompt)
            elif self.provider == "openai":
                return self._call_openai_text(prompt)
            elif self.provider == "deepseek":
                return self._call_deepseek_text(prompt)
            else:
                return self._call_gemini_text(prompt)  # 默认使用Gemini
        except Exception as e:
            print(f"文本生成失败: {e}")
            return self._fallback_generation(context)
    
    def _build_generation_prompt(self, context: Dict) -> str:
        """构建生成提示词"""
        subtitles_text = " ".join([sub.get('text', '') for sub in context['subtitles'][:5]])
        images_desc = " ".join(context['image_descriptions'][:3])
        
        prompt = f"""
        基于以下视频内容分析，请生成1-3句中文解说词：
        
        字幕内容：{subtitles_text}
        图片描述：{images_desc}
        
        要求：
        1. 生成1-3句解说词
        2. 每句必须是12-18个汉字
        3. 内容要突出视频亮点和关键信息
        4. 语言简洁生动，适合视频解说
        5. 不要包含数字标记（如1. 2. 3.）
        6. 必须保留人物名字、招式名称、法宝名称等不变。
        
        解说词：
        """
        
        return prompt

    def _call_qwen_text(self, prompt: str) -> str:
        """调用Qwen文本模型"""
        try:
            from openai import OpenAI
            
            # Use the base URL from config, default to the international compatible-mode endpoint
            base_url = self.base_url or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            
            client = OpenAI(
                api_key=self.api_key,
                base_url=base_url
            )
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7
            )
            
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"Qwen API调用失败: {e}")
            return self._fallback_generation({})

    def _call_gemini_text(self, prompt: str) -> str:
        """调用Gemini文本模型"""
        try:
            import google.generativeai as genai
            
            if self.api_key:
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel(self.model)
                response = model.generate_content(prompt)
                return response.text if response.text else ""
            else:
                return self._fallback_generation({})
        except Exception as e:
            print(f"Gemini API调用失败: {e}")
            return self._fallback_generation({})

    def _call_openai_text(self, prompt: str) -> str:
        """调用OpenAI文本模型"""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.api_key)
            if self.base_url:
                client.base_url = self.base_url
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"OpenAI API调用失败: {e}")
            return self._fallback_generation({})

    def _call_deepseek_text(self, prompt: str) -> str:
        """调用DeepSeek文本模型"""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.api_key, base_url=self.base_url or "https://api.deepseek.com")
            
            response = client.chat.completions.create(
                model=self.model or "deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"DeepSeek API调用失败: {e}")
            return self._fallback_generation({})

    def _fallback_generation(self, context: Dict) -> str:
        """备用生成方法"""
        return "视频内容精彩纷呈 值得观看的优质内容 不容错过的精彩瞬间"
    
    def _format_to_requirements(self, script_text: str) -> List[str]:
        """格式化为符合要求的解说词"""
        # 清理文本
        clean_text = script_text.replace('\n', ' ').replace('\r', ' ')
        
        # 按句号、感叹号等分割
        sentences = re.split(r'[。！？.!?]', clean_text)
        
        formatted_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 如果句子太长，尝试按逗号分割
            if len(sentence) > 18:
                parts = re.split(r'[，,]', sentence)
                for part in parts:
                    part = part.strip()
                    if 12 <= len(part) <= 18:
                        formatted_sentences.append(part)
                        if len(formatted_sentences) >= 3:
                            break
            elif 12 <= len(sentence) <= 18:
                formatted_sentences.append(sentence)
        
        # 确保不超过3句
        return formatted_sentences[:3]

    def process_all_projects(self) -> Dict[str, List[str]]:
        """处理所有项目"""
        all_results = {}
        
        project_dirs = self.find_project_dirs()
        
        if not project_dirs:
            print("未找到任何项目目录")
            return {}
        
        print(f"发现 {len(project_dirs)} 个项目:")
        for proj_dir in project_dirs:
            print(f"  - {proj_dir.name}")
        
        for project_dir in project_dirs:
            print(f"\n处理项目: {project_dir.name}")
            
            # 获取项目数据
            project_data = self.get_project_data(project_dir)
            
            # 分析图片
            print(f"  分析图片数量: {len(project_data['images'])}")
            if project_data['images']:
                project_data['image_descriptions'] = self.analyze_images_with_ai(
                    project_data['images'][:5]  # 只分析前5张图片避免过慢
                )
            
            # 生成解说词
            scripts = self.generate_script(project_data)
            
            print(f"  生成解说词: {scripts}")
            all_results[project_dir.name] = scripts
            
            # 保存结果到项目目录
            output_file = project_dir / "generated_scripts.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'project_id': project_dir.name,
                    'scripts': scripts,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }, f, ensure_ascii=False, indent=2)
            
            print(f"  结果已保存至: {output_file}")
        
        return all_results

    # Methods required by app.py
    def generate(self, subtitle_text: str, word_count: int, video_title: str = "", temperature: float = 0.7) -> List[Dict]:
        """Generate script from subtitle text"""
        # Create a prompt for script generation
        prompt = self._create_script_generation_prompt(subtitle_text, word_count, video_title)
        
        try:
            if self.provider == "qwen":
                result = self._call_qwen_text(prompt)
            elif self.provider == "gemini":
                result = self._call_gemini_text(prompt)
            elif self.provider == "openai":
                result = self._call_openai_text(prompt)
            elif self.provider == "deepseek":
                result = self._call_deepseek_text(prompt)
            else:
                result = self._call_gemini_text(prompt)
        except Exception as e:
            print(f"Script generation failed: {e}")
            result = self._fallback_script_generation(subtitle_text)
        
        # Parse the result into the expected format
        return self._parse_script_result(result)

    def _create_script_generation_prompt(self, subtitle_text: str, word_count: int, video_title: str) -> str:
        """Create prompt for script generation"""
        prompt = f"""
        基于以下视频字幕内容，请生成一段解说词：

        视频标题: {video_title or '未提供标题'}
        字幕内容: {subtitle_text[:1000]}  # Limit to first 1000 characters to avoid token limits

        要求：
        1. 将解说词分成多个段落，每个段落对应视频的一个部分
        2. 每个段落应该有明确的主题和关键词
        3. 解说词总字数控制在约 {word_count} 字左右
        4. 解说词要生动有趣，吸引观众
        5. 按照开头、正文、结尾的结构组织内容
        6. 返回格式：JSON格式，包含 narration（解说词）、keywords（关键词）、section（部分：开头/内容/结尾）

        请严格按照以下JSON格式返回，不要包含其他解释文字：
        [
          {{
            "narration": "解说词内容...",
            "keywords": ["关键词1", "关键词2"],
            "section": "开头"  // 或 "内容" 或 "结尾"
          }}
        ]
        """
        return prompt

    def _parse_script_result(self, result: str) -> List[Dict]:
        """Parse the script generation result"""
        try:
            # Try to extract JSON from the result
            import re
            json_match = re.search(r'\[(.*?)\]', result, re.DOTALL)
            if json_match:
                json_str = '[' + json_match.group(1) + ']'
                script_data = json.loads(json_str)
                return script_data
            else:
                # If no JSON found, create a basic script from the text
                paragraphs = [p.strip() for p in result.split('\n') if p.strip()]
                script_parts = []
                for i, para in enumerate(paragraphs[:3]):  # Limit to 3 parts
                    section = "开头" if i == 0 else ("结尾" if i == len(paragraphs[:3])-1 else "内容")
                    script_parts.append({
                        "narration": para,
                        "keywords": [],
                        "section": section
                    })
                return script_parts
        except Exception:
            # If parsing fails, return a simple format
            return [{
                "narration": result[:500],  # Truncate to first 500 chars
                "keywords": [],
                "section": "内容"
            }]

    def _fallback_script_generation(self, subtitle_text: str) -> List[Dict]:
        """Fallback script generation if API calls fail"""
        return [{
            "narration": f"这是一个关于视频内容的解说：{subtitle_text[:200]}...",
            "keywords": ["视频", "解说"],
            "section": "内容"
        }]

    def estimate_word_count(self, script_segments: List[Dict]) -> int:
        """Estimate total word count in script"""
        total = 0
        for segment in script_segments:
            narration = segment.get("narration", "")
            total += len(narration)
        return total

    def generate_combined_text(self, script_segments: List[Dict]) -> str:
        """Generate combined text from script segments"""
        combined = []
        for segment in script_segments:
            narration = segment.get("narration", "")
            if narration.strip():
                combined.append(narration)
        return "\n".join(combined)

    def generate_scene_narration(self, merged_scenes: List[Dict], word_count: int, video_title: str = "", temperature: float = 0.7) -> List[Dict]:
        """Generate narration for each scene based on visual analysis and subtitles"""
        scene_narrations = []
        total_scenes = len(merged_scenes)
        words_per_scene = word_count // max(total_scenes, 1)
        
        for i, scene in enumerate(merged_scenes):
            scene_info = {
                "scene_id": scene.get("scene", i+1),
                "start_time": scene.get("start", 0),
                "end_time": scene.get("end", 0),
                "duration": scene.get("duration", 0),
                "subtitle": scene.get("subtitle", ""),
                "visual": scene.get("visual", ""),
                "emotion": scene.get("emotion", "中性"),
                "actions": scene.get("visual_actions", [])
            }
            
            # Create prompt for this specific scene
            prompt = self._create_scene_narration_prompt(scene_info, words_per_scene, video_title)
            
            try:
                if self.provider == "qwen":
                    narration = self._call_qwen_text(prompt)
                elif self.provider == "gemini":
                    narration = self._call_gemini_text(prompt)
                elif self.provider == "openai":
                    narration = self._call_openai_text(prompt)
                elif self.provider == "deepseek":
                    narration = self._call_deepseek_text(prompt)
                else:
                    narration = self._call_gemini_text(prompt)
            except Exception as e:
                print(f"Scene narration generation failed for scene {i+1}: {e}")
                narration = f"场景解说：{scene_info['subtitle'][:100]}..."
            
            # Determine section type based on scene position
            if i == 0:
                section = "开头"
            elif i == len(merged_scenes) - 1:
                section = "结尾"
            else:
                section = "内容"
            
            scene_narrations.append({
                "narration": narration.strip(),
                "keywords": self._extract_keywords(narration),
                "section": section,
                # Timing information for the scene
                "scene_start": scene_info["start_time"],
                "scene_end": scene_info["end_time"],
                "scene_duration": scene_info["duration"],
            })
        
        return scene_narrations

    def _create_scene_narration_prompt(self, scene_info: Dict, target_words: int, video_title: str) -> str:
        """Create prompt for scene-specific narration"""
        actions_str = ", ".join(scene_info["actions"]) if scene_info["actions"] else "无明显动作"
        prompt = f"""
        视频标题: {video_title or '无标题视频'}
        场景ID: {scene_info['scene_id']}
        时间段: {scene_info['start_time']:.1f}s - {scene_info['end_time']:.1f}s (时长: {scene_info['duration']:.1f}s)
        情绪基调: {scene_info['emotion']}
        视觉动作: {actions_str}
        原始字幕: {scene_info['subtitle']}
        画面描述: {scene_info['visual']}

        请为这个场景生成一段解说词，要求：
        1. 内容紧扣画面描述和原始字幕
        2. 体现该场景的情绪基调
        3. 语言生动有趣，增强观看体验
        4. 字数控制在约 {target_words} 字左右
        5. 避免重复原始字幕的内容，而是进行扩展和解读
        6. 必须保留人物名字、招式名称、法宝名称等不变

        解说词：
        """
        return prompt

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction - in practice, you might use more sophisticated NLP
        import re
        # Find capitalized words, repeated words, or significant nouns
        words = re.findall(r'\b\w{2,}\b', text)
        unique_words = list(set(words))
        # Return top 3-5 most meaningful words
        return unique_words[:5] if len(unique_words) > 5 else unique_words
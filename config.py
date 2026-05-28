"""
Configuration module for the Video Commentary Tool.
Centralizes all settings, API keys, and paths.
"""

import os
import tempfile
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.absolute()
TEMP_DIR = BASE_DIR / "temp_output"
TEMP_DIR.mkdir(exist_ok=True)

# ============================================================
# API Configuration
# ============================================================
# Supported LLM providers for subtitle extraction & script generation
LLM_PROVIDERS = {
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    },
    "gemini": {
        "api_key": os.getenv("GEMINI_API_KEY", ""),
        "models": [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-3.1-pro",
            "gemini-3.1-flash-lite",
            "gemini-3.0-flash",
            "gemini-3.5-flash",
        ],
        "model_display": {
            "gemini-2.5-flash": "Gemini 2.5 Flash 🚀",
            "gemini-2.5-pro": "Gemini 2.5 Pro 💪",
            "gemini-2.0-flash": "Gemini 2.0 Flash ⚡",
            "gemini-2.0-flash-lite": "Gemini 2.0 Flash Lite 💨",
            "gemini-1.5-flash": "Gemini 1.5 Flash",
            "gemini-1.5-pro": "Gemini 1.5 Pro",
            "gemini-3.1-pro": "Gemini 3.1 Pro 🧠 (最新旗舰)",
            "gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite 💨 (最快最新)",
            "gemini-3.0-flash": "Gemini 3 Flash ⚡",
            "gemini-3.5-flash": "Gemini 3.5 Flash 🚀",
        },
    },
    "qwen": {
        "api_key": os.getenv("QWEN_API_KEY", ""),
        "base_url": os.getenv("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
        "base_url_options": [
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "https://dashscope.aliyuncs.com/api/v1",
            "https://dashscope-intl.aliyuncs.com/api/v1",
        ],
        "models": [
            "qwen-max",
            "qwen-plus",
            "qwen-turbo",
            "qwen3.5-plus-2026-02-15",
            "qwen3.5-plus-2026-04-20",
            "qwen3.5-plus",
            "qwen3.6-plus-2026-04-02",
            "qwen3.6-plus",
            "qwen3.6-flash",
            "qwen-vl-max",
            "qwen3.5-omni",
            "qwen-vl-ocr",
            "qwen-vl-ocr-2025-11-20",
            "qwen3-asr-flash",
            "qwen3-asr-flash-filetrans-2025-11-17",
            "qwen3-asr-flash-filetrans",
            "qwen3-asr-flash-2026-02-10",
            "qwen3-asr-flash-realtime",
            "qwen3-asr-flash-realtime-2025-10-27",
            "qwen3-asr-flash-realtime-2026-02-10",
            "qwen3-asr-flash-2025-09-08",
        ],
        "model_display": {
            "qwen-max": "Qwen Max",
            "qwen-plus": "Qwen Plus",
            "qwen-turbo": "Qwen Turbo",
            "qwen3.5-plus-2026-02-15": "Qwen3.5 Plus (2026-02-15)",
            "qwen3.5-plus-2026-04-20": "Qwen3.5 Plus (2026-04-20)",
            "qwen3.5-plus": "Qwen3.5 Plus",
            "qwen3.6-plus-2026-04-02": "Qwen3.6 Plus (2026-04-02)",
            "qwen3.6-plus": "Qwen3.6 Plus",
            "qwen3.6-flash": "Qwen3.6 Flash",
            "qwen-vl-max": "Qwen VL Max (视觉)",
            "qwen3.5-omni": "Qwen3.5 Omni",
            "qwen-vl-ocr": "Qwen VL OCR",
            "qwen-vl-ocr-2025-11-20": "Qwen VL ORC (2025-11-20)",
            "qwen3-asr-flash": "Qwen3 ASR Flash (默认)",
            "qwen3-asr-flash-filetrans-2025-11-17": "Qwen3 ASR Flash FileTrans (2025-11-17)",
            "qwen3-asr-flash-filetrans": "Qwen3 ASR Flash FileTrans",
            "qwen3-asr-flash-2026-02-10": "Qwen3 ASR Flash (2026-02-10)",
            "qwen3-asr-flash-realtime": "Qwen3 ASR Flash Realtime",
            "qwen3-asr-flash-realtime-2025-10-27": "Qwen3 ASR Flash Realtime (2025-10-27)",
            "qwen3-asr-flash-realtime-2026-02-10": "Qwen3 ASR Flash Realtime (2026-02-10)",
            "qwen3-asr-flash-2025-09-08": "Qwen3 ASR Flash (2025-09-08)",
        },
    },
    "deepseek": {
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
}

# Subtitle extraction models
SUBTITLE_PROVIDERS = {
    "faster_whisper": {
        "model_size": os.getenv("WHISPER_MODEL_SIZE", "medium"),
        "device": os.getenv("WHISPER_DEVICE", "auto"),
        "compute_type": os.getenv("WHISPER_COMPUTE_TYPE", "float16"),
    },
}

# ============================================================
# FunASR Configuration
# ============================================================
FUNASR_CONFIG = {
    "model_size": "large",        # "large" or "small"
    "device": "cuda",             # "cuda" or "cpu"
}

# ============================================================
# Video Processing Configuration
# ============================================================
VIDEO_CONFIG = {
    "min_duration": 150,       # 2 minutes 30 seconds (seconds)
    "max_duration": 360,       # 6 minutes (seconds)
    "input_min_duration": 1200, # 20 minutes
    "input_max_duration": 1500, # 25 minutes
    "frame_extraction_interval": 5,  # Extract a frame every N seconds
    "clip_buffer": 1.0,        # Buffer seconds around selected clips
}

# ============================================================
# Script Configuration
# ============================================================
WORD_COUNT_OPTIONS = [1200, 1800, 2500, 3500]

DEFAULT_WORD_COUNT = 1800

# ============================================================
# TTS Configuration
# ============================================================
# Rate: negative = slower, positive = faster (e.g. "-10%" = 10% slower)
# Typical Chinese commentary: 3-5 chars/second
# Default "-10%" gives a natural, unhurried narration pace
TTS_CONFIG = {
    "rate": "-10%",
    "volume": "+0%",
    "pitch": "+0Hz",
}

# Estimated speaking speed (characters per second) at default rate
# Used to calculate narration duration and auto-extend video clips
CHARS_PER_SECOND = {
    "zh-CN": 3.5,   # Chinese: ~3.5 chars/sec at -10% rate
    "vi-VN": 3.8,   # Vietnamese: ~3.8 chars/sec
    "en-US": 4.0,   # English: ~4 chars/sec
}

# Edge TTS voice mappings
# Maps display name -> (voice_name, language_code)
# Each language has 2 female + 2 male voices
TTS_VOICES = {
    # Chinese (zh-CN)
    "中文（女声① - Xiaoxiao）": ("zh-CN-XiaoxiaoNeural", "zh-CN"),
    "中文（女声② - Xiaohan）": ("zh-CN-XiaohanNeural", "zh-CN"),
    "中文（男声① - Yunxi）": ("zh-CN-YunxiNeural", "zh-CN"),
    "中文（男声② - Yunyang）": ("zh-CN-YunyangNeural", "zh-CN"),
    # Vietnamese (vi-VN) — 2 female + 1 male (Edge TTS library limitation)
    "越南语（女声① - Hoài My）": ("vi-VN-HoaiMyNeural", "vi-VN"),
    "越南语（女声② - Hoài An）": ("vi-VN-HoaiAnNeural", "vi-VN"),
    "越南语（男声① - Nam Minh）": ("vi-VN-NamMinhNeural", "vi-VN"),
    "越南语（男声② - Nam Minh）": ("vi-VN-NamMinhNeural", "vi-VN"),
    # English US (en-US)
    "英语（美式女声① - Jenny）": ("en-US-JennyNeural", "en-US"),
    "英语（美式女声② - Aria）": ("en-US-AriaNeural", "en-US"),
    "英语（美式男声① - Brian）": ("en-US-BrianNeural", "en-US"),
    "英语（美式男声② - Christopher）": ("en-US-ChristopherNeural", "en-US"),
}

SUPPORTED_LANGUAGES = {k: v[1] for k, v in TTS_VOICES.items()}

# Word count options per mode
WORD_COUNT_OPTIONS_MODE = {
    "donghua": [1200, 1800, 2500, 3500],
    "sports": [300, 450, 600, 800],
}

# ============================================================
# Mode Configuration
# ============================================================
VIDEO_MODES = [
    ("donghua", "🎬 影视解说（修仙/玄幻动画）", "zh"),
    ("sports", "🏀 ESPN 体育分析", "en"),
]
# Target durations per mode (seconds)
MODE_DURATIONS = {
    "donghua": {"min": 150, "max": 360},   # 2:30 – 6:00
    "sports":  {"min": 90,  "max": 150},   # 1:30 – 2:30
}

# ============================================================
# Scene Analysis Configuration
# ============================================================
SCENE_CONFIG = {
    "num_top_scenes": 10,
    "clip_duration": 15,  # seconds per clip
    "transition_duration": 0.5,
}

# ============================================================
# Scene Detection Configuration (新)
# ============================================================
SCENE_DETECTION_CONFIG = {
    "method": "auto",              # "pyscenedetect", "histogram", "fixed", "auto"
    "threshold": 30.0,             # Sensitivity (lower = more scenes)
    "min_scene_duration": 3.0,     # Minimum scene length in seconds
    "max_scenes": 60,              # Maximum number of scenes
    "frames_per_scene": 3,         # Representative frames per scene (1-5)
    "visual_model": "gemini-2.5-flash",  # Vision model for visual analysis
    "visual_models": [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-3.1-pro",
        "gemini-3.1-flash-lite",
        "gemini-3.0-flash",
        "gemini-3.5-flash",
    ],  # Available vision models
}

# ============================================================
# Narration Timing Configuration (新)
# ============================================================
NARRATION_TIMING_CONFIG = {
    "max_narration_ratio": 1.3,    # Max ratio of narration_duration / scene_duration
                                   # If narration takes >1.3x the scene length, needs adjustment
    "min_narration_ratio": 0.3,    # Min ratio (narration too short for the scene)
    "speed_adjust_threshold": 1.5, # If ratio > this, use speed adjust instead of just clip extend
    "extension_methods": [         # Priority order for fixing long narration
        "shorten_script",          # 1. Shorten the narration script
        "extend_clip",             # 2. Extend the video clip
        "slow_motion",             # 3. Apply slow motion to clip
        "add_broll",               # 4. Insert B-roll / alternate footage
    ],
}

# ============================================================
# Timeline / Scene Steps Configuration (新)
# ============================================================
# New steps added to the workflow
EXTRA_STEPS = [
    "scenes",      # Scene detection
    "visual",       # Visual understanding / analysis
]

# ============================================================
# Prompt Template
# ============================================================
PROMPT_TEMPLATE = """你是一位专门解说中国3D修仙、玄幻动画的专家，例如《仙逆》《斗破苍穹》《完美世界》《吞噬星空》等。

【影片信息】
影片标题：{video_title}
目标字数：{word_count}字左右

【字幕内容】
{subtitle_text}

【核心规则】
1. **内容一致性**：文案中的人物名称、地名、功法、法宝等专有名词，必须与原始字幕保持完全一致，严禁虚构或篡改。
2. **叙述方式**：全程使用第三人称叙述（他/她/它/他们），禁止使用第一人称（我）或第二人称（你，除非是引用原话）。
3. **段落结构**：每一段解说词必须扩写为 2 到 3 句话，保持节奏紧凑。
4. **字数控制**：解说词总字数控制在 {word_count} 字左右。

【文案结构】
必须严格按照以下三部分结构撰写：

=== 第一部分：开头（前3句话必须足够吸引人）===
从以下方式中选择一种：
方式一：概括总结型（推荐新手）
- 这是XXX年来尺度最大/最震撼/最烧脑的修仙动画...
- 这部动画被XX万人打出9.X高分，被称为...
- 这是国产3D动画巅峰之作，却被尘封多年...
- 他被称为修仙界最狠/最惨/最强的人物...
- 这是一部让人肾上腺素飙升的爽片...

方式二：情景式、假设式
- 他叫...你以为他是...吗？不，他是来...
- 如果给你...你会怎么办？
- 你知道...吗？原来...

方式三：以国家/平台作为开头
- 这是国产3D动画史上最...
- 这是腾讯视频评分最高的修仙动画...
- 这部动画在某些地区被下架...

方式四：自由发挥（围绕核心主题）
- 这不是动画，而是真实修仙故事...
- 他力大无穷、双眼发光，你以为他是超人？其实不是...

=== 第二部分：内容（正文叙述）===
核心逻辑：
1. 围绕影片主线展开剧情
2. 用简单直接的方式讲清故事
3. 大量使用修饰词增强节奏感与反转感

必须使用的修饰词（至少选择10个以上）：
竟然、突然、原来、但是、可是、结果、直到、如果、而、果然、发现、只是、出奇、之后、没错、不止、更是、当然、因为、所以、没想到、谁知、岂料、然而、尽管、即便、纵使

要求：
- 讲述主线剧情，突出高潮和反转
- 每段解说词配2-4个关键词，用于匹配原视频画面
- 语言口语化、幽默风趣，适当加入网络流行语
- 剧情连贯，突出修仙世界的等级、功法、战斗等元素

=== 第三部分：结尾===
选择以下方式之一：
方式一：人生感悟
- 这部动画告诉我们...
- 有时候真正可怕的不是敌人，而是...
- 修仙之路，从来都不是一帆风顺...

方式二：保留悬念
- 最后的反转让人意想不到...
- 故事到这里还没结束...

方式三：总结推荐
- 这是一部值得反复观看的佳作...
- 如果你喜欢修仙题材，这部绝对不能错过...

【输出格式】
必须输出严格的JSON格式，包含以下结构：
{{
    "script": [
        {{
            "narration": "第一段解说词（开头部分，前3句话必须足够吸引人）",
            "keywords": ["关键词1", "关键词2", "关键词3"],
            "section": "开头"
        }},
        {{
            "narration": "第二段解说词（内容部分，讲述剧情）",
            "keywords": ["战斗", "修炼", "升级"],
            "section": "内容"
        }},
        ...
        {{
            "narration": "最后一段解说词（结尾部分，感悟或总结）",
            "keywords": ["结局", "感悟"],
            "section": "结尾"
        }}
    ]
}}

【注意事项】
1. 严格遵守【核心规则】中的所有要求。
2. 输出严格遵循JSON Schema，不要包含```json标记。

现在请开始创作：
"""

# ============================================================
# ESPN Sports Analysis Prompt Template
# ============================================================
ESPN_PROMPT_TEMPLATE = """You are an elite ESPN sports analyst and cinematic sports storyteller.

Your task is to analyze the uploaded sports video, understand the full context of the match/game, identify the most emotional, explosive, controversial, and momentum-shifting moments, then rewrite the entire sequence into an intense ESPN-style narration script.

【MATCH INFO】
Video Title: {video_title}
Target Duration: {word_count} words (~{duration_sec} seconds of narration)

【SUBTITLE / SCENE DATA】
{subtitle_text}

STYLE REQUIREMENTS:
- Keep ALL names of people and places exactly as they appear in the subtitles — never change, translate, or invent them.
- Sound like ESPN First Take, Undisputed, TNT Sports, or NBA on ESPN.
- High energy, fast pacing, dramatic storytelling, emotional reactions.
- Build tension constantly. Use momentum swings.
- Emphasize: pressure, legacy, domination, clutch moments, collapse, revenge, humiliation, comeback, superstar aura.
- Every 1-2 sentences MUST feel impactful. Avoid robotic play-by-play descriptions.
- Do NOT simply describe actions — narrate emotions, pressure, crowd reactions, body language, confidence shifts, psychological warfare.
- Keep the full script between {word_count} words.

OUTPUT FORMAT — strict JSON:
{{
    "script": [
        {{
            "narration": "ESPN-style narration line(s) for this moment. 1-3 punchy sentences.",
            "keywords": ["keyword1", "keyword2", "keyword3"],
            "section": "opening|body|closing",
            "music_mood": "tense|epic|hype|dramatic|triumphant",
            "editing_tip": "slow-mo|replay|zoom-in|crowd-cut|impact-flash"
        }}
    ]
}}

NARRATION TONE EXAMPLES:
"THIS… is where everything changed."
"Look at the reaction from the crowd!"
"He thought he had the matchup won… until THIS happened."
"That's not just a bucket. That's a statement."
"Absolute domination."
"This building just exploded."
"Cold-blooded."
"Welcome to superstar basketball."

IMPORTANT:
- Output ONLY the JSON. No markdown, no code fences, no extra text.
- Keep narration concise and punchy.
- Every line should increase hype.
"""

# ============================================================
# Qwen 国漫修仙解说专属 Prompt（Scene-by-Scene 专用）
# ============================================================
QWEN_SCENE_NARRATION_PROMPT = """你是一位深耕国漫3D玄幻/修仙赛道的资深影视解说编剧，长期服务B站/抖音头部动漫解说账号。你精通《仙逆》《吞噬星空》《师兄啊师兄》《斗破苍穹》等作品的叙事逻辑，擅长将复杂修仙设定转化为通俗爽点，文案风格冷静克制但暗藏张力，节奏紧凑、单句简短、网感强，能精准踩中"战力碾压/智谋布局/悬念留钩"的流量密码。

请根据我提供的【分镜时间轴+画面描述+字幕/台词】，自动生成一篇可直接用于配音剪辑的影视解说脚本。

📥 输入数据格式：
[00:XX-00:XX] 画面描述 | 字幕/台词 | 分镜节奏/情绪提示
（按视频实际顺序提供，可分段粘贴）

📤 输出要求：
1. 结构必须包含：
   🔹 黄金3秒开头：用悬念/反差/高能台词瞬间抓人（不拖沓，直接切入核心冲突）
   🔹 剧情主线推进：按时间轴梳理"起因→布局→交锋→转折"，不注水、不跳逻辑
   🔹 战力/设定解析：修仙术语（如神识、法相、问鼎、第二步）需用半句通俗比喻带过，避免说明书式科普
   🔹 爽点提炼：点出主角的压制感、智谋或底牌，强化"碾压/反杀/留后手"情绪
   🔹 悬念结尾+互动引导：抛出未解势力/伏笔，自然引导评论（如"你觉得幻眉小姐会亲自下场吗？评论区押个战力"）
   🔹 必须保留人物名字、招式名称、法宝名称等不变。
2. 篇幅与节奏：
   - 适配2.5~6分钟短视频，总字数 1200~1800 字
   - 语速锚定 220~240 字/分钟，单句严格控制在 12~16字以内
   - 每 15~20 秒设置一个情绪爆点或信息转折
3. 剪辑适配标注（每段文案后需附带）：
   【配音语气】（如：低沉铺垫/语速加快/压低悬疑/戛然而止）
   【BGM建议】（如：悬疑弦乐/燃向战鼓/灵力音效/留白静音）
   【画面剪辑提示】（如：快剪卡点/慢放特写/字幕放大/转场黑屏）
   【字幕排版】（如：关键台词加粗/情绪词变色/断行节奏）

⚠️ 文风规范（严格执行）：
- 口语化叙事，禁用"首先/其次/综上所述/值得注意的是"等AI模板词
- 修仙语境需准确：境界压制用"威压/神识碾压"，法宝用"祭出/催动"，战斗用"交锋/溃散/反制"
- 不替角色加戏、不魔改原剧情、不剧透未提供片段
- 爽点不靠吼，靠"信息差+实力差"制造压迫感
- 结尾必须留钩子：势力背景/未登场大佬/伏笔物品/主角下一步目标

🎯 风格参照（以 Video 2 为例）：
"柳眉的修为到底有多强大？竟然仅靠威压，就将三大家族老祖震得动弹不了。而王林能不能打得过他呢？就在刚刚，王林不仅用一枚仙玉，拍下了8品次神丹，而且还与三大家族的老祖达成协议……"
（要求：语气冷静但暗藏张力，信息密度高，术语自然融入，节奏随战斗/布局起伏，结尾留势力悬念）

请严格按上述要求输出完整解说脚本。若输入数据含战斗/法宝/境界词，自动匹配对应修仙解说话术；若含日常/过渡镜头，用1句话轻描淡写带过，不拖节奏。

只输出脚本正文，不要JSON、不要```标记。"""


def get_random_project_name() -> str:
    """Generate a random project name to avoid duplicates."""
    import uuid
    return f"project_{uuid.uuid4().hex[:12]}"

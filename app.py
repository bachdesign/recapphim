"""
影视解说工具 - Video Commentary Tool (Step-by-Step Edition)
A Streamlit application for creating commentary videos from Chinese donghua/anime.
Each step is reviewable and editable before proceeding to the next.
"""

import json
import os
import shutil
import tempfile
import time
import traceback
from pathlib import Path
from typing import List, Dict, Optional

import streamlit as st

st.set_page_config(
    page_title="影视解说工具 - Video Commentary Tool",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Update page title via session
if "ui_language" not in st.session_state:
    st.session_state.ui_language = "zh"

from locales import _ as _translate

def __(key: str, **fmt) -> str:
    """Translate key using current session language."""
    lang = st.session_state.get("ui_language", "zh")
    return _translate(key, lang, **fmt)

from config import (
    LLM_PROVIDERS,
    VIDEO_CONFIG,
    WORD_COUNT_OPTIONS,
    WORD_COUNT_OPTIONS_MODE,
    SUPPORTED_LANGUAGES,
    SCENE_CONFIG,
    SCENE_DETECTION_CONFIG,
    NARRATION_TIMING_CONFIG,
    FUNASR_CONFIG,
    VIDEO_MODES,
    MODE_DURATIONS,
    get_random_project_name,
)
from utils.subtitle_extractor import SubtitleExtractor
from utils.script_generator import ScriptGenerator
from utils.video_processor import VideoProcessor
from utils.tts_engine import TTSEngine
from utils.video_editor import VideoEditor
from utils.scene_analyzer import SceneAnalyzer
from utils.scene_detector import SceneDetector
from utils.visual_analyzer import VisualAnalyzer
from utils.cancel_utils import (
    raise_if_cancelled, TaskCancelledError,
    request_cancel as _cancel_util_request,
    reset_cancel as _cancel_util_reset,
    pause as _cancel_util_pause,
    resume as _cancel_util_resume,
)


# ============================================================
# API Key persistence
# ============================================================
API_KEYS_FILE = Path("temp") / "api_keys.json"


def load_saved_api_keys():
    """Load previously saved API keys from disk into session state."""
    if API_KEYS_FILE.exists():
        try:
            with open(API_KEYS_FILE, "r") as f:
                saved = json.load(f)
            for key in ("subtitle_api_key", "script_api_key", "scene_api_key"):
                if key in saved and saved[key]:
                    st.session_state[key] = saved[key]
        except Exception:
            pass  # ignore corrupt file


def save_api_keys():
    """Persist current API keys to disk immediately."""
    data = {}
    for key in ("subtitle_api_key", "script_api_key", "scene_api_key"):
        val = st.session_state.get(key, "")
        if val:
            data[key] = val
    API_KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(API_KEYS_FILE, "w") as f:
        json.dump(data, f)


# ============================================================
# Project Save / Load persistence
# ============================================================
PROJECTS_INDEX_FILE = Path("temp") / "projects_index.json"

# Session state keys to persist (skip transient/temp keys)
SAVE_EXCLUDE_KEYS = {
    "previous_step", "api_keys_set",
    "subtitle_api_key", "script_api_key", "scene_api_key",
    "subtitle_qwen_model_selector", "script_qwen_model_selector",
    "scene_qwen_model_selector", "visual_qwen_model_selector",
    "subtitle_qwen_url_selector", "script_qwen_url_selector",
    "scene_qwen_url_selector", "visual_qwen_url_selector",
    "subtitle_gemini_selector", "script_gemini_selector",
    "scene_gemini_selector", "visual_gemini_selector",
}

# Streamlit widget key prefixes — excluded from save to avoid conflicts
# Widget key prefixes — excluded from save/load to avoid conflicts.
# NOTE: do NOT clean these on every rerun or button clicks break.
WIDGET_KEY_PREFIXES = (
    "prev_", "next_",
    "script_nar_", "script_kw_", "script_sec_", "new_script_",
    "hs_", "he_", "hr_", "add_hs", "add_he", "add_hr",
    "subtitle_gemini_selector", "script_gemini_selector",
    "scene_gemini_selector", "visual_gemini_selector",
    "subtitle_file_upload", "script_file_upload",
    "ui_", "mode_",
    "cancel_task_btn", "pause_task_btn",
    "load_project_selector",
    "load_",
    "del_",  # <--- THÊM DÒNG NÀY ĐỂ FIX LỖI CHO NÚT DELETE PROJECT
)

# Keys that store file paths — need path rewriting on load
PATH_KEYS = {"video_path", "narration_audio", "final_video"}


def get_saved_projects() -> List[Dict]:
    """Load the list of all saved projects from the index file."""
    if PROJECTS_INDEX_FILE.exists():
        try:
            with open(PROJECTS_INDEX_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_project_index(projects: List[Dict]):
    """Write the project index to disk."""
    PROJECTS_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROJECTS_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(projects, f, ensure_ascii=False, indent=2)


def save_project_snapshot(name: str = None) -> str:
    """
    Save the current session state as a project snapshot.
    Writes a .project.json file inside the project directory.
    Returns the snapshot name.
    """
    if name is None:
        name = st.session_state.get("project_name", get_random_project_name())

    snapshot = {
        "project_name": name,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "current_step": st.session_state.get("current_step", "upload"),
        "data": {},
    }

    # Collect all serialisable session state
    for k, v in st.session_state.items():
        if k in SAVE_EXCLUDE_KEYS:
            continue
        # Skip streamlit internal / widget keys
        if k.startswith("_") or k.startswith("Form") or k.startswith("form"):
            continue
        if k.startswith(WIDGET_KEY_PREFIXES):
            continue
        try:
            # Test JSON serialisability
            json.dumps(v)
            snapshot["data"][k] = v
        except (TypeError, OverflowError):
            pass  # skip unserialisable objects

    # Save to project directory
    project_dir = Path("temp") / name
    project_dir.mkdir(parents=True, exist_ok=True)
    snap_path = project_dir / "project_snapshot.json"
    with open(snap_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    # Update project index
    projects = get_saved_projects()
    # Remove old entry with same name
    projects = [p for p in projects if p["name"] != name]
    projects.append({
        "name": name,
        "saved_at": snapshot["saved_at"],
        "step": snapshot["current_step"],
        "step_label": STEP_NAMES.get(snapshot["current_step"], snapshot["current_step"]),
        "video_title": st.session_state.get("video_title", ""),
        "has_script": bool(st.session_state.get("script")),
        "has_audio": bool(st.session_state.get("narration_audio")),
        "has_video": bool(st.session_state.get("final_video")),
    })
    save_project_index(projects)

    print(f"[Save] Project '{name}' saved at step {snapshot['current_step']}")
    return name


def load_project_snapshot(name: str) -> bool:
    """
    Load a saved project snapshot into session state.
    Returns True on success.
    """
    snap_path = Path("temp") / name / "project_snapshot.json"
    if not snap_path.exists():
        return False

    try:
        with open(snap_path, "r", encoding="utf-8") as f:
            snapshot = json.load(f)
    except Exception as e:
        print(f"[Load] Failed to read snapshot: {e}")
        return False

    data = snapshot.get("data", {})

    # First, clean up ANY stale widget keys from session state
    widget_keys_to_del = [k for k in st.session_state.keys() if k.startswith(WIDGET_KEY_PREFIXES)]
    for k in widget_keys_to_del:
        del st.session_state[k]

    # Restore session state keys, skipping widget keys from snapshot
    for k, v in data.items():
        if k.startswith(WIDGET_KEY_PREFIXES):
            continue
        st.session_state[k] = v

    # Rewrite file paths to absolute paths in the current environment
    project_dir = Path("temp") / name
    for path_key in PATH_KEYS:
        old_path = st.session_state.get(path_key)
        if old_path and isinstance(old_path, str) and os.path.exists(old_path):
            # Path is valid as-is
            pass
        elif old_path:
            # Try to resolve relative to project directory
            candidate = project_dir / Path(old_path).name
            if candidate.exists():
                st.session_state[path_key] = str(candidate)
            else:
                # File doesn't exist — clear the path so user can regenerate
                print(f"[Load] File not found for {path_key}: {old_path}, clearing")
                st.session_state[path_key] = None

    # Ensure project dir structure exists
    for sub in ["frames", "clips", "audio", "output", "subtitles"]:
        (project_dir / sub).mkdir(parents=True, exist_ok=True)

    print(f"[Load] Project '{name}' loaded at step {st.session_state.get('current_step')}")
    return True


def delete_saved_project(name: str):
    """Remove a project from the index (does not delete files)."""
    projects = get_saved_projects()
    projects = [p for p in projects if p["name"] != name]
    save_project_index(projects)


def auto_save():
    """Auto-save current project on critical steps."""
    name = st.session_state.get("project_name", "")
    if name:
        save_project_snapshot(name)


# ============================================================
# Cancel / Pause Task Controls
# ============================================================

def reset_cancel():
    """Reset the cancel and pause flags."""
    _cancel_util_reset()
    st.session_state._cancel_requested = False
    st.session_state._paused = False
    st.session_state._task_running = False
    st.session_state._task_description = ""


def request_cancel():
    """Request cancellation of the current task."""
    _cancel_util_request()
    st.session_state._cancel_requested = True


def toggle_pause():
    """Toggle pause state."""
    if st.session_state.get("_paused", False):
        _cancel_util_resume()
        st.session_state._paused = False
    else:
        _cancel_util_pause()
        st.session_state._paused = True


class TaskCancelledError(Exception):
    """Raised when the user cancels a long-running task."""
    pass


def render_cancel_button(description: str = "当前任务"):
    """
    Render a cancel / pause control bar.
    Place this near long-running operations.
    """
    st.session_state._task_running = True
    st.session_state._task_description = description

    col_c, col_p = st.columns([1, 1])
    with col_c:
        st.button(
            "⏹️ 停止" if st.session_state.get("ui_language","zh")=="zh" else "⏹️ Stop",
            type="secondary",
            use_container_width=True,
            on_click=request_cancel,
            key="cancel_task_btn",
        )
    with col_p:
        pause_label = (
            "▶️ 继续" if st.session_state.get("_paused", False) else "⏸️ 暂停"
        ) if st.session_state.get("ui_language","zh")=="zh" else (
            "▶️ Resume" if st.session_state.get("_paused", False) else "⏸️ Pause"
        )
        st.button(
            pause_label,
            type="secondary",
            use_container_width=True,
            on_click=toggle_pause,
            key="pause_task_btn",
        )


# ============================================================
# Pages / Steps definition
# ============================================================
# 正确流程:
# 1. Upload → 2. Subtitles → 3. Scenes (场景检测)
# → 4. Visual (画面分析) → 5. Script (逐场景解说)
# → 6. Audio → 7. Export
STEPS = [
    ("upload",    "📁", "上传视频"),
    ("subtitles", "🎤", "音频/字幕提取"),
    ("scenes",    "🎞️", "场景检测"),
    ("visual",    "👁️", "画面分析"),
    ("script",    "✍️", "解说脚本"),
    ("audio",     "🔊", "配音生成"),
    ("export",    "🎬", "导出视频"),
]

STEP_NAMES = {s[0]: s[2] for s in STEPS}
STEP_INDICES = {s[0]: i for i, s in enumerate(STEPS)}


# ============================================================
# Session State Initialization
# ============================================================
def init_session_state():
    defaults = {
        # project
        "project_name": get_random_project_name(),
        "current_step": "upload",
        "previous_step": None,
        # video
        "video_path": None,
        "video_duration": 0.0,
        "video_title": "",
        # subtitles
        "subtitles": [],
        "subtitle_text": "",
        "subtitle_provider": "faster_whisper",
        "whisper_model": "medium",
        "funasr_model": "large",
        # frames (legacy, kept for backward compatibility)
        "frames": [],
        # highlights (legacy, kept for backward compatibility)
        "highlights": [],
        "highlight_edits": {},
        # === NEW: Scene detection ===
        "scenes": [],                # List of scene dicts from SceneDetector
        "scene_data_merged": [],     # Merged scenes with subtitles + visual
        # script
        "script": [],
        "script_json": "",
        "word_count": 1800,
        # audio
        "narration_audio": None,
        "tts_language": "中文（女声① - Xiaoxiao）",
        # final
        "final_video": None,
        # config
        "script_provider": "gemini",
        "scene_provider": "gemini",
        "api_keys_set": False,
        "ui_language": "zh",
        "video_mode": "donghua",
        # API key inputs (will be loaded from disk if saved)
        "subtitle_api_key": "",
        "script_api_key": "",
        "scene_api_key": "",
        # Visual analysis settings
        "visual_provider": "gemini",
        "scene_detection_method": "auto",
        "frames_per_scene": 3,
        # Selected Gemini models (per provider)
        "subtitle_gemini_model": "gemini-2.5-flash",
        "script_gemini_model": "gemini-2.5-flash",
        "scene_gemini_model": "gemini-2.5-flash",
        "visual_gemini_model": "gemini-2.5-flash",
        # Selected Qwen models & server (per provider)
        "subtitle_qwen_model": "qwen-max",
        "script_qwen_model": "qwen-max",
        "scene_qwen_model": "qwen-max",
        "visual_qwen_model": "qwen-vl-max",
        "qwen_base_url": LLM_PROVIDERS["qwen"]["base_url_options"][0],
        # Cancel / pause controls
        "_cancel_requested": False,
        "_paused": False,
        "_task_running": False,
        "_task_description": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    # Load previously saved API keys
    load_saved_api_keys()


init_session_state()


# ============================================================
# Helpers
# ============================================================
def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def get_project_dir() -> Path:
    d = Path("temp") / st.session_state.project_name
    for sub in ["frames", "clips", "audio", "output", "subtitles"]:
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def go_to_step(step: str):
    st.session_state.previous_step = st.session_state.current_step
    st.session_state.current_step = step
    auto_save()
    # Streamlit auto-reruns when session state changes during callbacks


def go_next():
    steps = [s[0] for s in STEPS]
    cur = STEP_INDICES[st.session_state.current_step]
    if cur + 1 < len(steps):
        go_to_step(steps[cur + 1])


def go_prev():
    steps = [s[0] for s in STEPS]
    cur = STEP_INDICES[st.session_state.current_step]
    if cur - 1 >= 0:
        go_to_step(steps[cur - 1])


def step_nav(current: str, next_label="下一步 →", next_disabled=False, position="top"):
    """Render prev/next navigation buttons."""
    c1, c2, c3 = st.columns([1, 2, 1])
    steps = [s[0] for s in STEPS]
    idx = STEP_INDICES[current]
    suffix = f"_{position}"
    prev_label = __("prev_step")
    with c1:
        if idx > 0:
            if st.button(prev_label, use_container_width=True, key=f"prev_{current}{suffix}"):
                go_prev()
    with c3:
        if idx < len(steps) - 1:
            next_label_loc = __("next_step")
            st.button(next_label_loc, type="primary", use_container_width=True,
                      disabled=next_disabled, on_click=go_next,
                      key=f"next_{current}{suffix}")
    return c2


# ============================================================
# Page: Upload & Configuration
# ============================================================
def render_upload_page():
    """Render the video upload and configuration page."""
    st.title("🎬 影视解说工具")
    st.markdown("### 将20-25分钟的视频，逐步编辑，自动生成2.5-6分钟的精彩解说视频")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(f"#### {__('upload_video')}")
        uploaded_file = st.file_uploader(
            __('upload_video_desc'),
            type=["mp4", "avi", "mov", "mkv"],
            help=__('upload_video_help'),
        )

        if uploaded_file is not None:
            project_dir = get_project_dir()
            video_path = str(project_dir / "input_video.mp4")
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.session_state.video_path = video_path
            st.success(f"✅ {__('video_uploaded')}: {uploaded_file.name}")

            processor = VideoProcessor()
            duration = processor.get_video_duration(video_path)
            st.session_state.video_duration = duration
            st.info(f"⏱️ {__('video_duration')}: {format_time(duration)}")

            min_dur = VIDEO_CONFIG["input_min_duration"] // 60
            max_dur = VIDEO_CONFIG["input_max_duration"] // 60
            if duration < VIDEO_CONFIG["input_min_duration"]:
                st.warning(f"⚠️ {__('video_too_short', min=min_dur)}")
            elif duration > VIDEO_CONFIG["input_max_duration"]:
                st.warning(f"⚠️ {__('video_too_long', max=max_dur)}")

            st.video(video_path)

    with col2:
        st.markdown(f"#### {__('config_options')}")

        with st.expander(f"🤖 {__('ai_model_config')}", expanded=True):
            st.session_state.subtitle_provider = st.selectbox(
                __('subtitle_engine'),
                options=["faster_whisper", "funasr", "qwen-asr", "openai", "gemini", "qwen", "deepseek"],
                index=["faster_whisper","funasr","qwen-asr","openai","gemini","qwen","deepseek"].index(st.session_state.subtitle_provider),
            )

            if st.session_state.subtitle_provider == "faster_whisper":
                st.session_state.whisper_model = st.selectbox(
                    __('whisper_model_size'),
                    options=["tiny", "base", "small", "medium", "large-v3"],
                    index=["tiny","base","small","medium","large-v3"].index(st.session_state.whisper_model),
                    help=__('whisper_help'),
                )
            elif st.session_state.subtitle_provider == "qwen-asr":
                _asr_models = [m for m in LLM_PROVIDERS["qwen"]["models"] if "asr" in m]
                _asr_display = LLM_PROVIDERS["qwen"].get("model_display", {})
                _asr_options = [f"{m} — {_asr_display.get(m, m)}" for m in _asr_models]
                _current_asr = st.session_state.get("qwen_asr_model", "fun-asr")
                _asr_idx = _asr_models.index(_current_asr) if _current_asr in _asr_models else 0
                st.session_state.qwen_asr_model = _asr_models[
                    _asr_options.index(st.selectbox(
                        "Qwen-ASR 模型",
                        options=_asr_options,
                        index=_asr_idx,
                        key="qwen_asr_selector",
                    ))
                ]
                st.text_input(
                    f"🔑 QWEN {__('api_key_label')}",
                    type="password",
                    key="subtitle_api_key",
                    placeholder=__('api_key_placeholder'),
                    on_change=save_api_keys,
                )
            elif st.session_state.subtitle_provider == "funasr":
                st.session_state.funasr_model = st.selectbox(
                    "FunASR 模型大小",
                    options=["large", "small"],
                    index=["large", "small"].index(st.session_state.get("funasr_model", "large")),
                    help="large: 更精准但需要更多显存 | small: 更快更省显存",
                )
            else:
                st.text_input(
                    f"🔑 {st.session_state.subtitle_provider.upper()} {__('api_key_label')}",
                    type="password",
                    key="subtitle_api_key",
                    placeholder=__('api_key_placeholder'),
                    on_change=save_api_keys,
                )
                # Gemini model selector for subtitle extraction
                if st.session_state.subtitle_provider == "gemini":
                    _gemini_models = LLM_PROVIDERS["gemini"]["models"]
                    _gemini_display = LLM_PROVIDERS["gemini"].get("model_display", {})
                    _model_options = [f"{m} — {_gemini_display.get(m, m)}" for m in _gemini_models]
                    _current_model = st.session_state.get("subtitle_gemini_model", "gemini-2.5-flash")
                    _current_idx = _gemini_models.index(_current_model) if _current_model in _gemini_models else 0
                    _selected = st.selectbox(
                        "Gemini 模型",
                        options=_model_options,
                        index=_current_idx,
                        key="subtitle_gemini_selector",
                    )
                    st.session_state.subtitle_gemini_model = _gemini_models[_model_options.index(_selected)]
                if st.session_state.subtitle_provider == "qwen":
                    _render_qwen_config("subtitle")
                    on_change=save_api_keys,                
                # Gemini model selector for subtitle extraction
                if st.session_state.subtitle_provider == "gemini":
                    _gemini_models = LLM_PROVIDERS["gemini"]["models"]
                    _gemini_display = LLM_PROVIDERS["gemini"].get("model_display", {})
                    _model_options = [f"{m} — {_gemini_display.get(m, m)}" for m in _gemini_models]
                    _current_model = st.session_state.get("subtitle_gemini_model", "gemini-2.5-flash")
                    _current_idx = _gemini_models.index(_current_model) if _current_model in _gemini_models else 0
                    _selected = st.selectbox(
                        "Gemini 模型",
                        options=_model_options,
                        index=_current_idx,
                        key="subtitle_gemini_selector",
                    )
                    st.session_state.subtitle_gemini_model = _gemini_models[_model_options.index(_selected)]
                if st.session_state.subtitle_provider == "qwen":
                    _render_qwen_config("subtitle")

            st.markdown("---")
            st.session_state.script_provider = st.selectbox(
                __('script_engine'),
                options=["gemini", "openai", "qwen", "deepseek"],
                index=["gemini","openai","qwen","deepseek"].index(st.session_state.script_provider),
            )
            st.text_input(
                f"🔑 {st.session_state.script_provider.upper()} {__('api_key_script')}",
                type="password",
                key="script_api_key",
                placeholder=__('api_key_placeholder'),
                on_change=save_api_keys,
            )
            # Gemini model selector for script generation
            if st.session_state.script_provider == "gemini":
                _gemini_models = LLM_PROVIDERS["gemini"]["models"]
                _gemini_display = LLM_PROVIDERS["gemini"].get("model_display", {})
                _model_options = [f"{m} — {_gemini_display.get(m, m)}" for m in _gemini_models]
                _current_model = st.session_state.get("script_gemini_model", "gemini-2.5-flash")
                _current_idx = _gemini_models.index(_current_model) if _current_model in _gemini_models else 0
                _selected = st.selectbox(
                    "Gemini 模型",
                    options=_model_options,
                    index=_current_idx,
                    key="script_gemini_selector",
                )
                st.session_state.script_gemini_model = _gemini_models[_model_options.index(_selected)]
            if st.session_state.script_provider == "qwen":
                _render_qwen_config("script")

            st.markdown("---")
            st.session_state.scene_provider = st.selectbox(
                __('scene_engine'),
                options=["gemini", "openai", "qwen", "deepseek"],
                index=["gemini","openai","qwen","deepseek"].index(st.session_state.scene_provider),
            )
            st.text_input(
                f"🔑 {st.session_state.scene_provider.upper()} {__('api_key_scene')}",
                type="password",
                key="scene_api_key",
                placeholder=__('api_key_placeholder'),
                on_change=save_api_keys,
            )
            # Gemini model selector for scene analysis
            if st.session_state.scene_provider == "gemini":
                _gemini_models = LLM_PROVIDERS["gemini"]["models"]
                _gemini_display = LLM_PROVIDERS["gemini"].get("model_display", {})
                _model_options = [f"{m} — {_gemini_display.get(m, m)}" for m in _gemini_models]
                _current_model = st.session_state.get("scene_gemini_model", "gemini-2.5-flash")
                _current_idx = _gemini_models.index(_current_model) if _current_model in _gemini_models else 0
                _selected = st.selectbox(
                    "Gemini 模型",
                    options=_model_options,
                    index=_current_idx,
                    key="scene_gemini_selector",
                )
                st.session_state.scene_gemini_model = _gemini_models[_model_options.index(_selected)]
            if st.session_state.scene_provider == "qwen":
                _render_qwen_config("scene")

            # Key status indicators
            saved_keys_labels = {"subtitle_api_key": __("subtitles_count"), "script_api_key": __("script_segments"), "scene_api_key": __("highlights_count")}
            saved_active = [v for k, v in saved_keys_labels.items() if st.session_state.get(k)]
            if saved_active:
                st.caption(__("keys_auto_saved", keys=", ".join(saved_active)))
            else:
                st.caption(__("keys_hint"))

        with st.expander(f"🎯 {__('narration_config')}", expanded=True):
            mode = st.session_state.get("video_mode", "donghua")
            wc_options = WORD_COUNT_OPTIONS_MODE.get(mode, WORD_COUNT_OPTIONS)
            # Adjust current word_count if it's not in the new options
            if st.session_state.word_count not in wc_options:
                st.session_state.word_count = wc_options[1] if len(wc_options) > 1 else wc_options[0]
            st.session_state.word_count = st.selectbox(
                __('target_words'),
                options=wc_options,
                index=wc_options.index(st.session_state.word_count),
            )
            st.session_state.video_title = st.text_input(
                __('video_title'),
                placeholder=__('video_title_placeholder'),
                value=st.session_state.video_title,
            )

        with st.expander(f"🎚️ {__('tts_config')}", expanded=True):
            # Speech rate slider
            if "speech_rate" not in st.session_state:
                st.session_state.speech_rate = -10
            st.session_state.speech_rate = st.slider(
                "语速调整 (%)" if st.session_state.get("ui_language","zh")=="zh" 
                else "Speech Rate (%)" if st.session_state.get("ui_language","zh")=="en"
                else "Tốc độ nói (%)",
                min_value=-50, max_value=50, value=st.session_state.speech_rate, step=5,
                help="负数=放慢，正数=加快。推荐 -10% 到 0%",
            )
            st.session_state.tts_language = st.selectbox(
                __('tts_language'),
                options=list(SUPPORTED_LANGUAGES.keys()),
                index=list(SUPPORTED_LANGUAGES.keys()).index(st.session_state.tts_language),
            )

        st.markdown("---")
        st.markdown(f"#### {__('mode_select')}")
        current_mode_idx = [m[0] for m in VIDEO_MODES].index(st.session_state.get("video_mode", "donghua"))
        mode_options = [f"{m[1]}" for m in VIDEO_MODES]
        selected_mode = st.radio(
            "Video Mode",
            options=mode_options,
            index=current_mode_idx,
            label_visibility="collapsed",
            key="mode_radio",
        )
        # Map back to mode id
        for m_id, m_label, _ in VIDEO_MODES:
            if m_label == selected_mode:
                st.session_state.video_mode = m_id
                break
        # Show mode-specific info
        if st.session_state.video_mode == "sports":
            dur = MODE_DURATIONS["sports"]
            st.info(f"🏀 ESPN Sports Mode — output {dur['min']//60}:{dur['min']%60:02d} ~ {dur['max']//60}:{dur['max']%60:02d} min")
        else:
            dur = MODE_DURATIONS["donghua"]
            st.info(f"🎬 影视解说模式 — 输出 {dur['min']//60}:{dur['min']%60:02d} ~ {dur['max']//60}:{dur['max']%60:02d} 分钟")

        if st.button(
            __('start_processing'),
            type="primary",
            use_container_width=True,
            disabled=st.session_state.video_path is None,
        ):
            st.session_state.api_keys_set = True
            go_to_step("subtitles")


# ============================================================
# Helper: get API key from session
# ============================================================
def _api_key(prefix: str) -> str:
    """Read API key from session state or environment variable."""
    # Map prefix to session state key
    key_map = {
        "openai": "subtitle_api_key",
        "gemini": "subtitle_api_key",
        "qwen": "subtitle_api_key",
        "deepseek": "subtitle_api_key",
        "script_openai": "script_api_key",
        "script_gemini": "script_api_key",
        "script_qwen": "script_api_key",
        "script_deepseek": "script_api_key",
        "scene_openai": "scene_api_key",
        "scene_gemini": "scene_api_key",
        "scene_qwen": "scene_api_key",
        "scene_deepseek": "scene_api_key",
        "visual_openai": "scene_api_key",
        "visual_gemini": "scene_api_key",
        "visual_qwen": "scene_api_key",
        "visual_deepseek": "scene_api_key",
    }
    val = st.session_state.get(key_map.get(prefix, ""), "")
    if val:
        return val
    # Fallback to environment variable
    env_map = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "qwen": "QWEN_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    import os
    return os.getenv(env_map.get(prefix, ""), "")


def _render_qwen_config(category: str, label_prefix: str = ""):
    """Render Qwen model selector + server URL selector.
    category: "subtitle", "script", "scene", or "visual"
    """
    _qwen_models = LLM_PROVIDERS["qwen"]["models"]
    _qwen_display = LLM_PROVIDERS["qwen"].get("model_display", {})
    _model_options = [f"{m} — {_qwen_display.get(m, m)}" for m in _qwen_models]
    _current_model = st.session_state.get(f"{category}_qwen_model", "qwen-max")
    _current_idx = _qwen_models.index(_current_model) if _current_model in _qwen_models else 0

    # Model selector
    _selected = st.selectbox(
        f"{label_prefix}Qwen 模型",
        options=_model_options,
        index=_current_idx,
        key=f"{category}_qwen_model_selector",
    )
    st.session_state[f"{category}_qwen_model"] = _qwen_models[_model_options.index(_selected)]

    # Server URL selector
    _url_options = LLM_PROVIDERS["qwen"]["base_url_options"]
    _current_url = st.session_state.get("qwen_base_url", _url_options[0])
    _current_url_idx = _url_options.index(_current_url) if _current_url in _url_options else 0
    _selected_url = st.selectbox(
        f"{label_prefix}Qwen 服务器",
        options=_url_options,
        index=_current_url_idx,
        key=f"{category}_qwen_url_selector",
        help="国内: dashscope.aliyuncs.com | 国际: dashscope-intl.aliyuncs.com",
    )
    st.session_state.qwen_base_url = _selected_url


def _resolve_qwen_key() -> str:
    """Resolve Qwen API key from all possible sources."""
    key = (
        st.session_state.get("scene_api_key", "")
        or st.session_state.get("script_api_key", "")
        or st.session_state.get("subtitle_api_key", "")
        or os.environ.get("QWEN_API_KEY", "")
    ).strip()  # 去除首尾空白

    if key:
        os.environ["DASHSCOPE_API_KEY"] = key
        masked = f"{key[:6]}****{key[-4:]}" if len(key) > 12 else "****"
        print(f"[Key] Qwen key ({len(key)} chars): {masked}")
        if not key.startswith("sk-") or len(key) < 20:
            print(f"[Key] \u26a0\ufe0f Warning: Key format may be invalid")
    return key


def validate_qwen_key(key: str) -> bool:
    """Quick sanity check: DashScope keys start with sk- and are >20 chars."""
    cleaned = key.strip()
    return cleaned.startswith("sk-") and len(cleaned) > 20


def _gemini_model(category: str) -> str:
    """Get the selected Gemini model for a given category.
    category: "subtitle", "script", "scene", or "visual"
    """
    key = f"{category}_gemini_model"
    model = st.session_state.get(key, "")
    if model and model in LLM_PROVIDERS["gemini"]["models"]:
        return model
    return "gemini-2.5-flash"  # default


def _resolve_model(provider: str, category: str) -> str:
    """Resolve the model name for a given provider and category.
    For Gemini/Qwen, returns the user-selected model.
    For others, returns the first model from config.
    """
    if provider == "gemini":
        return _gemini_model(category)
    if provider == "qwen":
        key = f"{category}_qwen_model"
        model = st.session_state.get(key, "")
        models = LLM_PROVIDERS["qwen"]["models"]
        if model and model in models:
            return model
        return models[0] if models else "qwen-max"
    models = LLM_PROVIDERS.get(provider, {}).get("models", [None])
    return models[0] if models else None


# ============================================================
# SRT / subtitle file parsing helpers
# ============================================================
def parse_srt(content: str) -> List[Dict]:
    """Parse SRT format text into list of {text, start, end}."""
    import re
    subtitles = []
    # SRT format: index \n HH:MM:SS,mmm --> HH:MM:SS,mmm \n text \n\n
    pattern = re.compile(
        r"\d+\s*\n(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*\n(.*?)(?=\n\n|\Z)",
        re.DOTALL,
    )
    for m in pattern.finditer(content):
        start = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + int(m.group(4)) / 1000
        end = int(m.group(5)) * 3600 + int(m.group(6)) * 60 + int(m.group(7)) + int(m.group(8)) / 1000
        text = m.group(9).strip().replace("\n", " ")
        subtitles.append({"text": text, "start": round(start, 2), "end": round(end, 2)})
    return subtitles


def parse_txt_subtitles(content: str) -> List[Dict]:
    """Read plain text, one subtitle per line, no timestamps."""
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    return [{"text": line, "start": i * 5.0, "end": (i + 1) * 5.0} for i, line in enumerate(lines)]


def load_subtitles_from_file(uploaded_file) -> List[Dict]:
    """Detect format and load subtitles from an uploaded file."""
    raw = uploaded_file.read().decode("utf-8", errors="replace")
    name = uploaded_file.name.lower()
    if name.endswith(".srt"):
        subs = parse_srt(raw)
        if subs:
            return subs
    # fallback: plain text
    return parse_txt_subtitles(raw)


# ============================================================
# Step Page: Subtitles
# ============================================================
def render_subtitles_page():
    st.title(f"🎤 {__('subtitle_title')}")
    st.markdown(__('subtitle_desc'))

    step_nav("subtitles", next_disabled=not st.session_state.subtitles)

    project_dir = get_project_dir()
    video_path = st.session_state.video_path

    with st.expander(f"📂 {__('upload_subtitle_file')}", expanded=not st.session_state.subtitles):
        uploaded_sub = st.file_uploader(
            __('subtitle_file_label'),
            type=["srt", "txt"],
            key="subtitle_file_upload",
            help=__('upload_subtitle_help'),
        )
        if uploaded_sub is not None:
            try:
                subs = load_subtitles_from_file(uploaded_sub)
                if subs:
                    st.session_state.subtitles = subs
                    st.session_state.subtitle_text = "\n".join([s["text"] for s in subs])
                    sub_path = project_dir / "subtitles" / "full_subtitles.json"
                    with open(sub_path, "w", encoding="utf-8") as f:
                        json.dump(subs, f, ensure_ascii=False, indent=2)
                    txt_path = project_dir / "subtitles" / "full_subtitles.txt"
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(st.session_state.subtitle_text)
                    st.success(f"✅ {__('subtitle_imported', count=len(subs), name=uploaded_sub.name)}")
                    st.rerun()
                else:
                    st.error(f"❌ {__('subtitle_parse_failed')}")
            except Exception as e:
                st.error(f"❌ {__('subtitle_import_failed')}: {e}")

    if not st.session_state.subtitles:
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(__('or_extract_from_video'))
        with col2:
            if st.button(f"🎤 {__('extract_subtitles')}", type="primary", use_container_width=True):
                reset_cancel()
                status_placeholder = st.empty()
                with status_placeholder.container():
                    st.info(f"⏳ {__('extracting_subtitles')}")
                    render_cancel_button("字幕提取")
                try:
                    provider = st.session_state.subtitle_provider
                    if provider == "faster_whisper":
                        extractor = SubtitleExtractor(provider=provider, project_dir=project_dir, model_size=st.session_state.whisper_model)
                    elif provider == "funasr":
                        extractor = SubtitleExtractor(
                            provider=provider,
                            project_dir=project_dir,
                            model_size=st.session_state.get("funasr_model", "large"),
                            device=FUNASR_CONFIG.get("device", "cuda"),
                        )
                    elif provider == "qwen-asr":
                        qwen_key = _resolve_qwen_key()
                        qwen_url = st.session_state.get("qwen_base_url", "https://dashscope-intl.aliyuncs.com")
                        asr_model = st.session_state.get("qwen_asr_model", "fun-asr")
                        extractor = SubtitleExtractor(
                            provider=provider,
                            project_dir=project_dir,
                            api_key=qwen_key,
                            base_url=qwen_url,
                            model=asr_model,
                        )
                    else:
                        model_name = _resolve_model(provider, "subtitle")
                        base_url = LLM_PROVIDERS.get(provider, {}).get("base_url", "")
                        extractor = SubtitleExtractor(provider=provider, project_dir=project_dir, api_key=_api_key(provider), model=model_name, base_url=base_url)

                    tts_lang = SUPPORTED_LANGUAGES[st.session_state.tts_language]
                    lang = "zh" if tts_lang == "zh-CN" else tts_lang[:2]
                    subtitles = extractor.extract(video_path, language=lang)
                    st.session_state.subtitles = subtitles
                    st.session_state.subtitle_text = "\n".join([s["text"] for s in subtitles])
                    auto_save()

                    sub_path = project_dir / "subtitles" / "full_subtitles.json"
                    with open(sub_path, "w", encoding="utf-8") as f:
                        json.dump(subtitles, f, ensure_ascii=False, indent=2)
                    txt_path = project_dir / "subtitles" / "full_subtitles.txt"
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(st.session_state.subtitle_text)

                    reset_cancel()
                    status_placeholder.empty()
                    st.success(f"✅ {__('subtitles_extracted', count=len(subtitles))}")
                    st.rerun()
                except TaskCancelledError:
                    status_placeholder.empty()
                    st.warning("⏹️ 字幕提取已取消")
                except Exception as e:
                    status_placeholder.empty()
                    st.error(f"❌ {__('subtitle_extract_failed')}: {e}")
    else:
        st.success(f"✅ {__('subtitles_extracted', count=len(st.session_state.subtitles))}")

        edited_text = st.text_area(
            f"📝 {__('edit_subtitles')}",
            st.session_state.subtitle_text,
            height=400,
        )

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button(f"💾 {__('save_changes')}", use_container_width=True):
                lines = [l.strip() for l in edited_text.split("\n") if l.strip()]
                old_timestamps = [(s["start"], s["end"]) for s in st.session_state.subtitles]
                new_subs = []
                for i, line in enumerate(lines):
                    ts = old_timestamps[i] if i < len(old_timestamps) else (0, 0)
                    new_subs.append({"text": line, "start": ts[0], "end": ts[1]})
                st.session_state.subtitles = new_subs
                st.session_state.subtitle_text = "\n".join(lines)
                txt_path = project_dir / "subtitles" / "full_subtitles.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(st.session_state.subtitle_text)
                auto_save()
                st.success(f"✅ {__('subtitle_saved')}")
                st.rerun()
        with col2:
            if st.button(f"🔄 {__('re_extract')}", use_container_width=True):
                st.session_state.subtitles = []
                st.session_state.subtitle_text = ""
                st.rerun()

        with st.expander(f"📋 {__('subtitle_list')}", expanded=False):
            for i, s in enumerate(st.session_state.subtitles):
                st.caption(f"{i+1:04d}. [{format_time(s['start'])} → {format_time(s['end'])}] {s['text']}")

    step_nav("subtitles", next_disabled=not st.session_state.subtitles, position="bottom")


# ============================================================
# Step Page: Frames
# ============================================================
# ============================================================
# Step Page: Scene Detection (新: 1. 场景检测)
# ============================================================
def render_scenes_page():
    st.title(f"🎞️ 场景检测")
    st.markdown('将视频自动拆分为 timeline scenes。AI 必须知道\u201c哪一段正在发生什么事情\u201d。')

    step_nav("scenes", next_disabled=not st.session_state.scenes)

    project_dir = get_project_dir()
    video_path = st.session_state.video_path

    if not video_path:
        st.warning("⚠️ 请先上传视频")
        return

    if not st.session_state.subtitles:
        st.warning("⚠️ 请先在「字幕提取」步骤提取字幕")
        return

    # Config options
    with st.expander("⚙️ 场景检测设置", expanded=not st.session_state.scenes):
        col1, col2 = st.columns(2)
        with col1:
            method = st.selectbox(
                "检测方法",
                options=["auto", "pyscenedetect", "histogram", "fixed"],
                index=["auto", "pyscenedetect", "histogram", "fixed"].index(
                    st.session_state.get("scene_detection_method", "auto")
                ),
                help="auto: 优先使用 PySceneDetect，失败自动降级",
            )
            st.session_state.scene_detection_method = method
        with col2:
            fps = st.number_input(
                "每场景帧数",
                min_value=1, max_value=5, value=st.session_state.get("frames_per_scene", 3),
                help="每个场景提取的代表帧数量（用于后续视觉分析）",
            )
            st.session_state.frames_per_scene = fps

    if not st.session_state.scenes:
        if st.button("🚀 检测场景", type="primary", use_container_width=True):
            reset_cancel()
            status_ph = st.empty()
            with status_ph.container():
                st.info("⏳ 正在检测场景边界，分析视频结构...")
                render_cancel_button("场景检测")
            try:
                detector = SceneDetector(
                    method=st.session_state.scene_detection_method,
                    threshold=SCENE_DETECTION_CONFIG["threshold"],
                )
                # 1. Detect scenes
                scenes = detector.detect_scenes(
                    video_path,
                    min_scene_duration=SCENE_DETECTION_CONFIG["min_scene_duration"],
                    max_scenes=SCENE_DETECTION_CONFIG["max_scenes"],
                )

                if not scenes:
                    st.error("❌ 未能检测到场景，请尝试其他方法")
                    return

                # 2. Match subtitles to scenes
                scenes = SceneDetector.get_scene_subtitles(scenes, st.session_state.subtitles)

                # 3. Extract representative frames per scene
                frame_dir = project_dir / "frames" / "scene_frames"
                scenes = SceneDetector.extract_scene_frames(
                    video_path, scenes, str(frame_dir),
                    frames_per_scene=st.session_state.frames_per_scene,
                )

                st.session_state.scenes = scenes
                auto_save()
                reset_cancel()
                status_ph.empty()
                st.success(f"✅ 检测到 {len(scenes)} 个场景，共提取 {sum(s['frame_count'] for s in scenes)} 帧画面")
                st.rerun()
            except TaskCancelledError:
                status_ph.empty()
                st.warning("⏹️ 场景检测已取消")
            except Exception as e:
                status_ph.empty()
                st.error(f"❌ 场景检测失败: {e}")
                import traceback
                st.code(traceback.format_exc())
    else:
        scenes = st.session_state.scenes
        st.success(f"✅ 检测到 {len(scenes)} 个场景")

        # Show scene overview
        total_dur = sum(s["duration"] for s in scenes)
        st.info(f"📊 总时长: {total_dur:.1f}s | 平均场景: {total_dur/len(scenes):.1f}s")

        # Display scenes in a table
        scene_data = []
        for s in scenes:
            scene_data.append({
                "场景": s["scene_id"],
                "开始": f"{format_time(s['start'])}",
                "结束": f"{format_time(s['end'])}",
                "时长": f"{s['duration']:.1f}s",
                "字幕": len(s.get("subtitles", [])),
                "帧数": s.get("frame_count", 0),
            })
        st.dataframe(scene_data, use_container_width=True)

        # Show scene details
        for s in scenes[:10]:  # Show first 10 in detail
            with st.expander(f"🎬 场景 {s['scene_id']}: {format_time(s['start'])} → {format_time(s['end'])} ({s['duration']:.1f}s)", expanded=False):
                # Show frames
                if s.get("frames"):
                    cols = st.columns(len(s["frames"]))
                    for j, fp in enumerate(s["frames"]):
                        with cols[j]:
                            # Check if the image file exists before trying to display it
                            if os.path.exists(fp):
                                st.image(fp, caption=f"帧 {j+1}", use_container_width=True)
                            else:
                                st.text(f"图像不存在: {os.path.basename(fp)}")
                # Show subtitles
                if s.get("subtitles"):
                    st.caption("📝 字幕:")
                    for sub in s["subtitles"]:
                        st.text(f"  [{sub['start']:.1f}s → {sub['end']:.1f}s] {sub['text']}")

        if st.button("🔄 重新检测", use_container_width=True):
            st.session_state.scenes = []
            st.session_state.scene_data_merged = []
            st.rerun()

    step_nav("scenes", next_disabled=not st.session_state.scenes, position="bottom")


# ============================================================
# Step Page: Visual Analysis (新: 4. AI视觉分析)
# ============================================================
def render_visual_page():
    st.title(f"👁️ 画面分析")
    st.markdown("""
    AI 分析每个 scene 的视觉内容，理解 visual semantics。
    
    **输入**: 每个场景的字幕 + 代表帧图片
    **输出**: 画面描述、动作列表、情绪基调
    """)

    step_nav("visual", next_disabled=not st.session_state.scene_data_merged)

    if not st.session_state.scenes:
        st.warning("⚠️ 请先在「场景检测」步骤检测场景")
        return

    if not st.session_state.scene_data_merged:
        # Provider selection
        col1, col2 = st.columns(2)
        with col1:
            vp = st.selectbox(
                "视觉分析引擎",
                options=["gemini", "openai", "qwen", "deepseek"],
                index=["gemini", "openai", "qwen", "deepseek"].index(
                    st.session_state.get("visual_provider", "gemini")
                ),
                help="Gemini: 原生视觉理解 | OpenAI/Qwen: 图片base64 | DeepSeek: 文本推断（无视觉）",
            )
            st.session_state.visual_provider = vp
            # Gemini model selector for visual analysis
            if vp == "gemini":
                _gemini_models = LLM_PROVIDERS["gemini"]["models"]
                _gemini_display = LLM_PROVIDERS["gemini"].get("model_display", {})
                _model_options = [f"{m} — {_gemini_display.get(m, m)}" for m in _gemini_models]
                _current_model = st.session_state.get("visual_gemini_model", "gemini-2.5-flash")
                _current_idx = _gemini_models.index(_current_model) if _current_model in _gemini_models else 0
                _selected = st.selectbox(
                    "Gemini 模型",
                    options=_model_options,
                    index=_current_idx,
                    key="visual_gemini_selector",
                )
                st.session_state.visual_gemini_model = _gemini_models[_model_options.index(_selected)]
            # Qwen model selector for visual analysis
            if vp == "qwen":
                _render_qwen_config("visual")
        with col2:
            st.caption("视觉分析需要 API Key")
            st.caption("使用上传页配置的 API Key")

        if st.button("👁️ 开始视觉分析", type="primary", use_container_width=True):
            reset_cancel()
            status_ph = st.empty()
            with status_ph.container():
                st.info("⏳ AI 正在分析每个场景的画面内容...")
                render_cancel_button("视觉分析")
            try:
                vp = st.session_state.visual_provider
                visual_model = _resolve_model(vp, "visual")
                # Pass base_url for Qwen (user-selected server)
                qwen_base_url = st.session_state.get("qwen_base_url", "")
                # Try all possible Qwen key locations
                if vp == "qwen":
                    qwen_key = _resolve_qwen_key()
                    if not qwen_key:
                        st.warning("⚠️ 未找到 Qwen API Key！请在上传页配置")
                        return
                    elif not validate_qwen_key(qwen_key):
                        st.warning("⚠️ Qwen Key 格式异常！应以 `sk-` 开头且长度 >20")
                        return
                else:
                    qwen_key = _api_key(f"visual_{vp}")
                analyzer = VisualAnalyzer(
                    provider=vp,
                    api_key=qwen_key,
                    model=visual_model,
                    base_url=qwen_base_url if vp == "qwen" else LLM_PROVIDERS.get(vp, {}).get("base_url", ""),
                    mode=st.session_state.get("video_mode", "donghua"),
                )

                # Analyze all scenes with visual data
                analyzed = analyzer.analyze_all_scenes(st.session_state.scenes)

                # Merge into unified format
                merged = VisualAnalyzer.merge_scene_data(analyzed)

                st.session_state.scene_data_merged = merged
                auto_save()
                reset_cancel()
                status_ph.empty()
                st.success(f"✅ 已完成 {len(merged)} 个场景的视觉分析")
                st.rerun()
            except TaskCancelledError:
                status_ph.empty()
                st.warning("⏹️ 视觉分析已取消")
            except Exception as e:
                status_ph.empty()
                st.error(f"❌ 视觉分析失败: {e}")
                import traceback
                st.code(traceback.format_exc())
    else:
        merged = st.session_state.scene_data_merged
        st.success(f"✅ 已完成 {len(merged)} 个场景的视觉分析")

        # Show merged data
        for scene in merged:
            emoji_map = {
                "高潮": "🔥", "紧张": "⚡", "悲伤": "😢", "平静": "🌊",
                "激烈": "💥", "欢乐": "😊", "神秘": "🔮", "中性": "➖",
            }
            emo_emoji = emoji_map.get(scene.get("emotion", "中性"), "➖")
            with st.expander(
                f"🎬 场景 {scene['scene']}: {format_time(scene['start'])} → {format_time(scene['end'])} "
                f"({scene['duration']:.1f}s) {emo_emoji} {scene.get('emotion', '中性')}",
                expanded=False,
            ):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("**📝 字幕:**")
                    st.info(scene.get("subtitle", "(无字幕)")[:200])
                    st.markdown("**👁️ 画面描述:**")
                    st.success(scene.get("visual", "(无描述)"))
                with col2:
                    st.markdown(f"**💭 情绪:** {scene.get('emotion', '中性')}")
                    st.markdown("**🎬 视觉动作:**")
                    for action in scene.get("visual_actions", []):
                        st.markdown(f"- {action}")
                    # Show frame thumbnails
                    if scene.get("frames"):
                        st.markdown("**🖼️ 代表帧:**")
                        for fp in scene["frames"]:
                            # Check if the image file exists before trying to display it
                            if os.path.exists(fp):
                                st.image(fp, use_container_width=True)
                            else:
                                st.text(f"图像不存在: {os.path.basename(fp)}")
        
        # Calculate total narration character count and estimated duration
        total_narration_chars = sum(len(s.get("subtitle", "")) for s in merged)
        
        # Estimate total duration using the same logic as in video_editor.py
        from utils.video_editor import VideoEditor
        est_total_duration = 0
        for s in merged:
            subtitle = s.get("subtitle", "")
            duration = VideoEditor._estimate_narration_duration(subtitle, "zh-CN", -10)  # Assuming default rate
            est_total_duration += duration
            
        total_scene_dur = sum(s["duration"] for s in merged)
        st.info(
            f"📊 字幕总长: ~{total_narration_chars} 字 | "
            f"预估朗读: {est_total_duration:.1f}s | "
            f"场景总长: {total_scene_dur:.1f}s | "
            f"比例: {est_total_duration/max(total_scene_dur,1):.2f}"
        )
def render_script_page():
    st.title(f"✍️ {__('script_title')}")
    st.markdown(__('script_desc'))

    step_nav("script", next_disabled=not st.session_state.script)

    if not st.session_state.subtitle_text:
        st.warning(f"⚠️ {__('no_script_warning')}")
        return

    with st.expander(f"📂 {__('upload_script_file')}", expanded=not st.session_state.script):
        uploaded_script = st.file_uploader(
            __('script_file_label'),
            type=["json", "txt"],
            key="script_file_upload",
            help=__('script_file_help'),
        )
        if uploaded_script is not None:
            try:
                script = load_script_from_file(uploaded_script)
                if script:
                    st.session_state.script = script
                    st.session_state.script_json = json.dumps({"script": script}, ensure_ascii=False, indent=2)
                    script_path = get_project_dir() / "output" / "script.json"
                    with open(script_path, "w", encoding="utf-8") as f:
                        f.write(st.session_state.script_json)
                    st.success(f"✅ {__('script_imported', count=len(script), name=uploaded_script.name)}")
                    st.rerun()
                else:
                    st.error(f"❌ {__('script_parse_failed')}")
            except Exception as e:
                st.error(f"❌ {__('script_import_failed')}: {e}")

    if not st.session_state.script:
        st.markdown("---")

        # Check if scene data is available for the new scene-by-scene method
        has_scene_data = bool(st.session_state.get("scene_data_merged"))

        if has_scene_data:
            st.info("✅ 检测到场景数据，将使用「逐场景解说」模式（推荐）")
            st.markdown("**正确流程：** 每个场景独立生成解说词，基于视觉分析 + 字幕")

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🎬 逐场景生成解说", type="primary", use_container_width=True):
                    reset_cancel()
                    status_ph = st.empty()
                    with status_ph.container():
                        st.info("⏳ 正在逐场景生成解说词（每个场景独立调用AI）...")
                        render_cancel_button("逐场景解说")
                    try:
                        sp = st.session_state.script_provider
                        model_name = _resolve_model(sp, "script")
                        if sp == "qwen":
                            script_key = _resolve_qwen_key()
                            if not script_key:
                                st.warning("⚠️ 未找到 Qwen API Key！请在上传页配置")
                                return
                            elif not validate_qwen_key(script_key):
                                st.warning("⚠️ Qwen Key 格式异常！应以 `sk-` 开头且长度 >20")
                                return
                        else:
                            script_key = _api_key(f"script_{sp}")
                        gen = ScriptGenerator(
                            provider=sp,
                            api_key=script_key,
                            model=model_name,
                            base_url=LLM_PROVIDERS.get(sp, {}).get("base_url", ""),
                            mode=st.session_state.get("video_mode", "donghua"),
                        )
                        script = gen.generate_scene_narration(
                            merged_scenes=st.session_state.scene_data_merged,
                            word_count=st.session_state.word_count,
                            video_title=st.session_state.video_title,
                            temperature=0.7,
                        )

                        # Check narration timing
                        from utils.video_editor import VideoEditor
                        lang_code = SUPPORTED_LANGUAGES[st.session_state.tts_language]
                        timed_script = VideoEditor.check_narration_timing(
                            script_segments=script,
                            scene_data=st.session_state.scene_data_merged,
                            language=lang_code,
                            rate=st.session_state.get("speech_rate", -10),
                        )

                        st.session_state.script = timed_script
                        st.session_state.script_json = json.dumps(
                            {"script": timed_script}, ensure_ascii=False, indent=2
                        )
                        est = gen.estimate_word_count(timed_script)
                        auto_save()
                        reset_cancel()
                        status_ph.empty()
                        st.success(f"✅ 逐场景解说生成完成！共 {len(timed_script)} 段，约 {est} 字")
                        st.rerun()
                    except TaskCancelledError:
                        status_ph.empty()
                        st.warning("⏹️ 解说生成已取消")
                    except Exception as e:
                        status_ph.empty()
                        st.error(f"❌ 逐场景解说生成失败: {e}")
                        import traceback
                        st.code(traceback.format_exc())

            with col2:
                st.caption("或使用传统方式：")
                if st.button(f"✍️ 传统全文生成", use_container_width=True):
                    reset_cancel()
                    status_ph = st.empty()
                    with status_ph.container():
                        st.info(f"⏳ {__('generating')}")
                        render_cancel_button("全文解说")
                    try:
                        sp = st.session_state.script_provider
                        model_name = _resolve_model(sp, "script")
                        script_key = _resolve_qwen_key() if sp == "qwen" else _api_key(f"script_{sp}")
                        gen = ScriptGenerator(
                            provider=sp,
                            api_key=script_key,
                            model=model_name,
                            base_url=LLM_PROVIDERS.get(sp, {}).get("base_url", ""),
                            mode=st.session_state.get("video_mode", "donghua"),
                        )
                        script = gen.generate(
                            subtitle_text=st.session_state.subtitle_text,
                            word_count=st.session_state.word_count,
                            video_title=st.session_state.video_title,
                            temperature=0.7,
                        )
                        st.session_state.script = script
                        st.session_state.script_json = json.dumps({"script": script}, ensure_ascii=False, indent=2)
                        est = gen.estimate_word_count(script)
                        auto_save()
                        reset_cancel()
                        status_ph.empty()
                        st.success(f"✅ {__('script_generated', words=est)}")
                        st.rerun()
                    except TaskCancelledError:
                        status_ph.empty()
                        st.warning("⏹️ 解说生成已取消")
                    except Exception as e:
                        status_ph.empty()
                        st.error(f"❌ {__('script_failed')}: {e}")
        else:
            # No scene data — use traditional full script generation
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(__('or_generate_with_ai', wc=st.session_state.word_count))
            with col2:
                if st.button(f"✍️ {__('generate_script')}", type="primary", use_container_width=True):
                    reset_cancel()
                    status_ph = st.empty()
                    with status_ph.container():
                        st.info(f"⏳ {__('generating')}")
                        render_cancel_button(__('generate_script'))
                    try:
                        sp = st.session_state.script_provider
                        model_name = _resolve_model(sp, "script")
                        script_key = _resolve_qwen_key() if sp == "qwen" else _api_key(f"script_{sp}")
                        gen = ScriptGenerator(
                            provider=sp,
                            api_key=script_key,
                            model=model_name,
                            base_url=LLM_PROVIDERS.get(sp, {}).get("base_url", ""),
                            mode=st.session_state.get("video_mode", "donghua"),
                        )
                        script = gen.generate(
                            subtitle_text=st.session_state.subtitle_text,
                            word_count=st.session_state.word_count,
                            video_title=st.session_state.video_title,
                            temperature=0.7,
                        )
                        st.session_state.script = script
                        st.session_state.script_json = json.dumps({"script": script}, ensure_ascii=False, indent=2)
                        est = gen.estimate_word_count(script)
                        auto_save()
                        reset_cancel()
                        status_ph.empty()
                        st.success(f"✅ {__('script_generated', words=est)}")
                        st.rerun()
                    except TaskCancelledError:
                        status_ph.empty()
                        st.warning("⏹️ 解说生成已取消")
                    except Exception as e:
                        status_ph.empty()
                        st.error(f"❌ {__('script_failed')}: {e}")
    else:
        gen = ScriptGenerator(provider="gemini")
        combined = gen.generate_combined_text(st.session_state.script)
        st.info(f"📊 {__('total_words')}: {len(combined)} | {__('segment_count')}: {len(st.session_state.script)}")

        sec_map = {__('section_opening'): "开头", __('section_body'): "内容", __('section_closing'): "结尾"}
        sec_emoji = {"开头": "🔴", "内容": "🟡", "结尾": "🔵"}
        sec_order = [__('section_opening'), __('section_body'), __('section_closing')]

        for i, seg in enumerate(st.session_state.script):
            section = seg.get("section", "内容")
            emoji = sec_emoji.get(section, "⚪")

            # Check for timing data (from scene-by-scene narration)
            timing_info = ""
            timing_ok = seg.get("timing_ok", None)
            if timing_ok is not None:
                est_dur = seg.get("estimated_duration", 0)
                scene_dur = seg.get("scene_duration", 0)
                scene_start = seg.get("scene_start", 0)
                scene_end = seg.get("scene_end", 0)
                ratio = seg.get("ratio", 0)
                if timing_ok:
                    timing_info = f" ✅ 时长匹配 | 朗读{est_dur:.1f}s / 场景{scene_dur:.1f}s (ratio={ratio:.2f})"
                elif ratio > 1.3:
                    timing_info = f" ⚠️ 朗读过长 | 朗读{est_dur:.1f}s > 场景{scene_dur:.1f}s | 建议缩短或慢放"
                else:
                    timing_info = f" ℹ️ 朗读{est_dur:.1f}s / 场景{scene_dur:.1f}s (ratio={ratio:.2f})"
                timing_info += f" | 时间段: {format_time(scene_start)} → {format_time(scene_end)}"

            with st.expander(f"{emoji} {__('highlight_segment', i=i+1)} ({section}){timing_info}", expanded=True):
                new_narration = st.text_area(
                    __('narration_label'),
                    seg.get("narration", ""),
                    height=150,
                    key=f"script_nar_{i}",
                )
                new_keywords = st.text_input(
                    __('keywords_label'),
                    ", ".join(seg.get("keywords", [])),
                    key=f"script_kw_{i}",
                )
                new_section = st.selectbox(
                    __('section_type'),
                    sec_order,
                    index=sec_order.index(__('section_body')) if section == "内容" else (
                        sec_order.index(__('section_opening')) if section == "开头" else sec_order.index(__('section_closing'))
                    ),
                    key=f"script_sec_{i}",
                    format_func=lambda x: x,
                )
                # Map localized section back to Chinese
                section_cn = sec_map.get(new_section, "内容")
                st.session_state.script[i] = {
                    "narration": new_narration,
                    "keywords": [k.strip() for k in new_keywords.split(",") if k.strip()],
                    "section": section_cn,
                }

        with st.expander(f"➕ {__('add_paragraph')}"):
            new_nar = st.text_area(__('narration_label'), height=100, key="new_script_nar")
            new_kw = st.text_input(__('keywords_label'), key="new_script_kw")
            new_sec = st.selectbox(__('section_type'), sec_order, key="new_script_sec", format_func=lambda x: x)
            if st.button(f"{__('add_paragraph_btn')}", use_container_width=True):
                st.session_state.script.append({
                    "narration": new_nar,
                    "keywords": [k.strip() for k in new_kw.split(",") if k.strip()],
                    "section": sec_map.get(new_sec, "内容"),
                })
                st.rerun()

        if st.button(f"💾 {__('save_script')}", type="primary", use_container_width=True):
            st.session_state.script_json = json.dumps({"script": st.session_state.script}, ensure_ascii=False, indent=2)
            script_path = get_project_dir() / "output" / "script.json"
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(st.session_state.script_json)
            auto_save()
            st.success(f"✅ {__('script_saved')}")

        if st.button(f"🔄 {__('regenerate')}", use_container_width=True):
            st.session_state.script = []
            st.rerun()

    step_nav("script", next_disabled=not st.session_state.script, position="bottom")


# ============================================================
# Step Page: Audio
# ============================================================
def render_audio_page():
    st.title(f"🔊 {__('audio_title')}")
    st.markdown(__('audio_desc'))

    step_nav("audio", next_disabled=not st.session_state.narration_audio)

    if not st.session_state.script:
        st.warning(f"⚠️ {__('no_script_warning')}")
        return

    if not st.session_state.narration_audio:
        gen = ScriptGenerator(provider="gemini")
        combined = gen.generate_combined_text(st.session_state.script)

        st.info(f"📝 {__('preview_text')}: ~{len(combined)} chars")
        st.text_area(__('preview_text'), combined[:1000] + ("..." if len(combined) > 1000 else ""), height=150, disabled=True)

        lang_code = SUPPORTED_LANGUAGES[st.session_state.tts_language]
        st.info(f"🔊 {__('audio_language')}: {st.session_state.tts_language}")

        if st.button(f"🔊 {__('generate_audio')}", type="primary", use_container_width=True):
            reset_cancel()
            status_ph = st.empty()
            with status_ph.container():
                st.info(f"⏳ {__('generating_audio')}")
                render_cancel_button(__('generate_audio'))
            try:
                tts = TTSEngine()
                audio_dir = get_project_dir() / "audio"
                speech_rate = st.session_state.get("speech_rate", -10)
                audio_path = tts.generate_full_script(
                    full_text=combined,
                    language=st.session_state.tts_language,
                    output_path=str(audio_dir / "full_narration.mp3"),
                    rate=f"{speech_rate:+.0f}%",
                )
                st.session_state.narration_audio = audio_path
                auto_save()
                reset_cancel()
                status_ph.empty()
                st.success(f"✅ {__('audio_generated')}")
                st.rerun()
            except TaskCancelledError:
                status_ph.empty()
                st.warning("⏹️ 配音生成已取消")
            except Exception as e:
                status_ph.empty()
                st.error(f"❌ {__('audio_failed')}: {e}")
    else:
        st.success(f"✅ {__('audio_generated')}")
        st.audio(st.session_state.narration_audio)

        with open(st.session_state.narration_audio, "rb") as f:
            st.download_button(f"📥 {__('download_mp3')}", f, file_name=f"{st.session_state.project_name}_audio.mp3", mime="audio/mpeg")

        if st.button(f"🔄 {__('regenerate_audio')}", use_container_width=True):
            st.session_state.narration_audio = None
            st.rerun()

    step_nav("audio", next_disabled=not st.session_state.narration_audio, position="bottom")


# ============================================================
# Step Page: Export
# ============================================================
def render_export_page():
    st.title("🎬 步骤 7/7：导出最终视频")
    st.markdown("合成所有素材，生成最终解说视频。")

    # Initialize effect toggles in session state
    if "enable_blur_mask" not in st.session_state:
        st.session_state.enable_blur_mask = False
    if "enable_mirror_reflect" not in st.session_state:
        st.session_state.enable_mirror_reflect = False

    project_dir = get_project_dir()

    # Summary of all steps
    with st.expander("📋 项目概览", expanded=True):
        scene_count = len(st.session_state.get("scenes", []))
        scene_info = f"{scene_count} 个场景" if scene_count else "❌ 未检测"
        visual_info = "✅ 已分析" if st.session_state.get("scene_data_merged") else "❌ 未分析"
        st.markdown(f"""
        - **项目**: `{st.session_state.project_name}`
        - **视频**: {format_time(st.session_state.video_duration)}
        - **字幕**: {len(st.session_state.subtitles)} 条
        - **场景**: {scene_info}
        - **画面分析**: {visual_info}
        - **画面帧**: {len(st.session_state.frames)} 帧
        - **精彩片段**: {len(st.session_state.highlights)} 个
        - **脚本分段**: {len(st.session_state.script)} 段
        - **配音**: {'✅ 已生成' if st.session_state.narration_audio else '❌ 未生成'}
        """)

    # Visual effects options
    with st.expander("🎨 视频特效设置", expanded=True):
        st.markdown("##### 模糊遮罩")
        st.session_state.enable_blur_mask = st.toggle(
            "启用模糊遮罩",
            value=st.session_state.enable_blur_mask,
            help="在视频顶部和底部添加半透明模糊遮罩，营造电影感氛围",
        )
        if st.session_state.enable_blur_mask:
            st.caption("效果：顶部和底部添加黑色半透明渐变 + 高斯模糊边缘")

        st.markdown("---")
        st.markdown("##### 镜像反射")
        st.session_state.enable_mirror_reflect = st.toggle(
            "启用镜像反射",
            value=st.session_state.enable_mirror_reflect,
            help="交替画面产生镜像反射效果：前一画面正常，后一画面分割为左原图+右镜像",
        )
        if st.session_state.enable_mirror_reflect:
            st.caption("效果：每隔一个画面，左侧保持原样，右侧生成镜像反射（分裂屏效果）")

    if not st.session_state.final_video:
        all_ready = (
            st.session_state.video_path
            and st.session_state.subtitles
            and st.session_state.script
            and st.session_state.narration_audio
        )
        if not all_ready:
            missing = []
            if not st.session_state.video_path: missing.append(__('video_duration'))
            if not st.session_state.subtitles: missing.append(__('subtitles_count'))
            if not st.session_state.script: missing.append(__('script_segments'))
            if not st.session_state.narration_audio: missing.append(__('audio_title'))
            st.warning(f"⚠️ {__('missing_content', items=', '.join(missing))}")
        else:
            effects_on = []
            if st.session_state.enable_blur_mask: effects_on.append(__('blur_mask'))
            if st.session_state.enable_mirror_reflect: effects_on.append(__('mirror_reflect'))
            if effects_on:
                st.info(f"✨ {__('effects_enabled', effects=', '.join(effects_on))}")
            else:
                st.info(f"ℹ️ {__('no_effects')}")

            if st.button(f"🎬 {__('compose_video')}", type="primary", use_container_width=True):
                reset_cancel()
                status_ph = st.empty()
                with status_ph.container():
                    st.info(f"⏳ {__('composing')}")
                    render_cancel_button(__('compose_video'))
                try:
                    editor = VideoEditor()
                    output_path = str(project_dir / "output" / "final_commentary.mp4")
                    lang_code = SUPPORTED_LANGUAGES[st.session_state.tts_language]
                    final = editor.compose_final_video(
                        original_video=st.session_state.video_path,
                        narration_audio=st.session_state.narration_audio,
                        script_segments=st.session_state.script,
                        subtitle_data=st.session_state.subtitles,
                        output_path=output_path,
                        highlight_clips=st.session_state.highlights,
                        language=lang_code,
                        enable_blur_mask=st.session_state.enable_blur_mask,
                        enable_mirror_reflect=st.session_state.enable_mirror_reflect,
                        speech_rate=st.session_state.get("speech_rate", -10),
                    )
                    st.session_state.final_video = final
                    auto_save()
                    reset_cancel()
                    status_ph.empty()
                    st.success(f"🎉 {__('video_composed')}")
                    st.rerun()
                except TaskCancelledError:
                    status_ph.empty()
                    st.warning("⏹️ 视频合成已取消")
                except Exception as e:
                    status_ph.empty()
                    st.error(f"❌ {__('compose_failed')}: {e}")
                    st.code(traceback.format_exc())
    else:
        st.success(f"🎉 {__('video_ready')}")
        st.video(st.session_state.final_video)

        with open(st.session_state.final_video, "rb") as f:
            st.download_button(
                f"📥 {__('download_video')}",
                f,
                file_name=f"{st.session_state.project_name}_video.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            if st.button(f"🔄 {__('recompose')}", use_container_width=True):
                st.session_state.final_video = None
                st.rerun()
        with c2:
            if st.button(f"🏠 {__('new_project')}", use_container_width=True):
                auto_save()  # Save current project before clearing
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                init_session_state()
                st.rerun()

    step_nav("export", next_disabled=True)


# ============================================================
# Page: Project Files (shared component)
# ============================================================
def render_project_files_tab():
    project_dir = get_project_dir()
    lang = st.session_state.get("ui_language", "zh")
    from locales import _ as _t
    st.markdown(f"**{_t('project', lang)}:** `{project_dir}`")

    all_files = []
    for root, dirs, files in os.walk(project_dir):
        for file in files:
            fp = Path(root) / file
            sz = fp.stat().st_size
            all_files.append((str(fp.relative_to(project_dir)), sz))

    if all_files:
        for rel_path, size in all_files:
            sz_str = f"{size/1024:.1f} KB" if size > 1024 else f"{size} B"
            st.text(f"📄 {rel_path} ({sz_str})")

    import zipfile
    zip_path = project_dir / f"{st.session_state.project_name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(project_dir):
            for file in files:
                fp = Path(root) / file
                zf.write(fp, fp.relative_to(project_dir))
    with open(zip_path, "rb") as f:
        st.download_button(f"📥 {_t('download_video', lang)} (ZIP)", f, file_name=f"{st.session_state.project_name}.zip", mime="application/zip", use_container_width=True)


# ============================================================
# Main App Router
# ============================================================
def render_sidebar():
    """Render the sidebar with step navigation and language selector."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/video-editing.png", width=48)
        st.markdown(f"## {__('app_title')}")
        st.markdown("---")

        # Language selector
        lang_opts = {"中文": "zh", "English": "en", "Tiếng Việt": "vi"}
        current_lang = st.session_state.get("ui_language", "zh")
        lang_label = {v: k for k, v in lang_opts.items()}[current_lang]
        selected_lang = st.selectbox(
            "🌐 Language / Ngôn ngữ",
            options=list(lang_opts.keys()),
            index=list(lang_opts.values()).index(current_lang),
            key="ui_language_selector",
            label_visibility="collapsed",
        )
        new_lang = lang_opts[selected_lang]
        if new_lang != current_lang:
            st.session_state.ui_language = new_lang
            st.rerun()

        st.markdown(f"**{__('project')}:** `{st.session_state.project_name[:16]}...`" if len(st.session_state.project_name) > 16 else f"**{__('project')}:** `{st.session_state.project_name}`")

        if st.session_state.video_path:
            st.markdown(f"✅ {__('video_status')} ({format_time(st.session_state.video_duration)})")

        st.markdown("---")
        st.markdown(f"**{__('processing_flow')}:**")

        current = st.session_state.current_step
        # Build step labels from locale
        step_key_map = {
            "upload": "step_upload",
            "subtitles": "step_subtitles",
            "scenes": "step_scenes",
            "visual": "step_visual",
            "script": "step_script",
            "audio": "step_audio",
            "export": "step_export",
        }
        for step_id, emoji, _ in STEPS:
            is_current = step_id == current
            localized_label = __(step_key_map.get(step_id, f"step_{step_id}"))
            if step_id == "upload":
                status = "curr" if is_current else "done" if st.session_state.video_path else ""
            elif step_id == "subtitles":
                status = "curr" if is_current else "done" if st.session_state.subtitles else ""
            elif step_id == "scenes":
                status = "curr" if is_current else "done" if st.session_state.scenes else ""
            elif step_id == "visual":
                status = "curr" if is_current else "done" if st.session_state.scene_data_merged else ""
            elif step_id == "script":
                status = "curr" if is_current else "done" if st.session_state.script else ""
            elif step_id == "audio":
                status = "curr" if is_current else "done" if st.session_state.narration_audio else ""
            elif step_id == "export":
                status = "curr" if is_current else "done" if st.session_state.final_video else ""

            if status == "curr":
                st.markdown(f"> **{emoji} {localized_label}** 👈")
            elif status == "done":
                st.markdown(f"  ~~{emoji} {localized_label}~~ ✅")
            else:
                st.markdown(f"  {emoji} {localized_label}")

        st.markdown("---")
        st.markdown(f"**💾 {__('project_save')}:**")

        # Save button
        if st.button(f"💾 {__('save_project')}", use_container_width=True):
            save_project_snapshot()
            st.success(f"✅ {__('project_saved')}")
            st.rerun()

        # List saved projects
        saved_projects = get_saved_projects()
        if saved_projects:
            proj_names = [p["name"] for p in saved_projects]
            proj_labels = []
            for p in saved_projects:
                status_icons = ""
                if p.get("has_video"):
                    status_icons += "🎬"
                elif p.get("has_audio"):
                    status_icons += "🔊"
                elif p.get("has_script"):
                    status_icons += "✍️"
                label = f"{p['name'][:12]}… {status_icons} ({p['step_label']})"
                proj_labels.append(label)

            selected_label = st.selectbox(
                __("load_project"),
                options=proj_labels,
                label_visibility="collapsed",
                key="load_project_selector",
            )
            if selected_label:
                selected_idx = proj_labels.index(selected_label)
                selected_name = proj_names[selected_idx]

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"📂 {__('load_btn')}", use_container_width=True, key=f"load_{selected_name}"):
                        if load_project_snapshot(selected_name):
                            st.success(f"✅ {__('project_loaded')}: {selected_name}")
                            st.rerun()
                        else:
                            st.error(f"❌ {__('load_failed')}")
                with col2:
                    if st.button(f"🗑️ {__('delete_btn')}", use_container_width=True, key=f"del_{selected_name}"):
                        delete_saved_project(selected_name)
                        st.rerun()

        st.markdown("---")
        st.markdown(f"**{__('supported_engines')}:**")
        st.caption("🤖 OpenAI / Gemini / Qwen / DeepSeek")
        st.caption("🎤 Faster-Whisper / 🔊 Edge TTS")

        st.markdown("---")
        st.caption("🌐 " + ", ".join(SUPPORTED_LANGUAGES.keys()))


def _cleanup_widget_keys():
    """Remove any stale Streamlit widget keys from session state."""
    keys_to_del = [k for k in st.session_state.keys() if k.startswith(WIDGET_KEY_PREFIXES)]
    for k in keys_to_del:
        del st.session_state[k]


def auto_restore_last_session():
    """On fresh startup, offer to restore the most recent project."""
    # Only run on first load (no project data yet, and not already restored)
    if st.session_state.get("_restore_checked"):
        return
    st.session_state._restore_checked = True

    # Only auto-restore if we're on a fresh state (no video, no subtitles)
    if st.session_state.video_path or st.session_state.subtitles:
        return

    saved = get_saved_projects()
    if not saved:
        return

    # Sort by saved_at descending, pick most recent
    saved.sort(key=lambda p: p.get("saved_at", ""), reverse=True)
    latest = saved[0]
    name = latest["name"]

    # Check if the snapshot file still exists
    snap_path = Path("temp") / name / "project_snapshot.json"
    if not snap_path.exists():
        return

    st.toast(f"💾 发现上次保存的项目「{name}」，已自动恢复", icon="🔄")
    if load_project_snapshot(name):
        # Restore the step
        target_step = st.session_state.get("current_step", "upload")
        st.session_state.current_step = target_step


def main():
    auto_restore_last_session()
    render_sidebar()

    step_map = {
        "upload": render_upload_page,
        "subtitles": render_subtitles_page,
        "scenes": render_scenes_page,
        "visual": render_visual_page,
        "script": render_script_page,
        "audio": render_audio_page,
        "export": render_export_page,
    }

    page = st.session_state.current_step
    if page in step_map:
        step_map[page]()
    else:
        render_upload_page()


if __name__ == "__main__":
    main()

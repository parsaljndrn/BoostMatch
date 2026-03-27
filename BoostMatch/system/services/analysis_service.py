import re
import subprocess
import os
from typing import Optional
from pathlib import Path
import joblib
import pandas as pd
from .matcher import check_misleading as matcher_check_misleading
from .article_tools import extract_article_for_nlp
from .fb_graph import normalize_language, is_meaningful_text, clean_caption_text
from services.article_tools import extract_article_headline
from urllib.parse import urlparse
import tempfile
import requests
import subprocess
import unicodedata
import sys
# Current file is in system/services/
THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one directory to system/
BASE_DIR = os.path.dirname(THIS_FILE_DIR)


if os.name == "nt":  # Windows
    FFMPEG_BIN = os.path.join(BASE_DIR, "ffmpeg", "ffmpeg.exe")
    FFPROBE_BIN = os.path.join(BASE_DIR, "ffmpeg", "ffprobe.exe")
else:  # Linux (Railway)
    # In Railway, the folder structure is: /app/BoostMatch/system/ffmpeg/ffmpeg
    FFMPEG_BIN = os.path.join(BASE_DIR, "ffmpeg", "ffmpeg")
    FFPROBE_BIN = os.path.join(BASE_DIR, "ffmpeg", "ffprobe")

# Optional: debug print
print("FFMPEG_BIN:", FFMPEG_BIN)
print("FFPROBE_BIN:", FFPROBE_BIN)

def normalize_text(text: str) -> str:
    # Normalize Unicode to NFC
    text = unicodedata.normalize("NFC", text)
    # Remove zero-width spaces
    text = text.replace("\u200B", "")
    # Replace non-breaking spaces with normal spaces
    text = text.replace("\u00A0", " ")
    # Strip leading/trailing whitespace
    text = text.strip()
    return text

MAX_VIDEO_DURATION = 7 * 60  # 6 minutes in seconds

# --------------------------------------
# Helper: get video duration using ffprobe
# --------------------------------------
def get_video_duration(video_path: str) -> float:
    """Return video duration in seconds using ffprobe."""
    if not os.path.exists(FFPROBE_BIN):
        raise FileNotFoundError(f"ffprobe binary not found at {FFPROBE_BIN}")

    result = subprocess.run(
        [
            FFPROBE_BIN,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    try:
        duration = float(result.stdout.strip())
    except ValueError:
        raise ValueError("Could not read video duration. Unsupported format or empty file.")

    return duration

def check_video_duration(video_path: str):
    duration = get_video_duration(video_path)
    if duration > MAX_VIDEO_DURATION:
        raise ValueError("Video is longer than 7 minutes. Cannot process.")

def is_facebook_url(url: str) -> bool:
    if not url:
        return False

    return any(domain in url for domain in [
        "facebook.com",
        "fb.watch",
        "m.facebook.com"
    ])

# --------------------------------------
# Download and extract audio for transcription
# --------------------------------------
def extract_audio_from_video(video_url: str) -> str:
    """
    Download a Facebook video using yt-dlp and extract audio as MP3 using FFmpeg.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "video.mp4")
        audio_path = os.path.join(tmpdir, "audio.mp3")

        # Download actual video
        subprocess.run(
            [
                "yt-dlp",
                "-f", "best[ext=mp4]/best",
                "-o", video_path,
                video_url
            ],
            check=True
        )
        if not os.path.exists(video_path) or os.path.getsize(video_path) < 1000:  # less than 1 KB
            raise ValueError(
                f"Downloaded video is too small or empty ({os.path.getsize(video_path) if os.path.exists(video_path) else 0} bytes). Likely failed download."
            )

        # Extract audio
        subprocess.run(
            [FFMPEG_BIN, "-i", video_path, "-vn", "-ar", "16000", "-ac", "1", audio_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            raise ValueError("Audio extraction failed. Cannot process.")

        return audio_path

# -----------------------------
# WhisperAI setup (using openai.whisper locally)
# -----------------------------
# Set environment for Faster-Whisper to use your local FFmpeg
os.environ["WHISPER_FFMPEG"] = "/app/BoostMatch/system/ffmpeg/ffmpeg"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from faster_whisper import WhisperModel

# Lazy-load the model
whisper_model = None

def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        whisper_model = WhisperModel("tiny", device="cpu")
    return whisper_model


# -----------------------------
# Video transcription helper
# -----------------------------
def transcribe_video(video_url: str) -> str:
    if not video_url:
        return ""

    domain = urlparse(video_url).netloc.lower()
    if not any(d in domain for d in ["facebook.com", "fb.watch", "fbcdn.net"]):
        raise ValueError("Only Facebook videos can be transcribed.")

    audio_path = extract_audio_from_video(video_url)

    try:
        # Transcribe
        model = get_whisper_model()
        segments, _ = model.transcribe(audio_path, task="translate")
        text = " ".join(seg.text for seg in segments).strip()

        if not text:
            raise ValueError("Transcription failed or returned empty text.")

        return text

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

# -----------------------------
# Prepare caption/article for analysis
# -----------------------------
def prepare_post_for_analysis(
    caption: str = "", article_link: Optional[str] = None, video_url: Optional[str] = None
):
    """
    Prepare caption and article content based on post type.
    Returns (caption_to_use, article_text_to_use)
    Raises ValueError if post cannot be analyzed.
    """
    caption = caption or ""
    article_link = article_link or ""
    video_url = video_url or ""

    # Case 1: caption + article link (ignore video)
    if caption and article_link:
        article_text = extract_article_headline(article_link)
        return caption, article_text

    # Case 2: caption + video (no article link)
    if caption and video_url and not article_link:
        domain = urlparse(video_url).netloc.lower()

        allowed_domains = ["facebook.com", "fb.watch", "fbcdn.net"]

        if not any(domain.endswith(d) for d in allowed_domains):
            raise ValueError(
                "Invalid video source: Only Facebook videos are supported. "
                "Please provide a Facebook post with an article link instead."
            )

        # Only Facebook videos reach here
        video_text = transcribe_video(video_url)
        return caption, video_text

    # Case 3: video + article link (no caption)
    if video_url and article_link and not caption:
        video_text = transcribe_video(video_url)
        article_text = extract_article_headline(article_link)
        return  video_text, article_text

    # Case 4: video only (no caption, no link)
    if video_url and not caption and not article_link:
        raise ValueError("Cannot analyze post: video-only posts are unsupported.")

    # Case 5: caption only (no article link)
    if caption and not article_link:
        raise ValueError(
            "Cannot analyze post: no article link attached. "
            "Please paste a Facebook post with an article attached."
        )

    # Case 6: article link only (no caption)
    if article_link and not caption:
        raise ValueError(
            "Cannot analyze post: no caption provided. "
            "Please paste a Facebook post with a caption"
        )

    # Any other case
    raise ValueError("Cannot analyze post: insufficient data.")

# -----------------------------
# High-level wrapper for Flask
# -----------------------------
def classify_post(caption: str = "", article_link: str = None, video_url: str = None):
    """
    High-level function to classify a Facebook post.
    Returns a dict with prediction, cosine similarity, and text used.
    """

    print("\n===== RAW INPUT DEBUG =====")
    print("RAW CAPTION:", repr(caption[:200]))
    print("===========================\n")

    MAX_TRANSLATE_CHARS = 6000

    caption = clean_caption_text(caption)

    # Step 4: Check maximum length
    if len(caption) > MAX_TRANSLATE_CHARS:
        raise ValueError(
            f"Cannot analyze input: caption too long ({len(caption)} characters). "
            f"Please enter a shorter caption."
        )

    # ✅ Step 3: Normalize language (translation if needed)
    caption_to_use, lang_detected = normalize_language(caption)

    if len(caption_to_use) > MAX_TRANSLATE_CHARS:
        raise ValueError(
            f"Cannot analyze post: caption too long ({len(caption_to_use)} chars). "
            f"Please try a post with shorter caption."
        )

    # ✅ Step 4: Prepare article/video text
    caption_to_use, article_to_use = prepare_post_for_analysis(
        caption_to_use, article_link, video_url
    )

    # ✅ Step 5: Run BoostMatch prediction
    cos_sim, prediction = matcher_check_misleading(caption_to_use, article_to_use)

    return {
        "prediction": prediction,
        "cosine_similarity": cos_sim,
        "caption_used": caption_to_use,
        "article_used": article_to_use
    }
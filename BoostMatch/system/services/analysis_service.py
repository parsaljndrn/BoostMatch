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


def is_facebook_url(url: str) -> bool:
    if not url:
        return False

    return any(domain in url for domain in [
        "facebook.com",
        "fb.watch",
        "m.facebook.com"
    ])

# -----------------------------
# WhisperAI setup (using openai.whisper locally)
# -----------------------------
try:
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel("small", device="cpu")
except ImportError:
    whisper_model = None
    print("[Warning] Faster-Whisper not installed.")


# -----------------------------
# Video transcription helper
# -----------------------------
def transcribe_video(video_url: str) -> str:
    if not video_url:
        return ""

    if whisper_model is None:
        raise ValueError("Faster-Whisper not installed.")

    fd, video_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    audio_path = video_path.replace(".mp4", ".mp3")

    try:
        # Download video
        r = requests.get(video_url, stream=True, timeout=30)
        r.raise_for_status()

        with open(video_path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)

        # Extract audio
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn", "-acodec", "mp3", audio_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # ✅ Correct Faster-Whisper usage
        segments, _ = whisper_model.transcribe(audio_path, task="translate")

        video_text = " ".join([seg.text for seg in segments]).strip()

        if not video_text:
            raise ValueError("Transcription failed or empty.")

        return video_text

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
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
        if any(x in domain for x in ["facebook.com", "fb.watch", "fbcdn.net"]):
            # Facebook video → transcribe
            video_text = transcribe_video(video_url)  # allow exceptions to propagate
            return caption, video_text
        else:
            # Non-Facebook videos cannot be transcribed → fallback
            raise ValueError(
                "This Facebook post does not contain a valid article link. "
                "Please paste a Facebook post with an article attached."
            )


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
    MAX_TRANSLATE_CHARS = 6000

    # ✅ Step 1: Clean caption FIRST
    caption = clean_caption_text(caption)

    """# ✅ Step 2: Validate meaningful text
    if not is_meaningful_text(caption):
        raise ValueError(
            "Cannot analyze post: no caption provided. "
            "Please paste a Facebook post with a caption.6000")"""

    # ✅ Step 3: Normalize language (translation if needed)
    caption_to_use, lang_detected = normalize_language(caption)

    if len(caption_to_use) > MAX_TRANSLATE_CHARS:
        raise ValueError(
            f"Cannot analyze post: caption too long ({len(caption_to_use)} chars). "
            f"Please try a post with shorten the caption and try again."
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
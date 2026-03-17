import re
import subprocess
import os
from typing import Optional
from pathlib import Path
import joblib
import pandas as pd
from .matcher import check_misleading as matcher_check_misleading
from .article_tools import extract_article_for_nlp
from .fb_graph import normalize_language
from services.article_tools import extract_article_headline
from urllib.parse import urlparse
import tempfile
import requests

# -----------------------------
# WhisperAI setup (using openai.whisper locally)
# -----------------------------
try:
    import whisper
except ImportError:
    whisper = None
    print("[Warning] Whisper not installed. Video transcription will not work.")


# -----------------------------
# Video transcription helper
# -----------------------------
def transcribe_video(video_url: str) -> str:
    """
    Transcribe a video URL using Whisper (Windows-safe version).
    """
    if not video_url:
        return ""

    if whisper is None:
        raise ValueError("Whisper not installed. Cannot transcribe video.")

    # Step 1: create temp file path WITHOUT opening it
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)  # close immediately so Whisper/FFmpeg can open it

    try:
        # Download video
        r = requests.get(video_url, stream=True, timeout=30)
        r.raise_for_status()

        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # ---- Check if video has audio ----
        check_audio = subprocess.run(
            ["ffprobe", "-i", tmp_path, "-show_streams", "-select_streams", "a", "-loglevel", "error"],
            capture_output=True,
            text=True
        )

        if check_audio.stdout.strip() == "":
            raise ValueError("This Facebook video contains no audio and cannot be transcribed.")

        # ---- Transcribe ----
        model = whisper.load_model("small")
        result = model.transcribe(tmp_path)

        return result.get("text", "")

    finally:
        # Step 4: clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


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
        article_text = extract_article_for_nlp(article_link)
        return video_text, article_text

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
            "Please paste a Facebook post with a caption."
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
 
    # Normalize language: translate if not English
    caption_to_use, lang_detected = normalize_language(caption)
    # caption_to_use is always defined, translation happens if needed

    # Prepare article/video text
    caption_to_use, article_to_use = prepare_post_for_analysis(
        caption_to_use, article_link, video_url
    )

    # Run BoostMatch prediction
    cos_sim, prediction = matcher_check_misleading(caption_to_use, article_to_use)

    return {
        "prediction": prediction,
        "cosine_similarity": cos_sim,
        "caption_used": caption_to_use,
        "article_used": article_to_use
    }
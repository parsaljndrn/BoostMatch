import os
import re
import sys
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path
from threading import Timer
sys.path.insert(0, str(Path(__file__).resolve().parent))
import whisper
import yt_dlp
from dotenv import load_dotenv
from flask import Flask, render_template, request
from langdetect import detect as lang_detect
from services.fb_graph import extract_post_id, fetch_facebook_post
from services.article_tools import get_article_content
from services.matcher import check_misleading
load_dotenv()

app = Flask(__name__)

try:
    whisper_model = whisper.load_model("small")
    print("[app] Whisper model loaded.")
except Exception as e:
    whisper_model = None
    print(f"[app] Whisper model loading failed: {e}")


def _is_facebook_url(url: str) -> bool:
    return bool(re.search(r"(?:facebook\.com|fb\.watch|fb\.com)", url, re.I))


def _classify_post(fb_url: str, post_data: dict) -> str:
    url_lower = fb_url.lower()
    is_video_url = any(x in url_lower for x in ["reel", "video", "watch"])
    has_video_src = bool(post_data.get("video_url"))
    has_link = bool(post_data.get("article_link"))
    has_caption = bool(post_data.get("caption", "").strip())
    is_video = is_video_url or has_video_src

    if is_video and has_link:
        return "video_with_link"
    if is_video and has_caption:
        return "video_with_caption"
    if has_caption and has_link:
        return "caption_with_link"
    if has_caption:
        return "caption_only"
    return "unknown"


def _is_english(text: str) -> bool:
    try:
        return lang_detect(text) == "en"
    except Exception:
        return True


def _translate_to_english(text: str) -> str:
    if not text or not text.strip():
        return text
    try:
        from deep_translator import GoogleTranslator
        chunk_size = 4500
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        translated_chunks = []
        for chunk in chunks:
            translated = GoogleTranslator(source="auto", target="en").translate(chunk)
            translated_chunks.append(translated or chunk)
        return " ".join(translated_chunks)
    except Exception as e:
        print(f"[app] Translation failed: {e}")
        return text


def _ensure_english(text: str) -> str:
    if not text or not text.strip():
        return text
    try:
        if lang_detect(text) == "en":
            return text
    except Exception:
        return text
    print(f"[app] Translating non-English text...")
    return _translate_to_english(text)


def _transcribe_audio(video_source: str) -> tuple[str, str]:
    if not whisper_model:
        return "", "[Whisper model not available]"

    uid = uuid.uuid4().hex[:10]
    audio_path = f"temp_audio_{uid}.mp3"

    try:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": f"temp_audio_{uid}",
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_source])

        if not os.path.exists(audio_path):
            return "", "[Audio file not found after extraction]"

        res_tl = whisper_model.transcribe(audio_path, language="tl")
        res_en = whisper_model.transcribe(audio_path, task="translate")

        return (
            (res_tl.get("text") or "").strip(),
            (res_en.get("text") or "").strip(),
        )

    except Exception as e:
        return "", f"[Transcription error: {e}]"

    finally:
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass


def _preprocess_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


LINK_TYPE_NOTES = {
    "shop":        "The attached link points to a shopping/e-commerce page. No article content to analyze.",
    "social":      "The attached link points to a social media page. No article content to analyze.",
    "video":       "The attached link points to a video platform. No article content to analyze.",
    "unknown":     "The attached link did not return enough readable content to analyze.",
    "ad_redirect": "LINK REDIRECTS TO AN AD",
    "not_article": "CANNOT ANALYZE: LINK ATTACHMENT IS NOT AN ARTICLE",
}


def _run_analysis(fb_url: str) -> dict:
    if not _is_facebook_url(fb_url):
        raise ValueError("Invalid Link: URL does not appear to be a Facebook link.")

    post_id, id_type = extract_post_id(fb_url)
    if not post_id:
        raise ValueError("Could not extract a valid Facebook post ID from the URL.")

    post_data = fetch_facebook_post(post_id, id_type)
    post_type = _classify_post(fb_url, post_data)

    if post_type == "unknown":
        raise ValueError("Cannot detect post type. No caption or link found in this post.")

    caption_raw      = post_data.get("caption", "")
    article_link     = post_data.get("article_link", "")
    video_url        = post_data.get("video_url", "")
    transcription_tl = ""
    transcription_en = ""

    if post_type in ("video_with_link", "video_with_caption"):
        src = video_url or fb_url
        transcription_tl, transcription_en = _transcribe_audio(src)
        transcription_en = _ensure_english(transcription_en or transcription_tl)
        caption_en = _ensure_english(caption_raw)

        if post_type == "video_with_link":
            article_text, link_type = get_article_content(article_link)

            if link_type in LINK_TYPE_NOTES:
                return {
                    "post_type": post_type,
                    "prediction": "N/A",
                    "similarity": None,
                    "caption": caption_raw or "[No caption]",
                    "transcription_tl": transcription_tl or "",
                    "transcription_en": transcription_en or "",
                    "article_link": article_link,
                    "article_text": "",
                    "has_transcript": True,
                    "link_type": link_type,
                    "analysis_note": LINK_TYPE_NOTES[link_type],
                }

            article_text = _ensure_english(article_text)
            text_to_check = _preprocess_text(" ".join(filter(None, [caption_en, transcription_en])))

        else:
            return {
                "post_type": post_type,
                "prediction": "N/A",
                "similarity": None,
                "caption": caption_raw or "[No caption]",
                "transcription_tl": transcription_tl or "[No Tagalog transcription]",
                "transcription_en": transcription_en or "[No English translation]",
                "article_link": "",
                "article_text": "",
                "has_transcript": True,
                "link_type": "none",
                "analysis_note": "This video post has no article link attached. No misleading-content score can be computed without a reference article.",
            }

    elif post_type == "caption_with_link":
        caption_en = _ensure_english(caption_raw)
        article_text, link_type = get_article_content(article_link)

        if link_type in LINK_TYPE_NOTES:
            return {
                "post_type": post_type,
                "prediction": "N/A",
                "similarity": None,
                "caption": caption_raw or "[No caption]",
                "transcription_tl": "",
                "transcription_en": "",
                "article_link": article_link,
                "article_text": "",
                "has_transcript": False,
                "link_type": link_type,
                "analysis_note": LINK_TYPE_NOTES[link_type],
            }

        article_text = _ensure_english(article_text)
        text_to_check = _preprocess_text(caption_en)

    elif post_type == "caption_only":
        caption_en = _preprocess_text(_ensure_english(caption_raw))
        return {
            "post_type": post_type,
            "prediction": "N/A",
            "similarity": None,
            "caption": caption_raw or "[No caption]",
            "transcription_tl": "",
            "transcription_en": "",
            "article_link": "",
            "article_text": "",
            "has_transcript": False,
            "link_type": "none",
            "analysis_note": "This post has no article link attached. No misleading-content score can be computed without a reference article.",
        }

    article_text_clean = _preprocess_text(article_text)
    similarity, prediction = check_misleading(text_to_check, article_text_clean)

    return {
        "post_type": post_type,
        "prediction": prediction.upper(),
        "similarity": round(similarity * 100, 1),
        "caption": caption_raw or "[No caption]",
        "transcription_tl": transcription_tl or "",
        "transcription_en": transcription_en or "",
        "article_link": article_link,
        "article_text": article_text[:6000] + ("…" if len(article_text) > 6000 else ""),
        "has_transcript": bool(transcription_tl or transcription_en),
        "link_type": "article",
        "analysis_note": "",
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        fb_url = request.form.get("fb_url", "").strip()
        if not fb_url:
            error = "Please paste a Facebook post URL."
        else:
            try:
                result = _run_analysis(fb_url)
            except Exception as e:
                now = datetime.now().strftime("%H:%M:%S")
                error = f"[{now}] {e}"

    return render_template("index.html", result=result, error=error)


def _open_browser():
    try:
        if not webbrowser.open_new("http://127.0.0.1:5000/"):
            print("->  http://127.0.0.1:5000/  (open manually)")
    except Exception:
        print("->  http://127.0.0.1:5000/  (open manually)")


if __name__ == "__main__":
    print("Starting BoostMatch Analyzer...")
    Timer(4.0, _open_browser).start()
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)

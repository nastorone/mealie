# ruff: noqa
import os
import re
import time
from typing import Any

import requests
import yt_dlp
from bs4 import BeautifulSoup


class ContentRetriever:
    def __init__(self, temp_dir: str = "/tmp/mealie_social_spike"):
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)
        self.mobile_headers = {
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1",
            "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://www.google.com/",
        }
        self.desktop_headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
        }

    def test_official_oembed(self, url: str, platform: str) -> dict[str, Any]:
        """Test Approach 1: Official preview endpoints."""
        t0 = time.time()
        result = {
            "method": "official_oembed",
            "success": False,
            "title": None,
            "author": None,
            "thumbnail_url": None,
            "has_description": False,
            "has_audio": False,
            "time_s": 0.0,
            "error": None,
        }
        try:
            if "instagram" in platform.lower():
                # Official Instagram oEmbed requires Meta App access_token
                # Calling without token demonstrates failure/requirement
                oembed_url = f"https://graph.facebook.com/v12.0/instagram_oembed?url={url}"
                resp = requests.get(oembed_url, timeout=10)
                result["time_s"] = round(time.time() - t0, 3)
                if resp.status_code == 200:
                    data = resp.json()
                    result["success"] = True
                    result["title"] = data.get("title")
                    result["thumbnail_url"] = data.get("thumbnail_url")
                    result["has_description"] = False
                    result["has_audio"] = False
                    result["error"] = "Restituisce solo widget iframe client-side, nessuna didascalia né traccia audio"
                else:
                    result["error"] = f"HTTP {resp.status_code}: Richiede Meta App ID e Client Token"
            elif "tiktok" in platform.lower():
                oembed_url = f"https://www.tiktok.com/oembed?url={url}"
                resp = requests.get(oembed_url, timeout=10)
                result["time_s"] = round(time.time() - t0, 3)
                if resp.status_code == 200:
                    data = resp.json()
                    result["success"] = True
                    result["title"] = data.get("title")
                    result["author"] = data.get("author_name")
                    result["thumbnail_url"] = data.get("thumbnail_url")
                    result["has_description"] = bool(data.get("title"))
                    result["has_audio"] = False
                    result["error"] = "Restituisce solo iframe e autore, titolo spesso nullo e nessun audio"
            elif "youtube" in platform.lower():
                oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
                resp = requests.get(oembed_url, timeout=10)
                result["time_s"] = round(time.time() - t0, 3)
                if resp.status_code == 200:
                    data = resp.json()
                    result["success"] = True
                    result["title"] = data.get("title")
                    result["author"] = data.get("author_name")
                    result["thumbnail_url"] = data.get("thumbnail_url")
                    result["has_description"] = False  # oEmbed does not return full description
                else:
                    result["error"] = f"HTTP {resp.status_code}"
        except Exception as e:
            result["time_s"] = round(time.time() - t0, 3)
            result["error"] = str(e)

        return result

    def test_metadata_scraping(self, url: str, platform: str) -> dict[str, Any]:
        """Test Approach 2: Public page HTML metadata scraping (OpenGraph, schema.org)."""
        t0 = time.time()
        result = {
            "method": "metadata_scraping",
            "success": False,
            "title": None,
            "description": None,
            "thumbnail_url": None,
            "time_s": 0.0,
            "error": None,
        }
        try:
            headers = self.mobile_headers if "tiktok" in platform.lower() else self.desktop_headers
            resp = requests.get(url, headers=headers, timeout=12)
            result["time_s"] = round(time.time() - t0, 3)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                og_title = soup.find("meta", property="og:title")
                og_desc = soup.find("meta", property="og:description")
                og_image = soup.find("meta", property="og:image")

                title = og_title.get("content") if og_title else None
                desc = og_desc.get("content") if og_desc else None
                image = og_image.get("content") if og_image else None

                # For TikTok with mobile headers, search JSON in HTML
                if "tiktok" in platform.lower() and not desc:
                    desc_matches = re.findall(r"\"desc\":\"(.*?)\"", resp.text)
                    if desc_matches:
                        desc = desc_matches[0].encode().decode("unicode_escape", errors="ignore")
                    cover_matches = re.findall(r"\"cover\":\"(.*?)\"", resp.text)
                    if cover_matches:
                        image = cover_matches[0].encode().decode("unicode_escape", errors="ignore")

                result["title"] = title
                result["description"] = desc
                result["thumbnail_url"] = image
                result["success"] = bool(desc or title)
                if not result["success"]:
                    result["error"] = "Nessun metadato utile o blocco login"
            else:
                result["error"] = f"HTTP {resp.status_code}"
        except Exception as e:
            result["time_s"] = round(time.time() - t0, 3)
            result["error"] = str(e)

        return result

    def retrieve_full_content(self, reel_id: int, url: str, platform: str) -> dict[str, Any]:
        """Test Approach 3: Complete retrieval using yt-dlp (audio + caption + images)."""
        t0 = time.time()
        result = {
            "reel_id": reel_id,
            "url": url,
            "platform": platform,
            "caption": None,
            "title": None,
            "audio_path": None,
            "thumbnail_url": None,
            "images": [],
            "retrieval_status": "FAILED",
            "method_used": "yt-dlp",
            "time_s": 0.0,
            "notes": "",
        }

        audio_output = os.path.join(self.temp_dir, f"reel_{reel_id}.mp3")

        # 1. Try yt-dlp first
        ydl_opts = {
            "format": "ba/b",
            "outtmpl": os.path.join(self.temp_dir, f"reel_{reel_id}.%(ext)s"),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "128",
                }
            ],
            "quiet": True,
            "no_warnings": True,
            "skip_download": False,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                result["title"] = info.get("title")
                result["caption"] = info.get("description")
                result["thumbnail_url"] = info.get("thumbnail")
                if os.path.exists(audio_output):
                    result["audio_path"] = audio_output

                # Check if it was a carousel or playlist
                entries = info.get("entries")
                if entries:
                    for i, entry in enumerate(entries):
                        thumb = entry.get("thumbnail") or (
                            entry.get("thumbnails", [{}])[-1].get("url") if entry.get("thumbnails") else None
                        )
                        if thumb:
                            result["images"].append(thumb)

                result["retrieval_status"] = "SUCCESS"
                result["method_used"] = "yt-dlp"
        except Exception as e:
            error_msg = str(e)
            result["notes"] = f"yt-dlp: {error_msg[:120]}"

            # Fallback for TikTok (which blocks yt-dlp desktop requests without login)
            if "tiktok" in platform.lower():
                try:
                    resp = requests.get(url, headers=self.mobile_headers, timeout=12)
                    if resp.status_code == 200:
                        desc_matches = re.findall(r"\"desc\":\"(.*?)\"", resp.text)
                        cover_matches = re.findall(r"\"cover\":\"(.*?)\"", resp.text)
                        music_matches = re.findall(r"\"playUrl\":\"(.*?)\"", resp.text)

                        if desc_matches:
                            result["caption"] = desc_matches[0].encode().decode("unicode_escape", errors="ignore")
                        if cover_matches:
                            result["thumbnail_url"] = (
                                cover_matches[0].encode().decode("unicode_escape", errors="ignore")
                            )

                        if music_matches:
                            m_url = music_matches[0].encode().decode("unicode_escape", errors="ignore")
                            mr = requests.get(m_url, headers=self.mobile_headers, timeout=15)
                            if mr.status_code == 200:
                                with open(audio_output, "wb") as f:
                                    f.write(mr.content)
                                if os.path.exists(audio_output):
                                    result["audio_path"] = audio_output

                        if result["caption"] or result["audio_path"]:
                            result["retrieval_status"] = (
                                "SUCCESS" if (result["caption"] and result["audio_path"]) else "PARTIAL"
                            )
                            result["method_used"] = "mobile_scrape_fallback"
                            result["notes"] += " | Recuperato con fallback mobile HTTP"
                except Exception as fb_err:
                    result["notes"] += f" | Fallback error: {str(fb_err)[:80]}"

        # Clean check
        if not result["caption"] and not result["audio_path"]:
            result["retrieval_status"] = "FAILED"
        elif not result["caption"] or not result["audio_path"]:
            if result["retrieval_status"] == "SUCCESS":
                result["retrieval_status"] = "PARTIAL"

        result["time_s"] = round(time.time() - t0, 3)
        return result

import os
import re
import sqlite3
import trafilatura
import yt_dlp
import requests
from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment

# --- Initial Setup ---
DB_FILE = "data_tracker.db"
CORPUS_DIR = "./corpus/"
# OpenAI's Whisper API has a 25MB file size limit. We'll use 24MB to be safe.
MAX_FILE_SIZE_MB = 24 * 1024 * 1024 
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Database Functions ---
def setup_database():
    """Creates the SQLite database and the processed_urls table if they don't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS processed_urls (
        id INTEGER PRIMARY KEY,
        url TEXT UNIQUE NOT NULL,
        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()
    print("🗃️ Database setup complete. `data_tracker.db` is ready.")

def is_url_processed(url: str) -> bool:
    """Checks if a URL has already been processed."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM processed_urls WHERE url = ?", (url,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def mark_url_as_processed(url: str):
    """Inserts a successfully processed URL into the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO processed_urls (url) VALUES (?)", (url,))
    conn.commit()
    conn.close()

# --- NEW HELPER FUNCTION for Audio Chunking ---
def split_audio_and_transcribe(audio_file_path: str) -> str:
    """Splits a large audio file into chunks and transcribes each, combining the results."""
    print("   Audio file is too large. Splitting into chunks...")
    audio = AudioSegment.from_file(audio_file_path)
    # Pydub works in milliseconds
    ten_minutes_in_ms = 10 * 60 * 1000
    chunks = [audio[i:i + ten_minutes_in_ms] for i in range(0, len(audio), ten_minutes_in_ms)]
    
    full_transcript = ""
    for i, chunk in enumerate(chunks):
        chunk_filename = f"temp_chunk_{i}.m4a"
        chunk.export(chunk_filename, format="mp4")
        print(f"      Transcribing chunk {i+1}/{len(chunks)}...")
        try:
            with open(chunk_filename, "rb") as f:
                transcript_part = client.audio.transcriptions.create(model="whisper-1", file=f)
            full_transcript += transcript_part.text + " "
        except Exception as e:
            print(f"      🔴 Error transcribing chunk {i+1}: {e}")
        finally:
            if os.path.exists(chunk_filename):
                os.remove(chunk_filename)
                
    return full_transcript.strip()


# --- Data Collection Functions ---
def transcribe_youtube_interview(url: str, output_filename: str):
    """Downloads audio from a YouTube URL and transcribes it using Whisper."""
    if is_url_processed(url):
        print(f"⏭️ Skipping YouTube URL (already processed): {url}")
        return

    print(f"▶️ Processing YouTube interview: {url}")
    audio_file_base = "temp_audio"
    final_audio_file = f"{audio_file_base}.m4a"
    
    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'outtmpl': audio_file_base,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'm4a'}],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("   Audio downloaded. Checking file size...")

        # UPGRADE: Check file size and decide whether to chunk
        file_size = os.path.getsize(final_audio_file)
        if file_size > MAX_FILE_SIZE_MB:
            transcript_text = split_audio_and_transcribe(final_audio_file)
        else:
            print("   File size is OK. Transcribing directly...")
            with open(final_audio_file, "rb") as f:
                transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
            transcript_text = transcript.text
        
        if transcript_text:
            with open(os.path.join(CORPUS_DIR, output_filename), 'w', encoding='utf-8') as f:
                f.write(transcript_text)
            print(f"   ✅ Successfully transcribed and saved to {output_filename}")
            mark_url_as_processed(url)
        else:
            print("   🔴 Transcription failed, no text to save.")

    except Exception as e:
        print(f"   🔴 Error processing {url}: {e}")
    finally:
        if os.path.exists(final_audio_file):
            os.remove(final_audio_file)

def scrape_blog(url: str, output_filename: str):
    """Scrapes a blog post using trafilatura and saves its text content."""
    if is_url_processed(url):
        print(f"⏭️ Skipping Blog URL (already processed): {url}")
        return

    print(f"▶️ Scraping blog: {url}")
    try:
        downloaded_content = trafilatura.fetch_url(url)
        if downloaded_content:
            main_text = trafilatura.extract(downloaded_content)
            with open(os.path.join(CORPUS_DIR, output_filename), 'w', encoding='utf-8') as f:
                f.write(main_text)
            print(f"   ✅ Successfully scraped and saved to {output_filename}")
            mark_url_as_processed(url)
        else:
            print(f"   🔴 Failed to download content from {url}")
    except Exception as e:
        print(f"   🔴 Error scraping {url}: {e}")

# --- Main Execution ---
def run_collection():
    """Main function to run the data collection process."""
    setup_database()
    if not os.path.exists(CORPUS_DIR):
        os.makedirs(CORPUS_DIR)

    youtube_sources = {
        "youtube_everyday_astronaut.txt": "https://www.youtube.com/watch?v=t705r8ICk2s",
        "youtube_ted_talk_2017.txt": "https://www.youtube.com/watch?v=zIwLWfaAg-8"
    }
    blog_sources = {
        "blog_wait_but_why_neuralink.txt": "https://waitbutwhy.com/2017/04/neuralink.html"
    }
    
    print("\n--- Starting Data Collection ---")
    for filename, url in youtube_sources.items():
        transcribe_youtube_interview(url, filename)
    for filename, url in blog_sources.items():
        scrape_blog(url, filename)
    print("\n--- Data Collection Complete ---")

# This is the "start button" for the script.
if __name__ == "__main__":
    run_collection()

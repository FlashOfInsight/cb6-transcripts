#!/usr/bin/env python3
"""
Pipeline orchestrator — main entry point for automated transcription.

Called by GitHub Actions daily cron. Detects new meeting videos on YouTube,
transcribes them with WhisperX on Modal, and rebuilds the site.

Usage:
    python scripts/pipeline.py              # Process all new videos
    python scripts/pipeline.py --dry-run    # Show what would be processed
    python scripts/pipeline.py --limit 1    # Process at most 1 new video
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add scripts dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from check_new_videos import check_new_videos, make_slug


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"
MEETINGS_JSON = PROJECT_ROOT / "src" / "_data" / "meetings.json"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def transcribe_with_modal(video_id):
    """Download and transcribe a YouTube video on Modal.

    Downloads audio on Modal's infrastructure (avoids YouTube bot detection)
    and transcribes with WhisperX + speaker diarization.

    Returns WhisperX result dict or None on failure.
    """
    try:
        import modal

        transcribe_yt = modal.Function.from_name(
            "cb6-transcribe", "transcribe_youtube"
        )

        print(f"    Sending to Modal for download + transcription...")
        result = transcribe_yt.remote(video_id)
        num_segments = len(result.get("segments", []))
        print(f"    Got {num_segments} segments from WhisperX")
        return result

    except Exception as e:
        print(f"    Modal transcription failed: {e}")
        return None


def add_meeting_entry(video_info, meetings):
    """Add a new meeting entry to meetings.json if not already present.

    Returns the meeting dict (existing or new).
    """
    video_id = video_info["video_id"]

    # Check if this video ID already exists
    for m in meetings:
        if m.get("videoId") == video_id:
            return m

    date_str = video_info.get("meeting_date")
    committee = video_info.get("committee", "Meeting")

    if not date_str:
        # Fall back to published date
        published = video_info.get("published", "")
        if published:
            date_str = published[:10]
        else:
            date_str = datetime.now().strftime("%Y-%m-%d")

    if not committee:
        committee = "Meeting"

    slug = make_slug(date_str, committee)

    # Ensure slug is unique
    existing_slugs = {m["slug"] for m in meetings}
    if slug in existing_slugs:
        slug = f"{slug}-{video_id[:6]}"

    entry = {
        "slug": slug,
        "date": date_str,
        "committee": committee,
        "topics": [],
        "youtubeUrl": f"https://www.youtube.com/watch?v={video_id}",
        "videoId": video_id,
        "agenda": None,
        "minutesPdf": None,
        "hasTranscript": True,
        "transcriptSource": "whisperx",
    }

    # Insert at the beginning (most recent first)
    meetings.insert(0, entry)
    print(f"    Added meeting entry: {slug}")
    return entry


def process_video(video_info, meetings, processed_videos):
    """Process a single video: download, transcribe, save.

    Returns True if successful, False otherwise.
    """
    video_id = video_info["video_id"]
    print(f"\n  Processing: {video_info['title']}")
    print(f"    Video ID: {video_id}")

    # 1. Download + transcribe on Modal (avoids YouTube bot detection on GH Actions)
    result = transcribe_with_modal(video_id)
    if not result or not result.get("segments"):
        print(f"    FAILED: Transcription returned no results")
        return False

    # 3. Save WhisperX JSON
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPTS_DIR / f"{video_id}.json"
    save_json(transcript_path, result)
    print(f"    Saved transcript: {transcript_path}")

    # 4. Add meeting entry to meetings.json
    add_meeting_entry(video_info, meetings)

    # 5. Mark as processed
    processed_videos[video_id] = {
        "processed_at": datetime.now().isoformat(),
        "status": "complete",
    }

    return True


def build_site():
    """Run build_site.py and pagefind to regenerate the site."""
    print("\n--- Building site ---")

    build_script = PROJECT_ROOT / "scripts" / "build_site.py"
    result = subprocess.run(
        [sys.executable, str(build_script)],
        cwd=str(PROJECT_ROOT),
        timeout=600,
    )
    if result.returncode != 0:
        print("WARNING: build_site.py exited with errors")
        return False

    # Run pagefind to rebuild search index
    print("\n--- Running pagefind ---")
    dist_dir = PROJECT_ROOT / "dist"
    if dist_dir.exists():
        pagefind_result = subprocess.run(
            ["npx", "pagefind", "--site", str(dist_dir)],
            cwd=str(PROJECT_ROOT),
            timeout=120,
        )
        if pagefind_result.returncode != 0:
            print("WARNING: pagefind exited with errors")
    else:
        print("WARNING: dist/ directory not found, skipping pagefind")

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CB6 Transcription Pipeline")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without doing it",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of videos to process",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip site rebuild after processing",
    )
    args = parser.parse_args()

    # Load data
    channels = load_json(DATA_DIR / "channels.json")
    processed_videos = load_json(DATA_DIR / "processed_videos.json")
    meetings = load_json(MEETINGS_JSON)

    print(f"Loaded {len(channels)} channels, {len(processed_videos)} processed videos")

    # Check each channel for new videos
    all_new_videos = []
    for channel in channels:
        channel_id = channel["id"]
        channel_name = channel["name"]
        print(f"\nChecking {channel_name} ({channel_id})...")

        try:
            new_videos = check_new_videos(channel_id, processed_videos)
            print(f"  Found {len(new_videos)} new videos")
            for v in new_videos:
                print(f"    - {v['title']} ({v['video_id']})")
            all_new_videos.extend(new_videos)
        except Exception as e:
            print(f"  ERROR checking channel: {e}")

    if not all_new_videos:
        print("\nNo new videos to process.")
        return

    if args.dry_run:
        print(f"\nDry run: would process {len(all_new_videos)} videos")
        return

    # Apply limit
    to_process = all_new_videos
    if args.limit:
        to_process = to_process[: args.limit]
        print(f"\nProcessing {len(to_process)} of {len(all_new_videos)} new videos (limit: {args.limit})")
    else:
        print(f"\nProcessing {len(to_process)} new videos")

    # Process each video
    success_count = 0
    for video_info in to_process:
        try:
            if process_video(video_info, meetings, processed_videos):
                success_count += 1
        except Exception as e:
            print(f"    ERROR processing {video_info['video_id']}: {e}")

        # Save progress after each video (in case of crash)
        save_json(DATA_DIR / "processed_videos.json", processed_videos)
        save_json(MEETINGS_JSON, meetings)

    print(f"\n--- Results: {success_count}/{len(to_process)} videos processed successfully ---")

    # Rebuild site if any new meetings were processed
    if success_count > 0 and not args.skip_build:
        build_site()
    elif success_count == 0:
        print("No videos processed successfully, skipping site rebuild")

    # Final save
    save_json(DATA_DIR / "processed_videos.json", processed_videos)
    save_json(MEETINGS_JSON, meetings)

    print("\nDone!")


if __name__ == "__main__":
    main()

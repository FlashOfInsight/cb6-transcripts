#!/usr/bin/env python3
"""
YouTube RSS feed poller — detects new videos on CB channels.

Uses stdlib only (no extra dependencies).
"""

import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime


RSS_URL_TEMPLATE = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"

# Namespace used in YouTube RSS feeds
YT_NS = "http://www.youtube.com/xml/schemas/2015"
MEDIA_NS = "http://search.yahoo.com/mrss/"
ATOM_NS = "http://www.w3.org/2005/Atom"

# Patterns for extracting committee name and date from video titles
# e.g. "1/14/2026 Full Board Meeting"
# e.g. "Transportation Committee Meeting - 1/5/2026"
# e.g. "Full Board Meeting 1/14/2026"
TITLE_PATTERNS = [
    # Date first: "1/14/2026 Full Board Meeting"
    re.compile(
        r"^(\d{1,2}/\d{1,2}/\d{4})\s+(.+?)(?:\s+Meeting)?$", re.IGNORECASE
    ),
    # Date last: "Full Board Meeting - 1/14/2026"
    re.compile(
        r"^(.+?)\s+(?:Meeting\s*)?[-–]\s*(\d{1,2}/\d{1,2}/\d{4})$",
        re.IGNORECASE,
    ),
    # Date last no separator: "Full Board Meeting 1/14/2026"
    re.compile(
        r"^(.+?)\s+Meeting\s+(\d{1,2}/\d{1,2}/\d{4})$", re.IGNORECASE
    ),
]


def parse_date_from_title(title):
    """Extract meeting date and committee name from video title.

    Returns (date_str, committee_name) or (None, None) if no match.
    date_str is in YYYY-MM-DD format.
    """
    for pattern in TITLE_PATTERNS:
        match = pattern.match(title.strip())
        if not match:
            continue

        groups = match.groups()
        # Figure out which group is the date and which is the committee
        date_str = None
        committee = None
        for g in groups:
            if re.match(r"\d{1,2}/\d{1,2}/\d{4}", g):
                date_str = g
            else:
                committee = g.strip()

        if date_str and committee:
            try:
                dt = datetime.strptime(date_str, "%m/%d/%Y")
                return dt.strftime("%Y-%m-%d"), clean_committee_name(committee)
            except ValueError:
                continue

    return None, None


def clean_committee_name(name):
    """Clean up committee name from video title."""
    # Remove "Meeting" suffix
    name = re.sub(r"\s*Meeting\s*$", "", name, flags=re.IGNORECASE)
    # Remove leading/trailing whitespace and dashes
    name = name.strip(" -–")
    # Normalize common names
    name_lower = name.lower()
    if "full board" in name_lower:
        return "Full Board"
    if "transportation" in name_lower:
        return "Transportation"
    if "land use" in name_lower:
        return "Land Use & Waterfront"
    if "business affairs" in name_lower:
        return "Business Affairs & Licensing"
    if "public safety" in name_lower:
        return "Public Safety & Sanitation"
    if "parks" in name_lower:
        return "Parks, Landmarks & Cultural Affairs"
    if "budget" in name_lower:
        return "Budget & Governmental Affairs"
    if "housing" in name_lower:
        return "Housing & Homelessness"
    if "health" in name_lower:
        return "Health & Education"
    if "youth" in name_lower:
        return "Youth & Education"
    if "executive" in name_lower:
        return "Executive"
    return name


def make_slug(date_str, committee):
    """Generate a URL-friendly slug from date and committee name."""
    slug_committee = committee.lower()
    slug_committee = re.sub(r"[&]", "and", slug_committee)
    slug_committee = re.sub(r"[^a-z0-9]+", "-", slug_committee)
    slug_committee = slug_committee.strip("-")
    return f"{date_str}-{slug_committee}"


def fetch_channel_videos(channel_id):
    """Fetch recent videos from a YouTube channel's RSS feed.

    Returns list of dicts with: video_id, title, published, link
    """
    url = RSS_URL_TEMPLATE.format(channel_id=channel_id)

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "cb6-transcripts-bot/1.0")

    with urllib.request.urlopen(req, timeout=30) as response:
        xml_data = response.read()

    root = ET.fromstring(xml_data)

    videos = []
    for entry in root.findall(f"{{{ATOM_NS}}}entry"):
        video_id_el = entry.find(f"{{{YT_NS}}}videoId")
        title_el = entry.find(f"{{{ATOM_NS}}}title")
        published_el = entry.find(f"{{{ATOM_NS}}}published")
        link_el = entry.find(f"{{{ATOM_NS}}}link")

        if video_id_el is None or title_el is None:
            continue

        video_id = video_id_el.text
        title = title_el.text
        published = published_el.text if published_el is not None else None
        link = link_el.get("href") if link_el is not None else None

        # Parse meeting info from title
        date_str, committee = parse_date_from_title(title)

        videos.append(
            {
                "video_id": video_id,
                "title": title,
                "published": published,
                "link": link or f"https://www.youtube.com/watch?v={video_id}",
                "meeting_date": date_str,
                "committee": committee,
            }
        )

    return videos


def check_new_videos(channel_id, processed_videos):
    """Check for new videos not yet in processed_videos.

    Args:
        channel_id: YouTube channel ID
        processed_videos: dict of video_id -> status info

    Returns:
        List of new video dicts
    """
    all_videos = fetch_channel_videos(channel_id)
    new_videos = [
        v for v in all_videos if v["video_id"] not in processed_videos
    ]
    return new_videos


if __name__ == "__main__":
    # Test with CB6 channel
    channel_id = "UCVn0y_oPVdX0vvRFl70YAig"
    print(f"Fetching videos from channel {channel_id}...")
    videos = fetch_channel_videos(channel_id)
    print(f"Found {len(videos)} videos in RSS feed:")
    for v in videos:
        date_info = f" (date: {v['meeting_date']}, committee: {v['committee']})" if v["meeting_date"] else ""
        print(f"  {v['video_id']}: {v['title']}{date_info}")

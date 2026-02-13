#!/usr/bin/env python3
"""
Build the CB6 Transcripts site - citymeetings.nyc style
Structure:
  - Homepage: List of meetings with descriptions and key topics
  - Meeting page: Expandable agenda with chapters inside each section
  - Chapter page: YouTube embed, summary, full transcript
"""

import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path


def load_whisperx_transcript(json_path):
    """Load WhisperX JSON and convert to paragraph format.

    Reads WhisperX output (segments with speaker labels and timestamps)
    and groups consecutive same-speaker segments into paragraphs.

    Returns list of dicts: {time: "HH:MM:SS", text: str, speaker: str}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    segments = data.get('segments', [])
    if not segments:
        return []

    # Group consecutive same-speaker segments into paragraphs
    paragraphs = []
    current = None

    for seg in segments:
        speaker = seg.get('speaker', 'UNKNOWN')
        text = seg.get('text', '').strip()
        if not text:
            continue

        # Convert seconds to HH:MM:SS
        start_seconds = seg.get('start', 0)
        hours = int(start_seconds // 3600)
        minutes = int((start_seconds % 3600) // 60)
        secs = int(start_seconds % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{secs:02d}"

        if current is None or current['speaker'] != speaker:
            # New speaker — start a new paragraph
            if current:
                paragraphs.append(current)
            current = {
                'time': time_str,
                'text': text,
                'speaker': speaker,
            }
        else:
            # Same speaker — append to current paragraph
            current['text'] += ' ' + text

    if current:
        paragraphs.append(current)

    return paragraphs


def download_captions(video_id, output_dir):
    """Download YouTube captions for a video."""
    output_path = output_dir / f"{video_id}.vtt"
    if output_path.exists():
        return output_path

    try:
        cmd = [
            "yt-dlp",
            "--write-auto-sub",
            "--sub-lang", "en",
            "--skip-download",
            "--sub-format", "vtt",
            "-o", str(output_dir / "%(id)s.%(ext)s"),
            f"https://www.youtube.com/watch?v={video_id}"
        ]
        subprocess.run(cmd, capture_output=True, timeout=60)

        actual_path = output_dir / f"{video_id}.en.vtt"
        if actual_path.exists():
            actual_path.rename(output_path)
            return output_path

    except Exception as e:
        print(f"    Error downloading captions: {e}")

    return None


def clean_vtt_with_timestamps(vtt_path):
    """Convert VTT to clean text with timestamps preserved."""
    with open(vtt_path, 'r') as f:
        content = f.read()

    content = content.replace('\ufeff', '')
    segments = []
    lines = content.split('\n')
    i = 0
    prev_text = ""

    while i < len(lines):
        line = lines[i].strip()
        time_match = re.match(r'(\d{2}:\d{2}:\d{2})\.\d{3} --> (\d{2}:\d{2}:\d{2})', line)
        if time_match:
            start_time = time_match.group(1)
            i += 1
            text_lines = []
            while i < len(lines) and not re.match(r'\d{2}:\d{2}:\d{2}', lines[i]):
                text = lines[i].strip()
                if text:
                    text = re.sub(r'<[\d:\.]+>', '', text)
                    text = re.sub(r'</?c>', '', text)
                    text = text.replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')
                    text_lines.append(text)
                i += 1

            full_text = ' '.join(text_lines)
            if full_text and full_text != prev_text:
                is_new_speaker = '>>' in full_text
                clean_text = full_text.replace('>>', '').strip()
                if clean_text:
                    segments.append({
                        'time': start_time,
                        'text': clean_text,
                        'new_speaker': is_new_speaker
                    })
                    prev_text = full_text
        else:
            i += 1

    return segments


def consolidate_segments(segments):
    """Merge consecutive segments from same speaker into paragraphs."""
    if not segments:
        return []

    cleaned = []
    prev_text = ""

    for seg in segments:
        curr_text = seg['text']
        new_text = curr_text

        if prev_text:
            for overlap_len in range(min(len(prev_text), len(curr_text)), 0, -1):
                if prev_text.endswith(curr_text[:overlap_len]):
                    new_text = curr_text[overlap_len:].strip()
                    break
            if curr_text in prev_text:
                new_text = ""

        if new_text:
            # Clean filler words
            new_text = re.sub(r'\b[Uu]m+\b[\s,]*', '', new_text)
            new_text = re.sub(r'\b[Uu]h+\b[\s,]*', '', new_text)
            new_text = re.sub(r'\s+', ' ', new_text).strip()
            if new_text:
                cleaned.append({
                    'time': seg['time'],
                    'text': new_text,
                    'new_speaker': seg['new_speaker']
                })
                prev_text = curr_text
        elif seg['new_speaker'] and cleaned:
            cleaned[-1]['new_speaker'] = True

    paragraphs = []
    current = None

    for seg in cleaned:
        if seg['new_speaker'] or current is None:
            if current:
                paragraphs.append(current)
            current = {'time': seg['time'], 'text': seg['text']}
        else:
            current['text'] += ' ' + seg['text']

    if current:
        paragraphs.append(current)

    return paragraphs


def parse_agenda(agenda_text):
    """Parse agenda into structured sections."""
    raw_sections = []
    lines = agenda_text.strip().split('\n')
    current_section = None
    current_items = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        roman_match = re.match(r'^([IVX]+)\.?\s+(.+)$', line)
        if roman_match:
            if current_section:
                raw_sections.append({
                    'id': current_section['id'],
                    'title': current_section['title'],
                    'items': current_items
                })
            current_section = {
                'id': roman_match.group(1),
                'title': roman_match.group(2).strip()
            }
            current_items = []
            continue

        num_match = re.match(r'^(\d+)\.?\s+(.+)$', line)
        if num_match:
            if current_section:
                raw_sections.append({
                    'id': current_section['id'],
                    'title': current_section['title'],
                    'items': current_items
                })
            parent = 'VI'
            for s in raw_sections:
                if 'committee' in s.get('title', '').lower():
                    parent = s['id']
                    break
            current_section = {
                'id': f"{parent}-{num_match.group(1)}",
                'title': num_match.group(2).strip()
            }
            current_items = []
            continue

        item_match = re.match(r'^([a-z])\)?\s+(.+)$', line)
        if item_match and current_section:
            current_items.append({
                'letter': item_match.group(1),
                'text': item_match.group(2).strip()
            })

    if current_section:
        raw_sections.append({
            'id': current_section['id'],
            'title': current_section['title'],
            'items': current_items
        })

    # Combine early procedural sections (Call to Order, Adoption, Roll Call)
    sections = []
    procedural_ids = {'I', 'II', 'III'}
    procedural_combined = None

    for section in raw_sections:
        sec_id = section['id']
        title_lower = section['title'].lower()

        # Check if this is a procedural opening section
        is_procedural = (
            sec_id in procedural_ids or
            'call to order' in title_lower or
            'adoption' in title_lower or
            'roll call' in title_lower or
            'attendance' in title_lower
        )

        if is_procedural and sec_id in {'I', 'II', 'III'}:
            if procedural_combined is None:
                procedural_combined = {
                    'id': 'I',
                    'title': 'Call to Order + Roll Call',
                    'items': section['items'].copy(),
                    'is_procedural': True
                }
            else:
                procedural_combined['items'].extend(section['items'])
        else:
            if procedural_combined and not sections:
                sections.append(procedural_combined)
                procedural_combined = None
            sections.append(section)

    # Add any remaining procedural section
    if procedural_combined and not sections:
        sections.append(procedural_combined)

    return sections


def find_section_boundaries(transcript, sections):
    """Find where each section starts based on content.

    Strategy: Look for explicit section transition phrases like:
    - "Next we have [committee name]"
    - "[committee name]. [Chair name] is chair"
    - "Okay, great. [Committee name]"
    """
    boundaries = {}
    section_ids = [s['id'] for s in sections]

    # Build full text with paragraph indices for searching
    full_text_parts = []
    para_starts = []  # Character position where each paragraph starts
    pos = 0
    for i, para in enumerate(transcript):
        para_starts.append(pos)
        full_text_parts.append(para['text'])
        pos += len(para['text']) + 1
    full_text = ' '.join(full_text_parts)
    full_text_lower = full_text.lower()

    def find_para_for_pos(char_pos):
        """Find which paragraph contains a character position."""
        for i in range(len(para_starts) - 1, -1, -1):
            if para_starts[i] <= char_pos:
                return i
        return 0

    # Find specific section markers
    # Format: (section_id, pattern, min_para, max_para)
    # None means no constraint
    total_paras = len(transcript)
    markers = [
        ('I', r'called to order', 0, 50),
        ('IV', r'(we have|next we have).{0,30}(council member|borough president|senator|assembly)', 20, 300),
        ('V', r'minutes from the.{0,20}(full board|december|january|february)', 100, None),
        ('VI', r'committee reports and resolutions', 200, None),
        ('VI-1', r'(health.{0,5}education|rupal.{0,10}chair|first committee.{0,20}health)', None, None),
        ('VI-2', r'(next.{0,30}transportation|transportation.{0,10}jason)', None, None),
        ('VI-3', r'(next.{0,30}business affairs|business affairs.{0,10}licensing)', None, None),
        ('VI-4', r'(next.{0,30}public safety|public safety.{0,10}sanitation)', None, None),
        ('VI-5', r'(sorry.{0,10}parks|parks.{0,5}landmarks.{0,10}cultural)', None, None),
        ('VI-6', r'(land use.{0,10}waterfront|land juice.{0,10}waterfront|gabe turzo)', None, None),
        ('VI-7', r'(budget.{0,10}government.{0,10}affairs|steve perez)', None, None),
        ('VI-8', r'(housing.{0,10}homelessness|alek miletic)', None, None),
        # These must be near the end of the meeting (last 15% of paragraphs)
        ('VIII', r'second roll call\.', int(total_paras * 0.85), None),
        ('IX', r'(will adjourn|we.{0,10}adjourn)', int(total_paras * 0.90), None),
    ]

    for sec_id, pattern, min_para, max_para in markers:
        if sec_id in boundaries:
            continue

        match = re.search(pattern, full_text_lower)
        if match:
            para_idx = find_para_for_pos(match.start())

            # Apply min/max constraints
            if min_para is not None and para_idx < min_para:
                continue
            if max_para is not None and para_idx > max_para:
                continue

            # For committee sections, ensure they come after VI
            if sec_id.startswith('VI-'):
                vi_boundary = boundaries.get('VI', 0)
                if para_idx <= vi_boundary:
                    # Try to find a later occurrence
                    search_start = para_starts[vi_boundary] if vi_boundary < len(para_starts) else 0
                    match = re.search(pattern, full_text_lower[search_start:])
                    if match:
                        para_idx = find_para_for_pos(search_start + match.start())
                    else:
                        continue

            boundaries[sec_id] = para_idx

    # Ensure committee sections are in order
    committee_order = ['VI-1', 'VI-2', 'VI-3', 'VI-4', 'VI-5', 'VI-6', 'VI-7', 'VI-8']
    last_boundary = boundaries.get('VI', 0)

    for sec_id in committee_order:
        if sec_id in boundaries:
            if boundaries[sec_id] < last_boundary:
                # This section was found before the previous one - invalid
                del boundaries[sec_id]
            else:
                last_boundary = boundaries[sec_id]

    return boundaries


def apply_sections(transcript, boundaries, sections):
    """Apply section labels based on boundaries."""
    sorted_bounds = sorted(boundaries.items(), key=lambda x: x[1])
    sorted_bounds.append(('END', len(transcript)))

    for i, (sec_id, start) in enumerate(sorted_bounds[:-1]):
        end = sorted_bounds[i + 1][1]
        for j in range(start, end):
            transcript[j]['section'] = sec_id

    first_boundary = sorted_bounds[0][1] if sorted_bounds else 0
    for j in range(first_boundary):
        transcript[j]['section'] = 'I' if 'I' in [s['id'] for s in sections] else sections[0]['id']

    return transcript


def extract_keywords_from_text(text):
    """Extract meaningful keywords from agenda item text."""
    # Remove common words and extract key terms
    text_lower = text.lower()

    # Common patterns to extract
    keywords = []

    # Extract proper nouns and specific terms
    words = re.findall(r'[A-Za-z]{3,}', text)
    stop_words = {'the', 'and', 'for', 'resolution', 'regarding', 'support', 'supporting',
                  'from', 'with', 'that', 'this', 'will', 'are', 'was', 'were', 'have',
                  'has', 'had', 'been', 'being', 'report', 'reports', 'committee'}

    for word in words:
        word_lower = word.lower()
        if word_lower not in stop_words and len(word_lower) > 3:
            keywords.append(word_lower)

    return keywords


def fuzzy_match_in_text(keyword, text):
    """Check if keyword fuzzy-matches anywhere in text (handles transcription errors)."""
    text_lower = text.lower()
    keyword_lower = keyword.lower()

    # Direct match
    if keyword_lower in text_lower:
        return True

    # Common transcription errors mapping
    error_mappings = {
        'bellevue': ['bel', 'bellview', 'belleview', 'belle view'],
        'helicopter': ['helcopter', 'hellicopter'],  # heliport is a real word, not an error
        'heliport': ['helport', 'heli port'],
        'conedison': ['coned', 'con ed', 'con edison'],
        'housing': ['houseing', 'housng'],
        'mental': ['mantal', 'mentl'],
        'transportation': ['transportaion', 'transport'],
        'cannabis': ['canabis', 'cannibis'],
        'budget': ['buget', 'budjet'],
        'education': ['educaton', 'eduacation'],
        'health': ['helth', 'heath'],
        'legislation': ['legislaton', 'legisation'],
    }

    # Check if this keyword has common transcription errors
    for correct, variants in error_mappings.items():
        if keyword_lower == correct or keyword_lower in variants:
            if correct in text_lower:
                return True
            for var in variants:
                if var in text_lower:
                    return True

    # Simple edit distance for short keywords (expensive, use sparingly)
    if len(keyword_lower) >= 5:
        words = text_lower.split()
        for word in words:
            if len(word) >= 4:
                # Check if first 4 chars match (common prefix)
                if word[:4] == keyword_lower[:4]:
                    return True

    return False


def find_keyword_in_section(keywords, section_content, start_from=0):
    """Find the first paragraph where any keyword appears."""
    for i in range(start_from, len(section_content)):
        para_text = section_content[i]['text']
        for kw in keywords:
            if fuzzy_match_in_text(kw, para_text):
                return i
    return None


def find_transition_phrase(section_content, item_index, start_from=0):
    """Look for transitional phrases that indicate moving to the next agenda item.

    Returns (para_index, char_offset) where char_offset is position within the paragraph.
    If the transition is at the start, char_offset is 0.
    """
    # Patterns must specifically indicate introducing a new agenda item
    # Avoid false positives like "second roll call"
    ordinals = {
        1: r'(the\s+second\s+(?:resolution|item|reso)|second\s+resolution|next\s+(?:resolution|item|we\s+have))',
        2: r'(the\s+third\s+(?:resolution|item)|third\s+resolution)',
        3: r'(the\s+fourth\s+(?:resolution|item)|fourth\s+resolution)',
        4: r'(the\s+fifth\s+(?:resolution|item)|fifth\s+resolution)',
        5: r'(the\s+sixth\s+(?:resolution|item)|sixth\s+resolution)',
        6: r'(the\s+seventh\s+(?:resolution|item)|seventh\s+resolution)',
        7: r'(the\s+eighth\s+(?:resolution|item)|eighth\s+resolution)',
    }

    if item_index not in ordinals:
        return None, None

    pattern = ordinals[item_index]

    # Search starting from the previous item's paragraph (allow same paragraph)
    for i in range(max(0, start_from - 1), len(section_content)):
        para_text = section_content[i]['text'].lower()
        match = re.search(pattern, para_text)
        if match:
            # Return both the paragraph index and the character offset
            return i, match.start()

    return None, None


def create_chapters_from_agenda(section, section_content, transcript):
    """Create chapters from agenda items with smart keyword-based alignment."""
    chapters = []
    sec_id = section['id']
    items = section.get('items', [])

    if not section_content:
        return chapters

    # Get section start time
    section_start_time = section_content[0]['time'] if section_content else '00:00:00'

    if not items:
        # Section has no sub-items, create a single chapter for the whole section
        # Generate a summary from the content
        all_text = ' '.join(p['text'] for p in section_content[:5])
        summary = all_text[:300] + '...' if len(all_text) > 300 else all_text

        chapters.append({
            'id': f"{sec_id}-main",
            'letter': '',
            'title': section['title'],
            'time': section_start_time,
            'para_start': 0,
            'para_end': len(section_content),
            'summary': summary,
            'category': categorize_content(section['title'], all_text)
        })
        return chapters

    # Try to find each item using transition phrases first, then keyword matching
    item_boundaries = []
    last_found_para = 0

    for i, item in enumerate(items):
        text = item.get('text', '')
        keywords = extract_keywords_from_text(text)
        found_para = None

        # For items after the first, look for transitional phrases first
        # e.g., "The second resolution", "Next we have", "Item B"
        if i > 0:
            found_para, char_offset = find_transition_phrase(section_content, i, last_found_para)
            # If found in same paragraph as previous item, still use it
            # (the speaker introduced multiple items in one breath)

        # If no transition phrase found, try keyword matching
        if found_para is None:
            found_para = find_keyword_in_section(keywords, section_content, last_found_para)

        if found_para is not None:
            item_boundaries.append(found_para)
            # Allow next item to be in same paragraph if transition found there
            last_found_para = found_para
        else:
            # Fallback: estimate based on even division
            paras_per_item = max(1, len(section_content) // len(items))
            estimated = i * paras_per_item
            # Ensure we don't go backwards
            if estimated < last_found_para:
                estimated = last_found_para
            item_boundaries.append(estimated)
            last_found_para = estimated

    # Build chapters with the boundaries we found
    # Track transition offsets for content splitting
    transition_offsets = {}  # para_index -> char_offset for items that found transitions

    # Re-scan for transition offsets
    for i in range(1, len(items)):
        para_idx, char_offset = find_transition_phrase(section_content, i, 0)
        if para_idx is not None and char_offset is not None:
            if para_idx not in transition_offsets:
                transition_offsets[para_idx] = []
            transition_offsets[para_idx].append((i, char_offset))

    for i, item in enumerate(items):
        letter = item.get('letter', chr(ord('a') + i))
        text = item.get('text', '')

        para_start = item_boundaries[i]
        if i < len(items) - 1:
            para_end = item_boundaries[i + 1]
        else:
            para_end = len(section_content)

        # Ensure valid ranges
        para_start = min(para_start, len(section_content) - 1)
        para_end = max(para_end, para_start + 1)

        # Get time from first paragraph of this item's range
        if para_start < len(section_content):
            item_time = section_content[para_start]['time']
        else:
            item_time = section_start_time

        # Generate summary from the content
        # If items share a paragraph, split the text at the transition phrase
        item_paras = section_content[para_start:min(para_end, len(section_content))]

        if item_paras:
            first_para_text = item_paras[0]['text']

            # Check if this paragraph has transitions and we need to split
            if para_start in transition_offsets:
                offsets = sorted(transition_offsets[para_start], key=lambda x: x[1])

                # Find the relevant portion for this item
                if i == 0:
                    # First item: use text before the first transition
                    first_offset = offsets[0][1] if offsets else len(first_para_text)
                    first_para_text = first_para_text[:first_offset].strip()
                else:
                    # Find this item's transition offset
                    my_offset = None
                    next_offset = len(first_para_text)
                    for item_idx, offset in offsets:
                        if item_idx == i:
                            my_offset = offset
                        elif item_idx > i and my_offset is not None:
                            next_offset = offset
                            break
                    if my_offset is not None:
                        first_para_text = first_para_text[my_offset:next_offset].strip()

            all_text = first_para_text
            for p in item_paras[1:5]:
                all_text += ' ' + p['text']

        else:
            all_text = ''

        summary = all_text[:300] + '...' if len(all_text) > 300 else all_text

        chapters.append({
            'id': f"{sec_id}-{letter}",
            'letter': letter,
            'title': text,
            'time': item_time,
            'para_start': para_start,
            'para_end': para_end,
            'summary': summary,
            'category': categorize_content(text, all_text)
        })

    return chapters


def categorize_content(title, text):
    """Determine the category of content based on title and text."""
    title_lower = title.lower()
    text_lower = text.lower()
    combined = title_lower + ' ' + text_lower

    if 'resolution' in combined or 'reso' in combined:
        return 'RESOLUTION'
    elif 'vote' in combined or 'ballot' in combined:
        return 'VOTE'
    elif 'report' in combined:
        return 'REPORT'
    elif 'public' in combined and ('session' in combined or 'comment' in combined):
        return 'PUBLIC COMMENT'
    elif 'present' in combined:
        return 'PRESENTATION'
    elif 'discuss' in combined:
        return 'DISCUSSION'
    elif 'adjourn' in combined:
        return 'CLOSING'
    elif 'call to order' in combined or 'roll call' in combined:
        return 'OPENING'
    else:
        return 'DISCUSSION'


def extract_chapters(paragraphs, section_title, section_id, is_procedural=False):
    """Legacy function - kept for compatibility but not used."""
    chapters = []
    if not paragraphs:
        return chapters

    # Skip detailed chapter extraction for procedural sections
    title_lower = section_title.lower()
    if is_procedural or 'call to order' in title_lower or 'roll call' in title_lower:
        # Just create one simple chapter for procedural sections
        first_para = paragraphs[0]
        chapters.append({
            'id': f"{section_id}-1",
            'category': 'OPENING',
            'title': 'Meeting Opening',
            'description': 'Call to order, attendance, and meeting ground rules.',
            'time': first_para['time'],
            'duration': estimate_duration(paragraphs, 0)
        })
        return chapters

    all_text = ' '.join(p['text'] for p in paragraphs)
    text_lower = all_text.lower()

    def find_para_for_position(char_pos):
        """Find which paragraph contains a character position."""
        count = 0
        for i, p in enumerate(paragraphs):
            if count + len(p['text']) >= char_pos:
                return i
            count += len(p['text']) + 1
        return 0

    # Topic keywords to search for
    topics = {
        'bellevue': ('Bellevue Hospital funding', 'RESOLUTION'),
        'helicopter': ('Non-essential helicopter flight ban', 'RESOLUTION'),
        'bike lane': ('Third Avenue bike lane concerns', 'PUBLIC COMMENT'),
        'mental health': ('Mental health diversion program', 'RESOLUTION'),
        'diversion': ('Criminal justice diversion bill', 'DISCUSSION'),
        'treatment court': ('Treatment court expansion', 'DISCUSSION'),
        'coned': ('Con Edison rate increase', 'DISCUSSION'),
        'utility': ('Utility rate concerns', 'DISCUSSION'),
        'housing': ('Housing affordability', 'DISCUSSION'),
        'cannabis': ('Cannabis licensing', 'DISCUSSION'),
    }

    found_topics = set()
    for keyword, (title, category) in topics.items():
        if keyword in text_lower and keyword not in found_topics:
            idx = text_lower.find(keyword)
            para_idx = find_para_for_position(idx)
            para = paragraphs[para_idx]

            # Get surrounding context
            start = max(0, idx - 50)
            end = min(len(all_text), idx + 300)
            context = all_text[start:end].strip()

            chapters.append({
                'id': f"{section_id}-{len(chapters)+1}",
                'category': category,
                'title': title,
                'description': context[:250] + '...' if len(context) > 250 else context,
                'time': para['time'],
                'duration': estimate_duration(paragraphs, para_idx)
            })
            found_topics.add(keyword)

    # Look for elected officials
    officials = [
        ('borough president', 'Borough President remarks', 'REPORT'),
        ('council member harvey', 'Council Member Harvey Epstein', 'REPORT'),
        ('council member virginia', 'Council Member Virginia Maloney', 'REPORT'),
        ('assembly member', 'Assembly Member report', 'REPORT'),
        ('senator krueger', 'Senator Krueger update', 'REPORT'),
        ('senator gonzalez', 'Senator Gonzalez update', 'REPORT'),
        ('controller', 'City Controller update', 'REPORT'),
    ]

    for keyword, title, category in officials:
        if keyword in text_lower and keyword not in found_topics:
            idx = text_lower.find(keyword)
            para_idx = find_para_for_position(idx)
            para = paragraphs[para_idx]

            context = all_text[max(0, idx-20):min(len(all_text), idx+200)].strip()

            chapters.append({
                'id': f"{section_id}-{len(chapters)+1}",
                'category': category,
                'title': title,
                'description': context[:200] + '...' if len(context) > 200 else context,
                'time': para['time'],
                'duration': estimate_duration(paragraphs, para_idx)
            })
            found_topics.add(keyword)

    # Look for votes - but be more careful with extraction
    if 'resolution' in text_lower and 'vote' in text_lower:
        # Look for resolution topics
        res_patterns = [
            r'resolution\s+(?:to\s+|for\s+|on\s+|regarding\s+)([a-z][a-z\s]{15,80})',
            r'vote\s+on\s+the\s+([a-z][a-z\s]{15,80})',
        ]
        for pattern in res_patterns:
            match = re.search(pattern, text_lower[:3000])
            if match:
                topic = match.group(1).strip()
                # Clean up the topic
                topic = re.sub(r'\s+', ' ', topic)
                if len(topic) > 15:
                    idx = match.start()
                    para_idx = find_para_for_position(idx)
                    para = paragraphs[para_idx]

                    chapters.append({
                        'id': f"{section_id}-{len(chapters)+1}",
                        'category': 'RESOLUTION',
                        'title': f"Resolution: {topic[:50].title()}",
                        'description': all_text[idx:idx+200].strip() + '...',
                        'time': para['time'],
                        'duration': estimate_duration(paragraphs, para_idx)
                    })
                    break

    # Look for public speakers - only match clear patterns
    speaker_pattern = re.compile(r'(?:next|first)\s+(?:on the list|speaker|we have)\s+(?:is\s+)?([A-Z][a-z]+\s+[A-Z][a-z]+)', re.IGNORECASE)
    for match in speaker_pattern.finditer(all_text[:5000]):
        name = match.group(1).strip()
        # Skip if name looks wrong
        if len(name) < 5 or name.lower() in ['we have', 'next we', 'first we']:
            continue

        idx = match.start()
        para_idx = find_para_for_position(idx)
        para = paragraphs[para_idx]

        # Get what they're speaking about
        context = all_text[idx:idx+300]
        topic_match = re.search(r'(?:speak|wishes to speak|to speak)\s+(?:on|about)\s+(.{20,80})', context, re.IGNORECASE)
        topic = topic_match.group(1)[:50].strip() if topic_match else ''

        title = f"Public comment: {name}"
        if topic:
            title += f" on {topic}"

        chapters.append({
            'id': f"{section_id}-{len(chapters)+1}",
            'category': 'PUBLIC COMMENT',
            'title': title,
            'description': context[:200].strip() + '...',
            'time': para['time'],
            'duration': estimate_duration(paragraphs, para_idx)
        })

    # Look for procedural/business items
    procedural = [
        ('called to order', 'Meeting called to order', 'ANNOUNCEMENT'),
        ('adopt the agenda', 'Agenda adoption', 'REMARKS'),
        ('take attendance', 'Roll call', 'REPORT'),
        ('ground rules', 'Meeting ground rules', 'REMARKS'),
        ('minutes from the', 'Minutes adoption', 'BUSINESS'),
        ('minutes are adopted', 'Minutes adopted', 'BUSINESS'),
        ("chair's report", "Chair's report", 'REPORT'),
        ('second roll call', 'Second roll call', 'ROLL CALL'),
        ('adjourned', 'Meeting adjourned', 'ANNOUNCEMENT'),
    ]

    for keyword, title, category in procedural:
        if keyword in text_lower:
            idx = text_lower.find(keyword)
            para_idx = find_para_for_position(idx)
            para = paragraphs[para_idx]

            context = all_text[max(0, idx-20):min(len(all_text), idx+150)].strip()

            chapters.append({
                'id': f"{section_id}-{len(chapters)+1}",
                'category': category,
                'title': title,
                'description': context[:180] + '...' if len(context) > 180 else context,
                'time': para['time'],
                'duration': estimate_duration(paragraphs, para_idx)
            })

    # If no chapters found, create one for the section
    if not chapters and paragraphs:
        first_para = paragraphs[0]
        chapters.append({
            'id': f"{section_id}-1",
            'category': 'REMARKS',
            'title': section_title,
            'description': all_text[:250] + '...' if len(all_text) > 250 else all_text,
            'time': first_para['time'],
            'duration': estimate_duration(paragraphs, 0)
        })

    # Sort by time and deduplicate
    chapters.sort(key=lambda x: get_time_seconds(x['time']))

    seen = set()
    unique = []
    for ch in chapters:
        key = ch['title'].lower()[:30]
        if key not in seen:
            seen.add(key)
            unique.append(ch)

    return unique[:10]  # Limit to 10 chapters per section


def estimate_duration(paragraphs, start_idx):
    """Estimate duration of a chapter in seconds."""
    if start_idx >= len(paragraphs) - 1:
        return 60  # Default 1 minute for last item

    start = get_time_seconds(paragraphs[start_idx]['time'])

    # Find next significant break (next speaker or topic change)
    for i in range(start_idx + 1, min(start_idx + 10, len(paragraphs))):
        if i < len(paragraphs):
            end = get_time_seconds(paragraphs[i]['time'])
            if end - start > 30:  # At least 30 seconds
                return min(end - start, 600)  # Cap at 10 minutes

    return 120  # Default 2 minutes


def get_time_seconds(time_str):
    """Convert time string to seconds."""
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        return h * 3600 + m * 60 + s
    return 0


def format_time_display(time_str):
    """Format time for display."""
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"
    return time_str


def format_duration(seconds):
    """Format duration for display."""
    if seconds < 60:
        return f"{seconds} sec"
    elif seconds < 3600:
        return f"{seconds // 60} min"
    else:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def format_date_display(date_str):
    """Format date for display."""
    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    try:
        parts = date_str.split('-')
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
        return f"{months[month-1]} {day}, {year}"
    except:
        return date_str


def process_meeting(meeting, project_root, misspellings):
    """Process a single meeting."""
    video_id = meeting.get('videoId')
    if not video_id:
        return None

    transcripts_dir = project_root / 'transcripts'
    transcripts_dir.mkdir(exist_ok=True)

    transcript_source = 'vtt'
    has_speaker_labels = False

    # Transcript source routing:
    # 1. WhisperX JSON (best quality, has speaker labels)
    # 2. Existing VTT file
    # 3. Download VTT from YouTube (fallback)
    whisperx_path = transcripts_dir / f"{video_id}.json"
    vtt_path = transcripts_dir / f"{video_id}.vtt"

    if whisperx_path.exists():
        print(f"    Using WhisperX transcript")
        paragraphs = load_whisperx_transcript(whisperx_path)
        transcript_source = 'whisperx'
        has_speaker_labels = True
    elif vtt_path.exists():
        print(f"    Using VTT transcript")
        segments = clean_vtt_with_timestamps(vtt_path)
        paragraphs = consolidate_segments(segments)
    else:
        print(f"    Downloading captions...")
        vtt_path = download_captions(video_id, transcripts_dir)
        if not vtt_path:
            return None
        segments = clean_vtt_with_timestamps(vtt_path)
        paragraphs = consolidate_segments(segments)

    if not paragraphs:
        return None

    # Apply name corrections
    for para in paragraphs:
        text = para['text']
        for wrong, right in misspellings.items():
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            text = pattern.sub(right, text)
        para['text'] = text

    has_agenda = bool((meeting.get('agenda') or '').strip())

    if has_agenda:
        sections = parse_agenda(meeting['agenda'])
        if not sections:
            sections = [{'id': 'I', 'title': 'Meeting', 'items': []}]

        boundaries = find_section_boundaries(paragraphs, sections)
        paragraphs = apply_sections(paragraphs, boundaries, sections)

        # Group transcript by section
        sections_content = {}
        for para in paragraphs:
            sec_id = para.get('section', 'I')
            if sec_id not in sections_content:
                sections_content[sec_id] = []
            sections_content[sec_id].append(para)

        # Create chapters from agenda items
        sections_chapters = {}
        for section in sections:
            sec_id = section['id']
            content = sections_content.get(sec_id, [])
            chapters = create_chapters_from_agenda(section, content, paragraphs)
            sections_chapters[sec_id] = chapters

        boundaries_result = boundaries
    else:
        # No agenda — single-section transcript (auto-detected meetings)
        sections = [{'id': 'I', 'title': meeting.get('committee', 'Meeting'), 'items': []}]
        sections_content = {'I': paragraphs}
        sections_chapters = {}
        boundaries_result = {}

    youtube_url = meeting.get('youtubeUrl', f"https://www.youtube.com/watch?v={video_id}")

    return {
        'meeting': {
            'slug': meeting['slug'],
            'date': meeting['date'],
            'committee': meeting['committee'],
            'youtubeUrl': youtube_url,
            'videoId': video_id,
            'topics': meeting.get('topics', []),
            'transcriptSource': transcript_source,
            'hasAgenda': has_agenda,
        },
        'sections': sections,
        'transcript': paragraphs,
        'sections_content': sections_content,
        'sections_chapters': sections_chapters,
        'section_boundaries': boundaries_result,
        'has_speaker_labels': has_speaker_labels,
    }


def generate_meeting_pages(data, output_dir):
    """Generate all pages for a meeting."""
    meeting = data['meeting']
    sections = data['sections']
    sections_content = data['sections_content']
    sections_chapters = data['sections_chapters']

    output_dir.mkdir(parents=True, exist_ok=True)

    # For meetings without an agenda, generate a simpler full-transcript page
    if not meeting.get('hasAgenda', True):
        generate_transcript_only_page(data, output_dir)
        return

    # Build a global chronological list of all chapters for cross-section navigation
    all_chapters = []
    for section in sections:
        sec_id = section['id']
        chapters = sections_chapters.get(sec_id, [])
        for ch in chapters:
            all_chapters.append({
                'chapter': ch,
                'section': section,
                'content': sections_content.get(sec_id, [])
            })

    # Sort by timestamp
    all_chapters.sort(key=lambda x: get_time_seconds(x['chapter']['time']))

    # Generate meeting index with expandable agenda (skip empty sections)
    generate_meeting_index(data, output_dir)

    # Generate individual chapter pages with global navigation
    for i, item in enumerate(all_chapters):
        prev_ch = all_chapters[i - 1] if i > 0 else None
        next_ch = all_chapters[i + 1] if i < len(all_chapters) - 1 else None
        generate_chapter_page(
            meeting,
            item['section'],
            item['chapter'],
            item['content'],
            prev_ch,
            next_ch,
            output_dir
        )


def generate_meeting_index(data, output_dir):
    """Generate meeting index with expandable agenda sections."""
    meeting = data['meeting']
    sections = data['sections']
    sections_content = data['sections_content']
    sections_chapters = data['sections_chapters']

    css = get_css()

    # Filter to only sections with chapters
    sections_with_chapters = [s for s in sections if sections_chapters.get(s['id'], [])]

    # Count total chapters
    total_chapters = sum(len(chs) for chs in sections_chapters.values())

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{meeting['committee']} - {format_date_display(meeting['date'])} | CB6 Transcripts</title>
  <style>{css}</style>
</head>
<body>
  <header>
    <div class="container">
      <nav class="breadcrumb">
        <a href="../../index.html">All Meetings</a>
        <span class="breadcrumb-sep">›</span>
        <span style="color: white;">{meeting['committee']} - {format_date_display(meeting['date'])}</span>
      </nav>
      <h1>{meeting['committee']}</h1>
      <div class="meta">{format_date_display(meeting['date'])} · Manhattan Community Board 6</div>
    </div>
  </header>

  <div class="container">
    <div class="meeting-summary">
      <p>This meeting contains <strong>{len(sections_with_chapters)} agenda sections</strong> and <strong>{total_chapters} indexed chapters</strong>. Click on any section to expand and see the chapters within.</p>
    </div>

    <h2 class="section-heading">Agenda</h2>
'''

    for section in sections_with_chapters:
        sec_id = section['id']
        title = section['title']
        chapters = sections_chapters.get(sec_id, [])
        content = sections_content.get(sec_id, [])

        first_time = content[0]['time'] if content else '00:00:00'
        time_display = format_time_display(first_time)

        # Determine section category/type
        sec_type = 'AGENDA ITEM'
        title_lower = title.lower()
        if 'call to order' in title_lower:
            sec_type = 'OPENING'
        elif 'roll call' in title_lower or 'attendance' in title_lower:
            sec_type = 'ROLL CALL'
        elif 'public session' in title_lower:
            sec_type = 'PUBLIC SESSION'
        elif 'business session' in title_lower:
            sec_type = 'BUSINESS'
        elif 'committee' in title_lower:
            sec_type = 'COMMITTEE REPORTS'
        elif 'adjourn' in title_lower:
            sec_type = 'CLOSING'

        html += f'''
    <div class="agenda-section" id="section-{sec_id.lower().replace('-', '')}">
      <div class="agenda-header" onclick="toggleSection('{sec_id.lower().replace('-', '')}')">
        <div class="agenda-header-left">
          <span class="section-type">{sec_type}</span>
          <h3>{sec_id}. {title}</h3>
        </div>
        <div class="agenda-header-right">
          <span class="chapter-count">{len(chapters)} chapter{"s" if len(chapters) != 1 else ""}</span>
          <span class="timestamp">{time_display}</span>
          <span class="expand-icon">▼</span>
        </div>
      </div>
      <div class="agenda-chapters" id="chapters-{sec_id.lower().replace('-', '')}">
'''

        for chapter in chapters:
            ch_slug = chapter['id'].lower().replace('-', '')
            ch_time = format_time_display(chapter['time'])
            letter = chapter.get('letter', '')
            title = chapter.get('title', 'Report')

            # Format the item label
            if letter:
                item_label = f"{letter}) {title}"
            else:
                item_label = title

            html += f'''
        <a href="chapter-{ch_slug}.html" class="agenda-item">
          <span class="agenda-item-letter">{letter})</span>
          <span class="agenda-item-title">{title}</span>
          <span class="agenda-item-time">{ch_time}</span>
        </a>
'''

        html += '''
      </div>
    </div>
'''

    html += '''
  </div>

  <footer>
    <p>Transcripts auto-generated from <a href="''' + meeting['youtubeUrl'] + '''" target="_blank">CB6 YouTube</a>.</p>
  </footer>

  <script>
    function toggleSection(id) {
      const chapters = document.getElementById('chapters-' + id);
      const section = document.getElementById('section-' + id);
      const wasExpanded = section.classList.contains('expanded');

      // Close all sections
      document.querySelectorAll('.agenda-section').forEach(s => s.classList.remove('expanded'));
      document.querySelectorAll('.agenda-chapters').forEach(c => c.classList.remove('expanded'));

      // Toggle clicked section
      if (!wasExpanded) {
        chapters.classList.add('expanded');
        section.classList.add('expanded');
        // Update URL hash
        history.replaceState(null, null, '#' + id);
      } else {
        history.replaceState(null, null, window.location.pathname);
      }
    }

    // Expand section from URL hash, or first section by default
    document.addEventListener('DOMContentLoaded', function() {
      const hash = window.location.hash.substring(1);
      let targetSection = null;

      if (hash) {
        targetSection = document.getElementById('section-' + hash);
      }

      if (!targetSection) {
        targetSection = document.querySelector('.agenda-section');
      }

      if (targetSection) {
        targetSection.classList.add('expanded');
        const chapters = targetSection.querySelector('.agenda-chapters');
        if (chapters) chapters.classList.add('expanded');
      }
    });
  </script>
</body>
</html>
'''

    with open(output_dir / 'index.html', 'w') as f:
        f.write(html)


def generate_chapter_page(meeting, section, chapter, section_content, prev_item, next_item, output_dir):
    """Generate a chapter detail page with global prev/next navigation."""
    css = get_css()

    ch_slug = chapter['id'].lower().replace('-', '')
    sec_id = section['id']
    start_seconds = get_time_seconds(chapter['time'])
    time_display = format_time_display(chapter['time'])

    video_id = meeting['videoId']
    embed_url = f"https://www.youtube.com/embed/{video_id}?start={start_seconds}"
    direct_url = f"https://www.youtube.com/watch?v={video_id}&t={start_seconds}"

    # Get chapter title with letter prefix
    letter = chapter.get('letter', '')
    title = chapter.get('title', section['title'])
    full_title = f"{letter}) {title}" if letter else title

    # Get chapter metadata
    category = chapter.get('category', 'DISCUSSION')
    summary = chapter.get('summary', '')
    category_color = get_category_color(category)

    # Get transcript for this chapter's paragraph range
    para_start = chapter.get('para_start', 0)
    para_end = chapter.get('para_end', len(section_content))

    transcript_html = ''
    for i in range(para_start, min(para_end, len(section_content))):
        para = section_content[i]
        para_time_display = format_time_display(para['time'])
        text = para['text']
        speaker = para.get('speaker', '')
        if speaker:
            speaker_class = f"speaker-{speaker.lower().replace('_', '-')}"
            transcript_html += f'<div class="transcript-paragraph"><span class="speaker-label {speaker_class}">{speaker}</span><span class="transcript-time">{para_time_display}</span><p>{text}</p></div>\n'
        else:
            transcript_html += f'<p><span class="transcript-time">{para_time_display}</span> {text}</p>\n'

    if not transcript_html and section_content:
        # Fallback to first 10 paragraphs of section
        for para in section_content[:10]:
            para_time_display = format_time_display(para['time'])
            speaker = para.get('speaker', '')
            if speaker:
                speaker_class = f"speaker-{speaker.lower().replace('_', '-')}"
                transcript_html += f'<div class="transcript-paragraph"><span class="speaker-label {speaker_class}">{speaker}</span><span class="transcript-time">{para_time_display}</span><p>{para["text"]}</p></div>\n'
            else:
                transcript_html += f'<p><span class="transcript-time">{para_time_display}</span> {para["text"]}</p>\n'

    # Build prev/next links from global chapter list
    prev_link = ''
    next_link = ''
    if prev_item:
        prev_ch = prev_item['chapter']
        prev_slug = prev_ch['id'].lower().replace('-', '')
        prev_letter = prev_ch.get('letter', '')
        prev_title = prev_ch.get('title', '')[:35]
        if len(prev_ch.get('title', '')) > 35:
            prev_title += '...'
        prev_label = f"{prev_letter}) {prev_title}" if prev_letter else prev_title
        prev_link = f'<a href="chapter-{prev_slug}.html" class="nav-prev">← {prev_label}</a>'
    if next_item:
        next_ch = next_item['chapter']
        next_slug = next_ch['id'].lower().replace('-', '')
        next_letter = next_ch.get('letter', '')
        next_title = next_ch.get('title', '')[:35]
        if len(next_ch.get('title', '')) > 35:
            next_title += '...'
        next_label = f"{next_letter}) {next_title}" if next_letter else next_title
        next_link = f'<a href="chapter-{next_slug}.html" class="nav-next">{next_label} →</a>'

    # Generate section hash for back link
    sec_hash = sec_id.lower().replace('-', '')
    date_short = format_date_display(meeting['date'])
    section_title_short = section['title'][:30] + '...' if len(section['title']) > 30 else section['title']

    # Build summary HTML with category badge
    summary_html = f'''
        <div class="summary-content">
          <span class="chapter-category" style="background: {category_color};">{category}</span>
          <p class="summary-text">{summary if summary else 'Summary not available for this section.'}</p>
        </div>
    '''

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title} | {meeting['committee']} | CB6 Transcripts</title>
  <style>{css}
    .toggle-tabs {{ display: flex; border-bottom: 1px solid var(--border); }}
    .toggle-tab {{ flex: 1; padding: 1rem; text-align: center; cursor: pointer; background: var(--bg-alt); border: none; font-size: 1rem; color: var(--text-light); }}
    .toggle-tab:hover {{ background: var(--border); }}
    .toggle-tab.active {{ background: white; color: var(--primary); font-weight: 600; }}
    .content-panel {{ padding: 1.5rem; display: none; }}
    .content-panel.active {{ display: block; }}
    .summary-content {{ }}
    .summary-content .chapter-category {{ display: inline-block; margin-bottom: 1rem; }}
    .summary-text {{ font-size: 1rem; line-height: 1.7; color: var(--text); }}
  </style>
</head>
<body data-pagefind-meta="meeting:{meeting['committee']} - {format_date_display(meeting['date'])}">
  <header>
    <div class="container">
      <nav class="breadcrumb">
        <a href="../../index.html">All Meetings</a>
        <span class="breadcrumb-sep">›</span>
        <a href="index.html">{meeting['committee']} - {date_short}</a>
        <span class="breadcrumb-sep">›</span>
        <a href="index.html#{sec_hash}">{sec_id}. {section_title_short}</a>
      </nav>
      <h1 data-pagefind-meta="title">{full_title}</h1>
      <div class="meta">{time_display}</div>
    </div>
  </header>

  <div class="container">
    <div class="video-container">
      <iframe src="{embed_url}" referrerpolicy="no-referrer-when-downgrade"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowfullscreen></iframe>
    </div>
    <p class="video-fallback">
      Video not loading? <a href="{direct_url}" target="_blank">Watch on YouTube</a>
    </p>

    <div class="content-card">
      <div class="toggle-tabs">
        <button class="toggle-tab active" onclick="showPanel('summary')">Summary</button>
        <button class="toggle-tab" onclick="showPanel('transcript')">Full Transcript</button>
      </div>
      <div id="summary" class="content-panel active">
        {summary_html}
      </div>
      <div id="transcript" class="content-panel">
        {transcript_html if transcript_html else '<p class="no-transcript">No transcript available for this section.</p>'}
      </div>
    </div>

    <div class="nav-links">
      {prev_link if prev_link else '<span></span>'}
      {next_link if next_link else '<span></span>'}
    </div>
  </div>

  <footer>
    <p>Transcripts auto-generated from <a href="{meeting['youtubeUrl']}" target="_blank">CB6 YouTube</a>.</p>
  </footer>

  <script>
    function showPanel(panelId) {{
      document.querySelectorAll('.content-panel').forEach(p => p.classList.remove('active'));
      document.querySelectorAll('.toggle-tab').forEach(t => t.classList.remove('active'));
      document.getElementById(panelId).classList.add('active');
      event.target.classList.add('active');
    }}
  </script>
</body>
</html>
'''

    with open(output_dir / f'chapter-{ch_slug}.html', 'w') as f:
        f.write(html)


def generate_transcript_only_page(data, output_dir):
    """Generate a full-transcript page for meetings without agendas.

    Shows YouTube embed at top, then full transcript with speaker labels.
    No chapter splitting — just a continuous, searchable transcript.
    """
    meeting = data['meeting']
    transcript = data['transcript']
    has_speaker_labels = data.get('has_speaker_labels', False)

    css = get_css()
    video_id = meeting['videoId']
    embed_url = f"https://www.youtube.com/embed/{video_id}"
    direct_url = f"https://www.youtube.com/watch?v={video_id}"

    # Build transcript HTML
    transcript_html = ''
    for para in transcript:
        para_time = format_time_display(para['time'])
        text = para['text']
        speaker = para.get('speaker', '')
        time_seconds = get_time_seconds(para['time'])

        if speaker and has_speaker_labels:
            speaker_class = f"speaker-{speaker.lower().replace('_', '-')}"
            transcript_html += f'''<div class="transcript-paragraph">
  <span class="speaker-label {speaker_class}">{speaker}</span>
  <a class="transcript-time" href="javascript:void(0)" onclick="seekTo({time_seconds})">{para_time}</a>
  <p>{text}</p>
</div>\n'''
        else:
            transcript_html += f'<p><a class="transcript-time" href="javascript:void(0)" onclick="seekTo({time_seconds})">{para_time}</a> {text}</p>\n'

    source_badge = ''
    if meeting.get('transcriptSource') == 'whisperx':
        source_badge = '<span class="transcript-source-badge">WhisperX + Speaker Diarization</span>'

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{meeting['committee']} - {format_date_display(meeting['date'])} | CB6 Transcripts</title>
  <style>{css}</style>
</head>
<body data-pagefind-meta="meeting:{meeting['committee']} - {format_date_display(meeting['date'])}">
  <header>
    <div class="container">
      <nav class="breadcrumb">
        <a href="../../index.html">All Meetings</a>
        <span class="breadcrumb-sep">›</span>
        <span style="color: white;">{meeting['committee']} - {format_date_display(meeting['date'])}</span>
      </nav>
      <h1 data-pagefind-meta="title">{meeting['committee']}</h1>
      <div class="meta">{format_date_display(meeting['date'])} · Manhattan Community Board 6</div>
    </div>
  </header>

  <div class="container">
    <div class="video-container">
      <iframe id="ytplayer" src="{embed_url}?enablejsapi=1" referrerpolicy="no-referrer-when-downgrade"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowfullscreen></iframe>
    </div>
    <p class="video-fallback">
      Video not loading? <a href="{direct_url}" target="_blank">Watch on YouTube</a>
    </p>

    <div class="content-card" data-pagefind-body>
      <div class="transcript-header">
        <h3>Full Transcript</h3>
        {source_badge}
      </div>
      <div class="transcript-content transcript-full">
        {transcript_html}
      </div>
    </div>
  </div>

  <footer>
    <p>Transcripts auto-generated from <a href="{meeting['youtubeUrl']}" target="_blank">CB6 YouTube</a>.</p>
  </footer>

  <script>
    function seekTo(seconds) {{
      const iframe = document.getElementById('ytplayer');
      if (iframe) {{
        iframe.src = '{embed_url}?start=' + seconds + '&autoplay=1&enablejsapi=1';
      }}
    }}
  </script>
</body>
</html>
'''

    with open(output_dir / 'index.html', 'w') as f:
        f.write(html)


def generate_home_page(meetings_data, output_dir):
    """Generate home page with meeting list."""
    css = get_css()

    by_year = {}
    for m in meetings_data:
        year = m['meeting']['date'][:4]
        if year not in by_year:
            by_year[year] = []
        by_year[year].append(m)

    html = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CB6 Meeting Transcripts | Manhattan Community Board 6</title>
  <style>''' + css + '''
  </style>
  <link href="pagefind/pagefind-ui.css" rel="stylesheet">
</head>
<body>
  <header>
    <div class="container">
      <h1>CB6 Meeting Transcripts</h1>
      <div class="meta">Manhattan Community Board 6</div>
    </div>
  </header>

  <div class="container">
    <div class="hero">
      <p>Searchable transcripts of Community Board 6 meetings. Find and share the specific discussion, vote, or public comment you care about.</p>
      <div class="search-container">
        <div id="search"></div>
      </div>
    </div>
    <script src="pagefind/pagefind-ui.js"></script>
    <script>
      window.addEventListener('DOMContentLoaded', () => {
        new PagefindUI({ element: "#search", showSubResults: true });
      });
    </script>

    <h2 class="section-heading">Latest Meetings</h2>
'''

    for year in sorted(by_year.keys(), reverse=True):
        meetings = sorted(by_year[year], key=lambda x: x['meeting']['date'], reverse=True)

        for m in meetings:
            meeting = m['meeting']
            sections = m['sections']
            chapters = m['sections_chapters']

            date_display = format_date_display(meeting['date'])
            total_chapters = sum(len(chs) for chs in chapters.values())

            # Determine meeting type
            committee = meeting['committee']
            meeting_type = 'FULL BOARD MEETING'
            if 'transportation' in committee.lower():
                meeting_type = 'COMMITTEE MEETING'
            elif 'land use' in committee.lower():
                meeting_type = 'COMMITTEE MEETING'

            # Get topics from Airtable data
            topics = meeting.get('topics', [])
            topics_html = ''
            if topics:
                topics_html = '<div class="meeting-topics"><strong>Topics:</strong> ' + ', '.join(topics) + '</div>'

            html += f'''
    <a href="meetings/{meeting['slug']}/index.html" class="meeting-card">
      <div class="meeting-header">
        <span class="meeting-type">{meeting_type}</span>
        <span class="meeting-date">{date_display}</span>
      </div>
      <h3>{committee}</h3>
      {topics_html}
    </a>
'''

    html += '''
  </div>

  <footer>
    <p>Auto-generated transcripts from CB6 YouTube. Not an official CB6 website.</p>
  </footer>
</body>
</html>
'''

    with open(output_dir / 'index.html', 'w') as f:
        f.write(html)


def get_category_color(category):
    """Get color for category badge."""
    colors = {
        'RESOLUTION': '#2563eb',
        'VOTE': '#059669',
        'REPORT': '#7c3aed',
        'REMARKS': '#64748b',
        'PRESENTATION': '#ea580c',
        'DISCUSSION': '#0891b2',
        'PUBLIC COMMENT': '#dc2626',
        'ANNOUNCEMENT': '#ca8a04',
        'BUSINESS': '#0d9488',
        'ROLL CALL': '#6366f1',
        'OPENING': '#64748b',
    }
    return colors.get(category, '#64748b')


def get_css():
    """Return shared CSS."""
    return '''
    :root {
      --primary: #1a365d;
      --primary-light: #2c5282;
      --accent: #3182ce;
      --text: #2d3748;
      --text-light: #718096;
      --bg: #ffffff;
      --bg-alt: #f7fafc;
      --border: #e2e8f0;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      color: var(--text);
      line-height: 1.7;
      background: var(--bg-alt);
    }
    .container { max-width: 900px; margin: 0 auto; padding: 2rem; }
    header { background: var(--primary); color: white; padding: 1.5rem 2rem; }
    header .container { padding: 0; }
    header .header-nav { margin-bottom: 0.5rem; }
    header .header-nav a { color: rgba(255,255,255,0.8); text-decoration: none; font-size: 0.9rem; }
    header .header-nav a:hover { color: white; }
    h1 { font-size: 1.5rem; margin-bottom: 0.25rem; font-weight: 600; }

    /* Breadcrumb navigation */
    .breadcrumb { margin-bottom: 0.75rem; font-size: 0.85rem; }
    .breadcrumb a { color: rgba(255,255,255,0.7); text-decoration: none; }
    .breadcrumb a:hover { color: white; text-decoration: underline; }
    .breadcrumb-sep { color: rgba(255,255,255,0.5); margin: 0 0.5rem; }
    .meta { opacity: 0.9; font-size: 0.95rem; }
    footer { text-align: center; padding: 2rem; color: var(--text-light); font-size: 0.9rem; }
    footer a { color: var(--accent); }

    /* Homepage styles */
    .hero { text-align: center; padding: 2rem 0; }
    .hero p { color: var(--text-light); max-width: 600px; margin: 0 auto 1.5rem; }
    .search-container { max-width: 500px; margin: 0 auto; }
    .section-heading { font-size: 1.1rem; color: var(--text-light); margin: 2rem 0 1rem; text-transform: uppercase; letter-spacing: 0.5px; }

    /* Meeting cards on homepage */
    .meeting-card {
      display: block;
      background: white;
      border-radius: 8px;
      padding: 1.25rem 1.5rem;
      margin-bottom: 1rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
      text-decoration: none;
      color: inherit;
      transition: box-shadow 0.2s;
    }
    .meeting-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
    .meeting-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
    .meeting-type { font-size: 0.7rem; font-weight: 600; color: var(--accent); text-transform: uppercase; letter-spacing: 0.5px; }
    .meeting-date { font-size: 0.85rem; color: var(--text-light); }
    .meeting-card h3 { font-size: 1.1rem; color: var(--primary); margin-bottom: 0.5rem; }
    .meeting-topics { font-size: 0.9rem; color: var(--text-light); }
    .meeting-topics strong { color: var(--text); }

    /* Meeting page - agenda sections */
    .meeting-summary { background: white; padding: 1rem 1.5rem; border-radius: 8px; margin-bottom: 1.5rem; }
    .agenda-section { background: white; border-radius: 8px; margin-bottom: 0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow: hidden; }
    .agenda-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 1rem 1.5rem;
      cursor: pointer;
      background: var(--bg-alt);
      border-bottom: 1px solid transparent;
    }
    .agenda-section.expanded .agenda-header { border-bottom-color: var(--border); }
    .agenda-header:hover { background: #edf2f7; }
    .agenda-header-left { display: flex; align-items: center; gap: 1rem; }
    .agenda-header-right { display: flex; align-items: center; gap: 1rem; }
    .section-type { font-size: 0.65rem; font-weight: 600; color: var(--accent); background: #ebf8ff; padding: 0.2rem 0.5rem; border-radius: 4px; text-transform: uppercase; }
    .agenda-header h3 { font-size: 1rem; font-weight: 600; color: var(--primary); }
    .chapter-count { font-size: 0.8rem; color: var(--text-light); }
    .timestamp { font-size: 0.8rem; color: var(--accent); font-family: monospace; }
    .expand-icon { font-size: 0.7rem; color: var(--text-light); transition: transform 0.2s; }
    .agenda-section.expanded .expand-icon { transform: rotate(180deg); }

    .agenda-chapters { display: none; padding: 0; }
    .agenda-chapters.expanded { display: block; padding: 0.5rem 1rem 1rem; }

    /* Agenda items (from Airtable) */
    .agenda-item {
      display: flex;
      align-items: flex-start;
      padding: 0.6rem 0.75rem;
      margin-bottom: 0.25rem;
      background: var(--bg-alt);
      border-radius: 6px;
      text-decoration: none;
      color: inherit;
      transition: background 0.2s;
      gap: 0.5rem;
    }
    .agenda-item:hover { background: #e2e8f0; }
    .agenda-item:last-child { margin-bottom: 0; }
    .agenda-item-letter {
      font-weight: 600;
      color: var(--accent);
      min-width: 1.5rem;
      flex-shrink: 0;
    }
    .agenda-item-title {
      flex: 1;
      color: var(--text);
      line-height: 1.4;
    }
    .agenda-item-time {
      font-size: 0.8rem;
      color: var(--text-light);
      font-family: monospace;
      flex-shrink: 0;
    }
    .no-items {
      padding: 0.75rem;
      color: var(--text-light);
      font-style: italic;
    }

    /* Chapter cards inside agenda sections */
    .chapter-card {
      display: block;
      background: var(--bg-alt);
      border-radius: 8px;
      padding: 1rem 1.25rem;
      margin-bottom: 0.75rem;
      border-left: 4px solid var(--accent);
      text-decoration: none;
      color: inherit;
      transition: background 0.2s;
    }
    .chapter-card:hover { background: #edf2f7; }
    .chapter-card:last-child { margin-bottom: 0; }
    .chapter-meta { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
    .chapter-category { font-size: 0.65rem; font-weight: 600; color: white; padding: 0.15rem 0.4rem; border-radius: 3px; text-transform: uppercase; }
    .chapter-timing { font-size: 0.8rem; color: var(--text-light); font-family: monospace; }
    .chapter-title { font-size: 0.95rem; font-weight: 600; color: var(--primary); margin-bottom: 0.4rem; }
    .chapter-desc { font-size: 0.85rem; color: var(--text); line-height: 1.5; margin: 0; }
    .no-chapters { padding: 1rem; color: var(--text-light); font-style: italic; }

    /* Chapter detail page */
    .chapter-category-header { display: inline-block; font-size: 0.7rem; font-weight: 600; color: white; padding: 0.2rem 0.6rem; border-radius: 4px; margin-bottom: 0.5rem; text-transform: uppercase; }
    .video-container { background: #000; border-radius: 8px; overflow: hidden; margin-bottom: 0.5rem; aspect-ratio: 16/9; }
    .video-container iframe { width: 100%; height: 100%; border: none; }
    .video-fallback { text-align: center; margin-bottom: 1.5rem; font-size: 0.9rem; color: var(--text-light); }
    .video-fallback a { color: var(--accent); }

    .content-card { background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow: hidden; }
    .transcript-header { padding: 1rem 1.5rem; background: var(--bg-alt); border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
    .transcript-header h3 { font-size: 1rem; color: var(--primary); margin: 0; }
    .transcript-header .transcript-note { font-size: 0.8rem; color: var(--text-light); font-style: italic; margin: 0; }
    .transcript-content { padding: 1.5rem; max-height: 500px; overflow-y: auto; }
    .transcript-content p { margin-bottom: 1rem; line-height: 1.6; }
    .transcript-time { font-family: monospace; font-size: 0.8rem; color: var(--accent); margin-right: 0.5rem; }
    .no-transcript { color: var(--text-light); font-style: italic; }

    .nav-links { display: flex; justify-content: space-between; margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid var(--border); }
    .nav-links a { color: var(--accent); text-decoration: none; font-size: 0.9rem; }
    .nav-links a:hover { text-decoration: underline; }

    /* Speaker labels (WhisperX diarization) */
    .transcript-paragraph { margin-bottom: 1rem; }
    .transcript-paragraph p { margin: 0.25rem 0 0 0; line-height: 1.6; }
    .speaker-label {
      display: inline-block;
      font-size: 0.7rem;
      font-weight: 600;
      color: white;
      padding: 0.1rem 0.5rem;
      border-radius: 3px;
      margin-right: 0.5rem;
      vertical-align: middle;
      background: #64748b;
    }
    /* Distinct colors per speaker */
    .speaker-speaker-00 { background: #2563eb; }
    .speaker-speaker-01 { background: #059669; }
    .speaker-speaker-02 { background: #dc2626; }
    .speaker-speaker-03 { background: #7c3aed; }
    .speaker-speaker-04 { background: #ea580c; }
    .speaker-speaker-05 { background: #0891b2; }
    .speaker-speaker-06 { background: #ca8a04; }
    .speaker-speaker-07 { background: #be185d; }
    .speaker-speaker-08 { background: #4f46e5; }
    .speaker-speaker-09 { background: #0d9488; }

    /* Transcript source badge */
    .transcript-source-badge {
      font-size: 0.75rem;
      color: var(--accent);
      background: #ebf8ff;
      padding: 0.2rem 0.6rem;
      border-radius: 4px;
      font-weight: 500;
    }

    /* Full transcript view (no height limit) */
    .transcript-full { max-height: none; overflow: visible; }
'''


def main():
    project_root = Path(__file__).parent.parent

    with open(project_root / 'src' / '_data' / 'meetings.json') as f:
        all_meetings = json.load(f)

    officials_path = project_root / 'data' / 'elected_officials.json'
    misspellings = {}
    if officials_path.exists():
        with open(officials_path) as f:
            officials_data = json.load(f)
            misspellings = officials_data.get('common_misspellings', {})

    meetings = [m for m in all_meetings if m.get('videoId')]

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    print(f"Building site with {limit} meetings")

    output_dir = project_root / 'dist'
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'meetings').mkdir(exist_ok=True)

    processed = []
    for i, meeting in enumerate(meetings[:limit]):
        print(f"\n[{i+1}/{limit}] Processing: {meeting['date']} - {meeting['committee']}")

        data = process_meeting(meeting, project_root, misspellings)
        if data:
            meeting_dir = output_dir / 'meetings' / meeting['slug']
            generate_meeting_pages(data, meeting_dir)
            processed.append(data)

            total_chapters = sum(len(chs) for chs in data['sections_chapters'].values())
            print(f"    Generated {len(data['sections'])} sections, {total_chapters} chapters")
        else:
            print(f"    Skipped (no captions)")

    print(f"\nGenerating home page...")
    generate_home_page(processed, output_dir)

    print(f"\nDone! Site built in {output_dir}")
    print(f"Open {output_dir / 'index.html'} to view")


if __name__ == '__main__':
    main()

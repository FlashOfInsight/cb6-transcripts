#!/usr/bin/env python3
"""
Build the full CB6 Transcripts site.
1. Download captions for meetings that need them
2. Process transcripts (parse, segment, correct names)
3. Generate meeting pages
4. Create home page
"""

import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path


def call_ollama(prompt, model="llama3.2", timeout=180):
    """Call Ollama API with a prompt."""
    try:
        data = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False
        }).encode('utf-8')

        req = urllib.request.Request(
            'http://localhost:11434/api/generate',
            data=data,
            headers={'Content-Type': 'application/json'}
        )

        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get('response', '').strip()

    except Exception as e:
        return None


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

        # yt-dlp names it with .en.vtt
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


def clean_filler_words(text):
    """Remove filler words like um, uh, etc."""
    # Remove standalone filler words
    text = re.sub(r'\b[Uu]m+\b[\s,]*', '', text)
    text = re.sub(r'\b[Uu]h+\b[\s,]*', '', text)
    text = re.sub(r'\b[Ee]r+\b[\s,]*', '', text)
    text = re.sub(r'\b[Aa]h+\b[\s,]*', '', text)
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([,.])', r'\1', text)
    return text.strip()


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
            new_text = clean_filler_words(new_text)
            if new_text:  # Only add if there's content after cleaning
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
            current = {
                'time': seg['time'],
                'text': seg['text'],
                'speaker': '[SPEAKER]'
            }
        else:
            current['text'] += ' ' + seg['text']

    if current:
        paragraphs.append(current)

    return paragraphs


def parse_agenda(agenda_text):
    """Parse agenda into structured sections."""
    sections = []
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
                sections.append({
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
                sections.append({
                    'id': current_section['id'],
                    'title': current_section['title'],
                    'items': current_items
                })
            # Find the parent section for committee reports
            parent = 'VI'
            for s in sections:
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
        sections.append({
            'id': current_section['id'],
            'title': current_section['title'],
            'items': current_items
        })

    return sections


def find_section_boundaries(transcript, sections):
    """Find where each section starts based on content."""
    boundaries = {}
    section_ids = [s['id'] for s in sections]

    for i, para in enumerate(transcript):
        text = para['text']
        text_lower = text.lower()

        # Call to Order
        if i < 15 and 'called to order' in text_lower and 'I' not in boundaries:
            boundaries['I'] = i

        # Roll Call
        if i < 30 and ('howdy' in text_lower or 'roll call' in text_lower) and 'III' not in boundaries:
            if 'IV' not in boundaries:
                boundaries['III'] = i

        # Public Session
        if i > 10 and i < 200:
            if ('borough president' in text_lower or 'council member' in text_lower or
                'assembly member' in text_lower) and 'IV' not in boundaries:
                if 'we have' in text_lower or 'next' in text_lower:
                    boundaries['IV'] = i

        # Business Session
        if i > 50:
            if ("chair's report" in text_lower or 'minutes from the' in text_lower) and 'V' not in boundaries:
                boundaries['V'] = i

        # Committee Reports header
        if 'committee reports' in text_lower and 'VI' not in boundaries:
            boundaries['VI'] = i

        # Committee sub-sections - look for specific phrases
        if 'VI' in boundaries and i > boundaries.get('VI', 0):
            # Health & Education (VI-1)
            if 'VI-1' not in boundaries:
                if ('first committee' in text_lower and 'health' in text_lower) or \
                   ('health and education' in text_lower) or \
                   ('health' in text_lower and 'education' in text_lower and 'rupal' in text_lower):
                    boundaries['VI-1'] = i

            # Transportation (VI-2)
            if 'VI-2' not in boundaries and i > boundaries.get('VI-1', boundaries.get('VI', 0)):
                if ('transportation' in text_lower and 'jason' in text_lower) or \
                   ('next' in text_lower and 'transportation' in text_lower):
                    boundaries['VI-2'] = i

            # Business Affairs (VI-3)
            if 'VI-3' not in boundaries and i > boundaries.get('VI-2', boundaries.get('VI-1', 0)):
                if ('business affairs' in text_lower) or \
                   ('we have business' in text_lower) or \
                   ('licensing' in text_lower and 'anton' in text_lower):
                    boundaries['VI-3'] = i

            # Public Safety (VI-4)
            if 'VI-4' not in boundaries and i > boundaries.get('VI-3', boundaries.get('VI-2', 0)):
                if ('public safety' in text_lower and 'sanitation' in text_lower) or \
                   ('next' in text_lower and 'public safety' in text_lower) or \
                   ('safety' in text_lower and ('stu' in text_lower or 'desser' in text_lower)):
                    boundaries['VI-4'] = i

            # Parks & Landmarks (VI-5)
            if 'VI-5' not in boundaries and i > boundaries.get('VI-4', boundaries.get('VI-3', 0)):
                if ('parks' in text_lower and 'landmarks' in text_lower) or \
                   ('next' in text_lower and 'parks' in text_lower) or \
                   ('neil' in text_lower and 'barclay' in text_lower):
                    boundaries['VI-5'] = i

            # Land Use (VI-6)
            if 'VI-6' not in boundaries and i > boundaries.get('VI-5', boundaries.get('VI-4', 0)):
                if ('land use' in text_lower and 'waterfront' in text_lower) or \
                   ('land' in text_lower and 'waterfront' in text_lower) or \
                   ('next' in text_lower and 'land use' in text_lower):
                    boundaries['VI-6'] = i

            # Budget (VI-7) - must come after VI-6
            if 'VI-7' not in boundaries and 'VI-6' in boundaries and i > boundaries.get('VI-6', 0):
                if ('budget' in text_lower and 'government' in text_lower) or \
                   ('budget' in text_lower and 'affairs' in text_lower) or \
                   ('steve' in text_lower and 'perez' in text_lower):
                    boundaries['VI-7'] = i

            # Housing (VI-8) - must come after VI-7
            if 'VI-8' not in boundaries and 'VI-7' in boundaries and i > boundaries.get('VI-7', 0):
                if ('housing' in text_lower and 'homelessness' in text_lower) or \
                   ('alek' in text_lower) or ('milit' in text_lower):
                    boundaries['VI-8'] = i

        # Adjournment
        if 'adjourned' in text_lower and i > len(transcript) - 50:
            if 'IX' in section_ids and 'IX' not in boundaries:
                boundaries['IX'] = i
            elif 'X' in section_ids and 'X' not in boundaries:
                boundaries['X'] = i

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


def correct_names(transcript, misspellings):
    """Apply name corrections to transcript."""
    for para in transcript:
        text = para['text']
        for wrong, right in misspellings.items():
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            text = pattern.sub(right, text)
        para['text'] = text
    return transcript


def generate_summary(paragraphs, section_title, agenda_items=None):
    """Generate chapter-based summary like citymeetings.nyc format."""
    if not paragraphs:
        return []

    first_time = paragraphs[0]['time'] if paragraphs else "00:00:00"

    # Use smart extraction that doesn't rely on LLM
    chapters = extract_smart_chapters(paragraphs, section_title, first_time, agenda_items)

    return chapters


def extract_smart_chapters(paragraphs, section_title, first_time, agenda_items=None):
    """Extract structured chapters from transcript using pattern matching."""
    chapters = []
    all_text = ' '.join(p['text'] for p in paragraphs)
    text_lower = all_text.lower()

    # Track which paragraph each chapter comes from for timestamps
    def get_time_for_index(idx):
        if idx < len(paragraphs):
            return paragraphs[idx]['time']
        return first_time

    # 1. Look for resolutions and votes
    resolution_keywords = {
        'bellevue': 'Bellevue Hospital funding resolution',
        'helicopter': 'Helicopter tour ban resolution',
        'bike lane': 'Third Avenue bike lane configuration',
        'mental health': 'Mental health diversion bill',
        'diversion': 'Criminal justice diversion program',
        'cannabis': 'Cannabis licensing application',
        'liquor license': 'Liquor license application',
        'treatment court': 'Treatment court expansion',
        'housing': 'Housing policy resolution',
        'rent': 'Rent and affordability discussion',
        'coned': 'Con Edison rate increase opposition',
        'utility': 'Utility rate hike concerns',
    }

    found_topics = set()
    for keyword, title in resolution_keywords.items():
        if keyword in text_lower and keyword not in found_topics:
            idx = text_lower.find(keyword)
            # Find the paragraph containing this keyword
            char_count = 0
            para_idx = 0
            for i, p in enumerate(paragraphs):
                if char_count + len(p['text']) >= idx:
                    para_idx = i
                    break
                char_count += len(p['text']) + 1

            # Extract context around the keyword
            start = max(0, idx - 30)
            end = min(len(all_text), idx + 250)
            context = all_text[start:end].strip()

            # Determine category
            category = 'DISCUSSION'
            if 'vote' in text_lower[max(0,idx-100):idx+100] or 'resolution' in text_lower[max(0,idx-100):idx+100]:
                category = 'RESOLUTION'

            chapters.append({
                "category": category,
                "title": title,
                "description": context[:280] + "..." if len(context) > 280 else context,
                "time": get_time_for_index(para_idx)
            })
            found_topics.add(keyword)

    # 2. Look for elected officials and their reports
    officials = [
        ('borough president', 'Borough President', 'REPORT'),
        ('council member harvey', 'Council Member Harvey Epstein', 'REPORT'),
        ('council member virginia', 'Council Member Virginia Maloney', 'REPORT'),
        ('assembly member', 'Assembly Member', 'REPORT'),
        ('senator gonzalez', 'Senator Gonzalez', 'REPORT'),
        ('senator krueger', 'Senator Krueger', 'REPORT'),
        ('district attorney', 'District Attorney', 'REPORT'),
        ('controller', 'City Controller', 'REPORT'),
    ]

    for keyword, title, category in officials:
        if keyword in text_lower:
            idx = text_lower.find(keyword)
            # Find paragraph
            char_count = 0
            para_idx = 0
            for i, p in enumerate(paragraphs):
                if char_count + len(p['text']) >= idx:
                    para_idx = i
                    break
                char_count += len(p['text']) + 1

            # Get context
            start = max(0, idx - 20)
            end = min(len(all_text), idx + 200)
            context = all_text[start:end].strip()

            # Check if this is a speaker introduction or report
            local_text = text_lower[max(0,idx-50):idx+100]
            if 'next we have' in local_text or 'now we have' in local_text or 'introduce' in local_text:
                desc = f"{title} addresses the board. {context}"
            else:
                desc = context

            chapters.append({
                "category": category,
                "title": f"{title} remarks",
                "description": desc[:280] + "..." if len(desc) > 280 else desc,
                "time": get_time_for_index(para_idx)
            })

    # 3. Look for public speakers
    if 'public session' in section_title.lower() or 'public comment' in text_lower:
        speaker_pattern = re.compile(r'(next|first)\s+(on the list|speaker|we have)\s+(?:is\s+)?([A-Z][a-z]+\s+[A-Z][a-z]+)', re.IGNORECASE)
        for match in speaker_pattern.finditer(all_text[:3000]):
            name = match.group(3)
            idx = match.start()
            # Get context after the speaker name
            context = all_text[idx:idx+250].strip()
            chapters.append({
                "category": "PUBLIC COMMENT",
                "title": f"Public comment: {name}",
                "description": context[:280] + "..." if len(context) > 280 else context,
                "time": first_time
            })

    # 4. Look for meeting procedural items
    procedural_items = [
        ('called to order', 'Meeting called to order', 'ANNOUNCEMENT'),
        ('adopt the agenda', 'Agenda adoption', 'REMARKS'),
        ('roll call', 'Roll call / Attendance', 'REPORT'),
        ('take attendance', 'Attendance taken', 'REPORT'),
        ('quorum', 'Quorum established', 'REMARKS'),
        ('ground rules', 'Meeting ground rules explained', 'REMARKS'),
        ('public speaker', 'Public speaker registration', 'ANNOUNCEMENT'),
        ('adjourned', 'Meeting adjourned', 'ANNOUNCEMENT'),
    ]

    for keyword, title, category in procedural_items:
        if keyword in text_lower:
            idx = text_lower.find(keyword)
            char_count = 0
            para_idx = 0
            for i, p in enumerate(paragraphs):
                if char_count + len(p['text']) >= idx:
                    para_idx = i
                    break
                char_count += len(p['text']) + 1

            context = all_text[max(0, idx-20):min(len(all_text), idx+200)].strip()
            chapters.append({
                "category": category,
                "title": title,
                "description": context[:280] + "..." if len(context) > 280 else context,
                "time": get_time_for_index(para_idx)
            })

    # 5. Look for questions and discussion
    if 'question' in text_lower:
        question_pattern = re.compile(r'([A-Z][a-z]+)\s+has\s+a\s+question', re.IGNORECASE)
        for match in question_pattern.finditer(all_text[:5000]):
            name = match.group(1)
            idx = match.start()
            context = all_text[idx:idx+200].strip()
            chapters.append({
                "category": "DISCUSSION",
                "title": f"Question from {name}",
                "description": context[:250] + "..." if len(context) > 250 else context,
                "time": first_time
            })

    # 6. If we have agenda items but no chapters, use them
    if not chapters and agenda_items:
        for item in agenda_items[:4]:
            item_text = item.get('text', '')
            if item_text:
                chapters.append({
                    "category": "DISCUSSION",
                    "title": item_text[:80],
                    "description": f"Discussion of agenda item: {item_text}",
                    "time": first_time
                })

    # 7. Fallback: create basic summary from first paragraphs
    if not chapters:
        # Use first 2-3 paragraphs as overview
        first_paras = ' '.join(p['text'] for p in paragraphs[:3])
        sentences = re.split(r'[.!?]+', first_paras)
        meaningful = [s.strip() for s in sentences if len(s.strip()) > 30][:4]

        if meaningful:
            chapters.append({
                "category": "REMARKS",
                "title": section_title,
                "description": '. '.join(meaningful)[:350] + "...",
                "time": first_time
            })
        else:
            chapters.append({
                "category": "REMARKS",
                "title": section_title,
                "description": first_paras[:350] + "...",
                "time": first_time
            })

    # Remove duplicates and limit to 8 chapters
    seen_titles = set()
    unique_chapters = []
    for ch in chapters:
        title_key = ch['title'].lower()[:30]
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_chapters.append(ch)

    return unique_chapters[:8]


def process_meeting(meeting, project_root, misspellings, use_llm=True):
    """Process a single meeting."""
    video_id = meeting.get('videoId')
    if not video_id:
        return None

    transcripts_dir = project_root / 'transcripts'
    transcripts_dir.mkdir(exist_ok=True)

    # Download captions if needed
    vtt_path = transcripts_dir / f"{video_id}.vtt"
    if not vtt_path.exists():
        print(f"    Downloading captions...")
        vtt_path = download_captions(video_id, transcripts_dir)
        if not vtt_path:
            return None

    # Parse transcript
    segments = clean_vtt_with_timestamps(vtt_path)
    paragraphs = consolidate_segments(segments)

    if not paragraphs:
        return None

    # Parse agenda
    sections = parse_agenda(meeting.get('agenda', ''))
    if not sections:
        sections = [{'id': 'I', 'title': 'Meeting', 'items': []}]

    # Find section boundaries and apply
    boundaries = find_section_boundaries(paragraphs, sections)
    paragraphs = apply_sections(paragraphs, boundaries, sections)

    # Correct names
    paragraphs = correct_names(paragraphs, misspellings)

    return {
        'meeting': {
            'slug': meeting['slug'],
            'date': meeting['date'],
            'committee': meeting['committee'],
            'youtubeUrl': meeting['youtubeUrl'],
            'topics': meeting.get('topics', [])
        },
        'sections': sections,
        'transcript': paragraphs,
        'section_boundaries': boundaries
    }


def format_time_display(time_str):
    """Format time for display."""
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"
    return time_str


def format_date_display(date_str):
    """Format date for display (e.g., 'January 14, 2026')."""
    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    try:
        parts = date_str.split('-')
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
        return f"{months[month-1]} {day}, {year}"
    except:
        return date_str


def format_summary_html(summary, video_id=None, start_seconds=0):
    """Convert chapter-based summary to HTML like citymeetings.nyc."""
    # Handle new list-of-chapters format
    if isinstance(summary, list):
        if not summary:
            return '<p>No summary available.</p>'

        html = '<div class="chapters-list">'
        for chapter in summary:
            category = chapter.get('category', 'REMARKS')
            title = chapter.get('title', '')
            description = chapter.get('description', '')
            time = chapter.get('time', '00:00:00')

            # Format time for display
            time_display = format_time_display(time)
            time_seconds = get_time_seconds(time)

            # Create YouTube timestamp link
            if video_id:
                time_link = f'https://www.youtube.com/watch?v={video_id}&t={time_seconds}'
                time_html = f'<a href="{time_link}" target="_blank" class="chapter-time">{time_display}</a>'
            else:
                time_html = f'<span class="chapter-time">{time_display}</span>'

            # Category color mapping
            category_colors = {
                'RESOLUTION': '#2563eb',  # blue
                'VOTE': '#059669',        # green
                'REPORT': '#7c3aed',      # purple
                'REMARKS': '#64748b',     # gray
                'PRESENTATION': '#ea580c', # orange
                'DISCUSSION': '#0891b2',   # cyan
                'PUBLIC COMMENT': '#dc2626', # red
                'ANNOUNCEMENT': '#ca8a04',  # yellow
            }
            color = category_colors.get(category, '#64748b')

            html += f'''
        <div class="chapter-card">
          <div class="chapter-header">
            <span class="chapter-category" style="background: {color};">{category}</span>
            {time_html}
          </div>
          <h3 class="chapter-title">{title}</h3>
          <p class="chapter-description">{description}</p>
        </div>'''

        html += '\n      </div>'
        return html

    # Legacy fallback for string summaries
    if isinstance(summary, str):
        if '•' in summary:
            lines = [line.strip() for line in summary.split('\n') if line.strip()]
            items = [line.lstrip('•').strip() for line in lines if line.startswith('•')]
            if items:
                return '<ul class="summary-list">' + ''.join(f'<li>{item}</li>' for item in items) + '</ul>'
        return f'<p>{summary}</p>'

    return '<p>No summary available.</p>'


def get_time_seconds(time_str):
    """Convert time string to seconds."""
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        return h * 3600 + m * 60 + s
    return 0


def generate_meeting_pages(processed_data, output_dir, use_llm=True):
    """Generate all pages for a meeting."""
    meeting = processed_data['meeting']
    sections = processed_data['sections']
    transcript = processed_data['transcript']

    output_dir.mkdir(parents=True, exist_ok=True)

    # Group transcript by section
    sections_content = {}
    for para in transcript:
        sec_id = para.get('section', 'I')
        if sec_id not in sections_content:
            sections_content[sec_id] = []
        sections_content[sec_id].append(para)

    # For committee sub-sections without content, try to use parent section content
    for section in sections:
        sec_id = section['id']
        if sec_id.startswith('VI-') and sec_id not in sections_content:
            # Use section VI content as fallback, filtered by committee keywords
            parent_content = sections_content.get('VI', [])
            if parent_content:
                # Try to find paragraphs mentioning this committee
                title_lower = section['title'].lower()
                keywords = []
                if 'health' in title_lower:
                    keywords = ['health', 'education', 'bellevue', 'hospital']
                elif 'transportation' in title_lower:
                    keywords = ['transportation', 'bike', 'lane', 'speed']
                elif 'business' in title_lower:
                    keywords = ['business', 'license', 'liquor', 'sidewalk cafe', 'cannabis']
                elif 'safety' in title_lower:
                    keywords = ['safety', 'sanitation', 'police', 'treatment court']
                elif 'parks' in title_lower:
                    keywords = ['parks', 'landmarks', 'cultural']
                elif 'land use' in title_lower:
                    keywords = ['land use', 'waterfront', 'zoning']
                elif 'budget' in title_lower:
                    keywords = ['budget', 'governmental']
                elif 'housing' in title_lower:
                    keywords = ['housing', 'homelessness', 'shelter']

                # Find relevant paragraphs
                relevant = []
                for para in parent_content:
                    text_lower = para['text'].lower()
                    if any(kw in text_lower for kw in keywords):
                        relevant.append(para)

                if relevant:
                    sections_content[sec_id] = relevant
                elif parent_content:
                    # Fallback: use first few paragraphs of parent
                    sections_content[sec_id] = parent_content[:5]

    # Generate index page
    generate_meeting_index(meeting, sections, sections_content, output_dir)

    # Generate section pages for ALL sections
    for i, section in enumerate(sections):
        paragraphs = sections_content.get(section['id'], [])
        # Generate page even if no paragraphs (will show placeholder)
        generate_section_page(meeting, section, i, sections, paragraphs, sections_content, output_dir, use_llm)


def generate_meeting_index(meeting, sections, sections_content, output_dir):
    """Generate meeting index/agenda page."""
    css = get_css()

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{meeting['committee']} Meeting - {meeting['date']} | CB6 Transcripts</title>
  <style>{css}</style>
</head>
<body>
  <header>
    <div class="container">
      <div class="header-nav"><a href="../../index.html">← All Meetings</a></div>
      <h1>{meeting['committee']} Meeting</h1>
      <div class="meta">Manhattan Community Board 6 · {meeting['date']}</div>
    </div>
  </header>

  <div class="container">
    <p style="margin-bottom: 1.5rem; color: var(--text-light);">
      Click on any section to view the video and transcript.
    </p>
'''

    for sec in sections:
        sec_id = sec['id']
        title = sec['title']
        items = sec.get('items', [])
        paragraphs = sections_content.get(sec_id, [])

        first_time = paragraphs[0]['time'] if paragraphs else '00:00:00'
        time_display = format_time_display(first_time)
        sec_slug = sec_id.lower().replace('-', '')

        html += f'''    <div class="agenda-section">
      <div class="agenda-section-header">
        <h2><a href="section-{sec_slug}.html">{sec_id}. {title}</a></h2>
        <span class="timestamp-badge">{time_display}</span>
      </div>
'''

        if items:
            html += '''      <ul class="agenda-items">
'''
            for item in items:
                html += f'''        <li><strong>{item.get('letter', '')})</strong> {item.get('text', '')}</li>
'''
            html += '''      </ul>
'''

        html += '''    </div>
'''

    html += f'''  </div>
  <footer>
    <p>Transcripts auto-generated from <a href="{meeting['youtubeUrl']}" target="_blank">CB6 YouTube</a>.</p>
  </footer>
</body>
</html>
'''

    with open(output_dir / 'index.html', 'w') as f:
        f.write(html)


def generate_section_page(meeting, section, section_index, all_sections, paragraphs, sections_content, output_dir, use_llm):
    """Generate a section detail page."""
    css = get_css()
    sec_id = section['id']
    title = section['title']
    sec_slug = sec_id.lower().replace('-', '')

    # Handle empty paragraphs
    if paragraphs:
        first_time = paragraphs[0]['time']
    else:
        # Estimate time based on section position or use parent section time
        first_time = '00:00:00'
        if sec_id.startswith('VI-'):
            parent_content = sections_content.get('VI', [])
            if parent_content:
                first_time = parent_content[0]['time']

    start_seconds = get_time_seconds(first_time)
    time_display = format_time_display(first_time)

    # YouTube embed
    if 'youtu.be/' in meeting['youtubeUrl']:
        video_id = meeting['youtubeUrl'].split('youtu.be/')[-1].split('?')[0]
    else:
        video_id = meeting['youtubeUrl'].split('watch?v=')[-1].split('&')[0]

    embed_url = f"https://www.youtube.com/embed/{video_id}?start={start_seconds}"
    direct_url = f"https://www.youtube.com/watch?v={video_id}&t={start_seconds}"

    # Generate summary - always use smart extraction (use_llm flag ignored since pattern matching works better)
    agenda_items = section.get('items', [])
    if paragraphs:
        summary = generate_summary(paragraphs, title, agenda_items)
    else:
        # No transcript content - create summary from agenda items
        if agenda_items:
            summary = [{"category": "REMARKS", "title": title,
                       "description": "Agenda items: " + "; ".join(item.get('text', '') for item in agenda_items[:3]),
                       "time": first_time}]
        else:
            summary = [{"category": "REMARKS", "title": title,
                       "description": "Transcript not yet available for this section.",
                       "time": first_time}]

    # Format transcript
    transcript_html = ''
    if paragraphs:
        current_para = []
        for i, para in enumerate(paragraphs):
            current_para.append(para['text'])
            if len(' '.join(current_para)) > 500 or i % 6 == 5:
                text = ' '.join(current_para)
                text = re.sub(r'\s+', ' ', text.strip())
                transcript_html += f'        <p>{text}</p>\n'
                current_para = []
        if current_para:
            text = ' '.join(current_para)
            text = re.sub(r'\s+', ' ', text.strip())
            transcript_html += f'        <p>{text}</p>\n'
    else:
        transcript_html = '        <p><em>Transcript not yet segmented for this section. See the parent section for full transcript.</em></p>\n'

    # Prev/next nav - link to all sections, not just those with content
    prev_link = ''
    next_link = ''
    for i, sec in enumerate(all_sections):
        sec_slug_nav = sec['id'].lower().replace('-', '')
        if i < section_index:
            prev_link = f'<a href="section-{sec_slug_nav}.html">← {sec["id"]}. {sec["title"][:25]}...</a>'
        elif i > section_index and not next_link:
            next_link = f'<a href="section-{sec_slug_nav}.html">{sec["id"]}. {sec["title"][:25]}... →</a>'

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{sec_id}. {title} | {meeting['committee']} - {meeting['date']}</title>
  <style>{css}</style>
</head>
<body data-pagefind-meta="meeting:{meeting['committee']} - {format_date_display(meeting['date'])}">
  <header>
    <div class="container">
      <h1 data-pagefind-meta="title">{sec_id}. {title}</h1>
      <div class="meta">{meeting['committee']} · {meeting['date']} · {time_display}</div>
    </div>
  </header>

  <div class="breadcrumb">
    <a href="index.html">← Back to Agenda</a>
  </div>

  <div class="container">
    <div class="video-container">
      <iframe src="{embed_url}" referrerpolicy="no-referrer-when-downgrade"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowfullscreen></iframe>
    </div>
    <p style="text-align: center; margin-bottom: 1.5rem; font-size: 0.9rem; color: var(--text-light);">
      Video not loading? <a href="{direct_url}" target="_blank">Watch on YouTube</a>
    </p>

    <div class="content-card">
      <div class="toggle-tabs">
        <button class="toggle-tab active" onclick="showPanel('summary')">Summary</button>
        <button class="toggle-tab" onclick="showPanel('transcript')">Full Transcript</button>
      </div>
      <div id="summary" class="content-panel active">
        <div class="summary-text">{format_summary_html(summary, video_id, start_seconds)}</div>
      </div>
      <div id="transcript" class="content-panel">
{transcript_html}      </div>
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

    with open(output_dir / f'section-{sec_slug}.html', 'w') as f:
        f.write(html)


def generate_home_page(meetings_data, output_dir):
    """Generate the main home page listing all meetings."""
    css = get_css()

    # Group by year
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
    .meeting-card {
      background: white;
      border-radius: 8px;
      padding: 1rem 1.5rem;
      margin-bottom: 0.75rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .meeting-card h3 {
      font-size: 1rem;
      margin: 0;
    }
    .meeting-card h3 a {
      color: var(--primary);
      text-decoration: none;
    }
    .meeting-card h3 a:hover {
      color: var(--accent);
      text-decoration: underline;
    }
    .meeting-card .date {
      color: var(--text-light);
      font-size: 0.9rem;
    }
    .year-section {
      margin-bottom: 2rem;
    }
    .year-section h2 {
      font-size: 1.25rem;
      margin-bottom: 1rem;
      color: var(--primary);
    }
    .hero {
      text-align: center;
      padding: 2rem 0;
    }
    .hero p {
      color: var(--text-light);
      max-width: 600px;
      margin: 0 auto;
    }
    .search-container { max-width: 500px; margin: 1.5rem auto; }
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
      <p>Searchable transcripts of Community Board 6 meetings. Click any meeting to view the agenda, video, and full transcript.</p>
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
'''

    for year in sorted(by_year.keys(), reverse=True):
        meetings = sorted(by_year[year], key=lambda x: x['meeting']['date'], reverse=True)
        html += f'''    <div class="year-section">
      <h2>{year}</h2>
'''
        for m in meetings:
            meeting = m['meeting']
            date_display = format_date_display(meeting['date'])
            html += f'''      <div class="meeting-card">
        <div>
          <h3><a href="meetings/{meeting['slug']}/index.html">{meeting['committee']}</a></h3>
          <span class="date">{date_display}</span>
        </div>
      </div>
'''
        html += '''    </div>
'''

    html += '''  </div>
  <footer>
    <p>Auto-generated transcripts from CB6 YouTube. Not an official CB6 website.</p>
  </footer>
</body>
</html>
'''

    with open(output_dir / 'index.html', 'w') as f:
        f.write(html)


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
    .meta { opacity: 0.9; font-size: 0.95rem; }
    .breadcrumb { padding: 1rem 2rem; background: var(--bg); border-bottom: 1px solid var(--border); font-size: 0.9rem; }
    .breadcrumb a { color: var(--accent); text-decoration: none; }
    .agenda-section { background: white; border-radius: 8px; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow: hidden; }
    .agenda-section-header { padding: 1rem 1.5rem; background: var(--bg-alt); border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
    .agenda-section-header h2 { font-size: 1.1rem; font-weight: 600; color: var(--primary); }
    .agenda-section-header h2 a { color: var(--primary); text-decoration: none; }
    .agenda-section-header h2 a:hover { color: var(--accent); text-decoration: underline; }
    .agenda-items { padding: 1rem 1.5rem; list-style: none; }
    .agenda-items li { padding: 0.5rem 0; border-bottom: 1px solid var(--border); }
    .agenda-items li:last-child { border-bottom: none; }
    .timestamp-badge { font-size: 0.8rem; color: var(--text-light); background: var(--bg-alt); padding: 0.2rem 0.5rem; border-radius: 4px; }
    .video-container { background: #000; border-radius: 8px; overflow: hidden; margin-bottom: 0.5rem; aspect-ratio: 16/9; }
    .video-container iframe { width: 100%; height: 100%; border: none; }
    .content-card { background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow: hidden; }
    .toggle-tabs { display: flex; border-bottom: 1px solid var(--border); }
    .toggle-tab { flex: 1; padding: 1rem; text-align: center; cursor: pointer; background: var(--bg-alt); border: none; font-size: 1rem; color: var(--text-light); }
    .toggle-tab:hover { background: var(--border); }
    .toggle-tab.active { background: white; color: var(--primary); font-weight: 600; }
    .content-panel { padding: 1.5rem; display: none; }
    .content-panel.active { display: block; }
    .content-panel p { margin-bottom: 1rem; text-align: justify; }
    .summary-text { font-size: 1.05rem; line-height: 1.8; }
    .summary-list { list-style: none; padding: 0; margin: 0; }
    .summary-list li { padding: 0.5rem 0; padding-left: 1.5rem; position: relative; border-bottom: 1px solid var(--border); }
    .summary-list li:last-child { border-bottom: none; }
    .summary-list li::before { content: "•"; position: absolute; left: 0; color: var(--accent); font-weight: bold; }
    .nav-links { display: flex; justify-content: space-between; margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid var(--border); }
    .nav-links a { color: var(--accent); text-decoration: none; }
    footer { text-align: center; padding: 2rem; color: var(--text-light); font-size: 0.9rem; }
    footer a { color: var(--accent); }

    /* Chapter cards - citymeetings.nyc style */
    .chapters-list { display: flex; flex-direction: column; gap: 1rem; }
    .chapter-card {
      background: var(--bg-alt);
      border-radius: 8px;
      padding: 1rem 1.25rem;
      border-left: 4px solid var(--accent);
    }
    .chapter-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.5rem;
    }
    .chapter-category {
      font-size: 0.7rem;
      font-weight: 600;
      color: white;
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .chapter-time {
      font-size: 0.8rem;
      color: var(--accent);
      text-decoration: none;
      font-family: monospace;
    }
    .chapter-time:hover { text-decoration: underline; }
    .chapter-title {
      font-size: 1rem;
      font-weight: 600;
      color: var(--primary);
      margin-bottom: 0.4rem;
      line-height: 1.4;
    }
    .chapter-description {
      font-size: 0.95rem;
      color: var(--text);
      line-height: 1.6;
      margin: 0;
    }
'''


def main():
    project_root = Path(__file__).parent.parent

    # Load meetings
    with open(project_root / 'src' / '_data' / 'meetings.json') as f:
        all_meetings = json.load(f)

    # Load name corrections
    officials_path = project_root / 'data' / 'elected_officials.json'
    misspellings = {}
    if officials_path.exists():
        with open(officials_path) as f:
            officials_data = json.load(f)
            misspellings = officials_data.get('common_misspellings', {})

    # Filter to meetings with videoId
    meetings = [m for m in all_meetings if m.get('videoId')]

    # Command line args
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    use_llm = '--no-llm' not in sys.argv

    print(f"Building site with {limit} meetings (LLM: {use_llm})")

    # Output directory
    output_dir = project_root / 'dist'
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'meetings').mkdir(exist_ok=True)

    # Process meetings
    processed = []
    for i, meeting in enumerate(meetings[:limit]):
        print(f"\n[{i+1}/{limit}] Processing: {meeting['date']} - {meeting['committee']}")

        data = process_meeting(meeting, project_root, misspellings, use_llm)
        if data:
            # Generate pages
            meeting_dir = output_dir / 'meetings' / meeting['slug']
            generate_meeting_pages(data, meeting_dir, use_llm)
            processed.append(data)
            print(f"    Generated {len([s for s in data['sections'] if data['transcript']])} section pages")
        else:
            print(f"    Skipped (no captions)")

    # Generate home page
    print(f"\nGenerating home page...")
    generate_home_page(processed, output_dir)

    print(f"\nDone! Site built in {output_dir}")
    print(f"Open {output_dir / 'index.html'} to view")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Process a meeting transcript using a local LLM to:
1. Segment the transcript by agenda items
2. Identify speakers
"""

import json
import re
import urllib.request
from pathlib import Path


def call_ollama(prompt, model="llama3.2", timeout=300):
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
        print(f"Error calling Ollama: {e}")
        return None


def clean_vtt_with_timestamps(vtt_path):
    """Convert VTT to clean text with timestamps preserved."""
    with open(vtt_path, 'r') as f:
        content = f.read()

    # Remove BOM
    content = content.replace('\ufeff', '')

    segments = []
    lines = content.split('\n')
    i = 0
    prev_text = ""

    while i < len(lines):
        line = lines[i].strip()

        # Match timestamp line
        time_match = re.match(r'(\d{2}:\d{2}:\d{2})\.\d{3} --> (\d{2}:\d{2}:\d{2})', line)
        if time_match:
            start_time = time_match.group(1)
            i += 1

            # Collect text lines until next timestamp or blank
            text_lines = []
            while i < len(lines) and not re.match(r'\d{2}:\d{2}:\d{2}', lines[i]):
                text = lines[i].strip()
                if text:
                    # Clean VTT formatting
                    text = re.sub(r'<[\d:\.]+>', '', text)
                    text = re.sub(r'</?c>', '', text)
                    text = text.replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')
                    text_lines.append(text)
                i += 1

            full_text = ' '.join(text_lines)

            # Avoid duplicates (VTT scrolling effect)
            if full_text and full_text != prev_text:
                # Check for speaker change marker
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
    """Merge consecutive segments from same speaker into paragraphs, removing duplicates."""
    if not segments:
        return []

    # First pass: extract only NEW text from each segment (remove overlaps)
    cleaned = []
    prev_text = ""

    for seg in segments:
        curr_text = seg['text']

        # Find the new portion of text (not in previous)
        new_text = curr_text

        # Check if current starts with end of previous (overlap)
        if prev_text:
            # Try to find overlap
            for overlap_len in range(min(len(prev_text), len(curr_text)), 0, -1):
                if prev_text.endswith(curr_text[:overlap_len]):
                    new_text = curr_text[overlap_len:].strip()
                    break

            # Also check if entire current is contained in previous
            if curr_text in prev_text:
                new_text = ""

        if new_text:
            cleaned.append({
                'time': seg['time'],
                'text': new_text,
                'new_speaker': seg['new_speaker']
            })
            prev_text = curr_text
        elif seg['new_speaker'] and cleaned:
            # Mark speaker change even if text is duplicate
            cleaned[-1]['new_speaker'] = True

    # Second pass: merge into paragraphs by speaker
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

        # Match Roman numeral sections (I., II., III., etc.)
        roman_match = re.match(r'^([IVX]+)\.\s+(.+)$', line)
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

        # Match numbered committee sections (1., 2., etc.)
        num_match = re.match(r'^(\d+)\.\s+(.+)$', line)
        if num_match:
            if current_section:
                sections.append({
                    'id': current_section['id'],
                    'title': current_section['title'],
                    'items': current_items
                })
            current_section = {
                'id': f"VI-{num_match.group(1)}",  # Mark as sub-section of Committee Reports
                'title': num_match.group(2).strip()
            }
            current_items = []
            continue

        # Match lettered items (a), b), etc.)
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


def identify_speakers_batch(paragraphs, attendees, batch_size=5):
    """Identify speakers in batches using LLM."""
    labeled = []
    context = ""

    for i in range(0, len(paragraphs), batch_size):
        batch = paragraphs[i:i+batch_size]

        # Format batch for LLM
        batch_text = "\n".join([
            f"[{p['time']}] [SPEAKER]: {p['text'][:300]}"
            for p in batch
        ])

        prompt = f"""Label the speakers in this community board meeting transcript.

ATTENDEES: {', '.join(attendees[:20])}

RULES:
- Replace [SPEAKER] with the actual name like [Sandy McKee]
- Use context clues: "Thank you, Mr. Smith" means the previous speaker was Mr. Smith
- The Chair (Sandy McKee) runs the meeting and recognizes speakers
- If unsure, use [Unknown] or [Member of Public]

PREVIOUS CONTEXT: {context[:200]}

TRANSCRIPT:
{batch_text}

OUTPUT each line with the speaker identified. Format: [TIME] [NAME]: text"""

        print(f"  Processing batch {i//batch_size + 1}...")
        response = call_ollama(prompt, timeout=120)

        if response:
            # Parse response to extract speaker labels
            for j, para in enumerate(batch):
                # Try to find matching line in response
                time_pattern = re.escape(para['time'])
                match = re.search(rf'\[{time_pattern}\]\s*\[([^\]]+)\]', response)
                if match:
                    para['speaker'] = match.group(1)
                labeled.append(para)

            # Update context for next batch
            context = batch[-1]['text'][:200] if batch else ""
        else:
            # If LLM fails, keep original
            labeled.extend(batch)

    return labeled


def assign_sections(paragraphs, sections):
    """Assign each paragraph to an agenda section based on content and time."""
    # Define section order for sequential progression
    section_order = ['I', 'II', 'III', 'IV', 'V', 'VI']
    # Add committee sections
    for sec in sections:
        if sec['id'].startswith('VI-'):
            section_order.append(sec['id'])
    section_order.extend(['VII', 'VIII', 'IX'])

    # Known section markers (only trigger forward progression)
    section_markers = {
        'I': ['call to order', 'called to order', 'meeting to order'],
        'II': ['adopt the agenda', 'adoption of the agenda'],
        'III': ['roll call', 'take attendance', 'first roll call'],
        'IV': ['public session', 'public hearing', 'elected and agency'],
        'V': ['business session', "chair's report", 'district manager report'],
        'VI': ['committee reports'],
        'VII': ['old business', 'new business', 'old/new business'],
        'VIII': ['second roll call'],
        'IX': ['meeting is adjourned', 'we will adjourn', 'motion to adjourn']
    }

    # Committee markers
    committee_markers = {}
    for sec in sections:
        if sec['id'].startswith('VI-'):
            # Extract committee name before the dash
            name = sec['title'].lower().split('–')[0].strip()
            committee_markers[sec['id']] = [name, name.replace('&', 'and')]

    current_section = 'I'
    current_idx = 0

    for para in paragraphs:
        text_lower = para['text'].lower()
        found_new_section = False

        # Check for section transition markers (only move forward, take first match)
        for sec_id in section_order:
            if found_new_section:
                break

            # Check main section markers
            if sec_id in section_markers:
                sec_idx = section_order.index(sec_id)
                # Only advance by one section at a time (or skip to next if current+1)
                if sec_idx == current_idx + 1:
                    for marker in section_markers[sec_id]:
                        if marker in text_lower:
                            current_section = sec_id
                            current_idx = sec_idx
                            found_new_section = True
                            break

            # Check committee section markers
            if sec_id in committee_markers:
                sec_idx = section_order.index(sec_id)
                if sec_idx == current_idx + 1:
                    for marker in committee_markers[sec_id]:
                        if marker in text_lower and ('report' in text_lower or 'committee' in text_lower or 'chair' in text_lower):
                            current_section = sec_id
                            current_idx = sec_idx
                            found_new_section = True
                            break

        para['section'] = current_section

    return paragraphs


def main():
    project_root = Path(__file__).parent.parent

    # Load meeting data
    meetings_path = project_root / 'src' / '_data' / 'meetings.json'
    with open(meetings_path) as f:
        meetings = json.load(f)

    # Find test meeting
    meeting = next((m for m in meetings if m['slug'] == '2026-01-14-full-board'), None)
    if not meeting:
        print("Meeting not found!")
        return

    print(f"Processing: {meeting['committee']} - {meeting['date']}")

    # Load transcript
    vtt_path = project_root / 'transcripts' / f"{meeting['videoId']}.vtt"
    print(f"Loading transcript from {vtt_path}")

    segments = clean_vtt_with_timestamps(vtt_path)
    print(f"Loaded {len(segments)} segments")

    # Consolidate into paragraphs
    paragraphs = consolidate_segments(segments)
    print(f"Consolidated into {len(paragraphs)} paragraphs")

    # Parse agenda
    sections = parse_agenda(meeting['agenda'])
    print(f"Parsed {len(sections)} agenda sections")

    # Assign sections to paragraphs
    paragraphs = assign_sections(paragraphs, sections)

    # Known attendees (from agenda + common names)
    attendees = [
        "Sandy McKee", "John Keller", "Jerry Weinstein", "Jesus Perez",
        "Brennan Bur", "Tania Arias", "Rupal Kakkad", "Jason Froimowitz",
        "Anton Mallner", "Stu Desser", "Neil Barclay", "Gabe Turzo",
        "Steve Perez", "Alek Miletic", "Elvie Barroso", "Daniel Bernstein",
        "Liz Burr", "Beatrice Disman", "Eric Goldberg", "Kevin O'Keefe",
        "Katie Landon", "Richard Mintz"
    ]

    # Print sample
    print("\n" + "="*60)
    print("SAMPLE TRANSCRIPT:")
    print("="*60)
    for p in paragraphs[:5]:
        print(f"[{p['time']}] [{p.get('section', '?')}]: {p['text'][:100]}...")
        print()

    # Build output structure
    output = {
        'meeting': {
            'slug': meeting['slug'],
            'date': meeting['date'],
            'committee': meeting['committee'],
            'youtubeUrl': meeting['youtubeUrl']
        },
        'sections': sections,
        'attendees': attendees,
        'transcript': paragraphs
    }

    output_path = project_root / 'poc' / 'processed_meeting.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")
    print(f"Total paragraphs: {len(paragraphs)}")


if __name__ == '__main__':
    main()

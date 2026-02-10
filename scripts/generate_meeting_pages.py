#!/usr/bin/env python3
"""
Generate multi-page meeting site:
1. Agenda landing page with sections and resolutions
2. Individual section pages with embedded video, summary, and transcript
"""

import json
import re
import urllib.request
from pathlib import Path


def call_ollama(prompt, model="llama3.2", timeout=120):
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
        print(f"    Warning: Ollama error: {e}")
        return None


def format_time_link(time_str, youtube_url):
    """Convert HH:MM:SS to YouTube timestamp link."""
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        total_seconds = h * 3600 + m * 60 + s
        return total_seconds
    return 0


def format_time_display(time_str):
    """Format time for display (remove leading zeros)."""
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        else:
            return f"{m}:{s:02d}"
    return time_str


def get_youtube_embed_url(youtube_url, start_seconds):
    """Get YouTube embed URL with start time."""
    # Extract video ID
    if 'youtu.be/' in youtube_url:
        video_id = youtube_url.split('youtu.be/')[-1].split('?')[0]
    elif 'watch?v=' in youtube_url:
        video_id = youtube_url.split('watch?v=')[-1].split('&')[0]
    else:
        video_id = youtube_url

    return f"https://www.youtube.com/embed/{video_id}?start={start_seconds}"


def trim_pre_meeting(transcript):
    """Remove pre-meeting chatter, start at 'call to order'."""
    for i, para in enumerate(transcript):
        text_lower = para['text'].lower()
        if 'call to order' in text_lower or 'called to order' in text_lower:
            return transcript[i:]
    return transcript


def generate_summary(paragraphs, section_title, use_llm=True):
    """Generate a summary using Ollama LLM."""
    if not paragraphs:
        return "No content recorded for this section."

    all_text = ' '.join(p['text'] for p in paragraphs)

    # If text is very short, just return it
    if len(all_text) < 300:
        return all_text

    if use_llm:
        # Truncate to ~2000 chars for faster LLM processing
        text_for_summary = all_text[:2000]

        prompt = f"""Summarize this community board meeting section in 2-3 sentences. Focus on what was discussed, any votes, and key outcomes.

Section: {section_title}

Transcript:
{text_for_summary}

Summary:"""

        summary = call_ollama(prompt, timeout=180)
        if summary:
            # Clean up the summary
            summary = summary.strip()
            # Remove any "Here is" or "This section" preambles
            summary = re.sub(r'^(Here is|This section|The section|In this section)[^.]*[.:]\s*', '', summary, flags=re.IGNORECASE)
            return summary

    # Fallback: extract first meaningful sentences
    sentences = re.split(r'[.!?]+', all_text)
    summary_parts = []
    char_count = 0
    for sent in sentences[:8]:
        sent = sent.strip()
        if len(sent) > 30:  # Skip very short fragments
            summary_parts.append(sent)
            char_count += len(sent)
            if char_count > 400:
                break

    if summary_parts:
        return '. '.join(summary_parts) + '.'

    return all_text[:500] + '...'


def generate_css():
    """Generate shared CSS."""
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
    .container {
      max-width: 900px;
      margin: 0 auto;
      padding: 2rem;
    }
    header {
      background: var(--primary);
      color: white;
      padding: 1.5rem 2rem;
    }
    header .container { padding: 0; }
    h1 {
      font-size: 1.5rem;
      margin-bottom: 0.25rem;
      font-weight: 600;
    }
    .meta {
      opacity: 0.9;
      font-size: 0.95rem;
    }
    .breadcrumb {
      padding: 1rem 2rem;
      background: var(--bg);
      border-bottom: 1px solid var(--border);
      font-size: 0.9rem;
    }
    .breadcrumb a {
      color: var(--accent);
      text-decoration: none;
    }
    .breadcrumb a:hover { text-decoration: underline; }

    /* Agenda page styles */
    .agenda-section {
      background: white;
      border-radius: 8px;
      margin-bottom: 1rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
      overflow: hidden;
    }
    .agenda-section-header {
      padding: 1rem 1.5rem;
      background: var(--bg-alt);
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .agenda-section-header h2 {
      font-size: 1.1rem;
      font-weight: 600;
      color: var(--primary);
    }
    .agenda-section-header h2 a {
      color: var(--primary);
      text-decoration: none;
    }
    .agenda-section-header h2 a:hover {
      color: var(--accent);
      text-decoration: underline;
    }
    .agenda-items {
      padding: 1rem 1.5rem;
      list-style: none;
    }
    .agenda-items li {
      padding: 0.5rem 0;
      border-bottom: 1px solid var(--border);
    }
    .agenda-items li:last-child { border-bottom: none; }
    .timestamp-badge {
      font-size: 0.8rem;
      color: var(--text-light);
      background: var(--bg-alt);
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      margin-left: 0.5rem;
    }
    .no-items {
      color: var(--text-light);
      font-style: italic;
      padding: 0.5rem 0;
    }

    /* Section page styles */
    .video-container {
      background: #000;
      border-radius: 8px;
      overflow: hidden;
      margin-bottom: 1.5rem;
      aspect-ratio: 16/9;
    }
    .video-container iframe {
      width: 100%;
      height: 100%;
      border: none;
    }
    .content-card {
      background: white;
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
      overflow: hidden;
    }
    .toggle-tabs {
      display: flex;
      border-bottom: 1px solid var(--border);
    }
    .toggle-tab {
      flex: 1;
      padding: 1rem;
      text-align: center;
      cursor: pointer;
      background: var(--bg-alt);
      border: none;
      font-size: 1rem;
      color: var(--text-light);
      transition: all 0.2s;
    }
    .toggle-tab:hover {
      background: var(--border);
    }
    .toggle-tab.active {
      background: white;
      color: var(--primary);
      font-weight: 600;
    }
    .content-panel {
      padding: 1.5rem;
      display: none;
    }
    .content-panel.active {
      display: block;
    }
    .content-panel p {
      margin-bottom: 1rem;
      text-align: justify;
    }
    .content-panel p:last-child { margin-bottom: 0; }
    .summary-text {
      font-size: 1.05rem;
      line-height: 1.8;
    }
    .nav-links {
      display: flex;
      justify-content: space-between;
      margin-top: 1.5rem;
      padding-top: 1.5rem;
      border-top: 1px solid var(--border);
    }
    .nav-links a {
      color: var(--accent);
      text-decoration: none;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .nav-links a:hover { text-decoration: underline; }
    footer {
      text-align: center;
      padding: 2rem;
      color: var(--text-light);
      font-size: 0.9rem;
    }
    footer a { color: var(--accent); }
'''


def generate_agenda_page(processed_data, output_dir):
    """Generate the main agenda landing page."""
    meeting = processed_data['meeting']
    sections = processed_data['sections']
    transcript = processed_data['transcript']
    boundaries = processed_data.get('section_boundaries', {})

    # Group transcript by section
    sections_content = {}
    for para in transcript:
        sec_id = para.get('section', 'I')
        if sec_id not in sections_content:
            sections_content[sec_id] = []
        sections_content[sec_id].append(para)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{meeting['committee']} Meeting - {meeting['date']} | CB6 Transcripts</title>
  <style>{generate_css()}</style>
</head>
<body>
  <header>
    <div class="container">
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
        has_content = len(paragraphs) > 0

        # Get timestamp
        first_time = paragraphs[0]['time'] if paragraphs else '00:00:00'
        time_display = format_time_display(first_time)

        # Create section slug for URL
        sec_slug = sec_id.lower().replace('-', '')

        html += f'''    <div class="agenda-section">
      <div class="agenda-section-header">
'''

        if has_content:
            html += f'''        <h2><a href="section-{sec_slug}.html">{sec_id}. {title}</a></h2>
        <span class="timestamp-badge">{time_display}</span>
'''
        else:
            html += f'''        <h2>{sec_id}. {title}</h2>
'''

        html += '''      </div>
'''

        if items:
            html += '''      <ul class="agenda-items">
'''
            for item in items:
                letter = item.get('letter', '')
                text = item.get('text', '')
                html += f'''        <li><strong>{letter})</strong> {text}</li>
'''
            html += '''      </ul>
'''
        elif has_content:
            # Show brief preview
            preview = paragraphs[0]['text'][:150] + '...' if len(paragraphs[0]['text']) > 150 else paragraphs[0]['text']
            html += f'''      <div class="agenda-items">
        <p class="no-items" style="font-style: normal;">{preview}</p>
      </div>
'''
        else:
            html += '''      <div class="agenda-items">
        <p class="no-items">No transcript available</p>
      </div>
'''

        html += '''    </div>
'''

    html += f'''  </div>

  <footer>
    <p>Transcripts auto-generated from <a href="{meeting['youtubeUrl']}" target="_blank">CB6 YouTube</a>.</p>
    <p>Not an official CB6 website.</p>
  </footer>
</body>
</html>
'''

    # Write file
    output_path = output_dir / 'index.html'
    with open(output_path, 'w') as f:
        f.write(html)

    return output_path


def generate_section_page(processed_data, section, section_index, all_sections, output_dir):
    """Generate individual section page with video and transcript."""
    meeting = processed_data['meeting']
    transcript = processed_data['transcript']

    sec_id = section['id']
    title = section['title']
    sec_slug = sec_id.lower().replace('-', '')

    # Get paragraphs for this section
    paragraphs = [p for p in transcript if p.get('section') == sec_id]

    if not paragraphs:
        return None

    # Get timestamp and embed URL
    first_time = paragraphs[0]['time']
    start_seconds = format_time_link(first_time, meeting['youtubeUrl'])
    embed_url = get_youtube_embed_url(meeting['youtubeUrl'], start_seconds)
    time_display = format_time_display(first_time)

    # Generate summary with LLM
    print(f"    Generating summary for {sec_id}...")
    summary = generate_summary(paragraphs, title, use_llm=True)

    # Format transcript text
    transcript_paragraphs = []
    current_para = []
    for i, para in enumerate(paragraphs):
        current_para.append(para['text'])
        combined = ' '.join(current_para)
        if len(combined) > 500 or (i > 0 and i % 6 == 0):
            transcript_paragraphs.append(combined)
            current_para = []
    if current_para:
        transcript_paragraphs.append(' '.join(current_para))

    # Get prev/next sections with content
    prev_section = None
    next_section = None

    for i, sec in enumerate(all_sections):
        sec_paragraphs = [p for p in transcript if p.get('section') == sec['id']]
        if sec_paragraphs:
            if i < section_index:
                prev_section = sec
            elif i > section_index and next_section is None:
                next_section = sec
                break

    # Direct YouTube link with timestamp
    if 'youtu.be/' in meeting['youtubeUrl']:
        video_id = meeting['youtubeUrl'].split('youtu.be/')[-1].split('?')[0]
    else:
        video_id = meeting['youtubeUrl'].split('watch?v=')[-1].split('&')[0]
    direct_yt_link = f"https://www.youtube.com/watch?v={video_id}&t={start_seconds}"

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{sec_id}. {title} | {meeting['committee']} - {meeting['date']}</title>
  <style>{generate_css()}</style>
</head>
<body>
  <header>
    <div class="container">
      <h1>{sec_id}. {title}</h1>
      <div class="meta">{meeting['committee']} · {meeting['date']} · {time_display}</div>
    </div>
  </header>

  <div class="breadcrumb">
    <a href="index.html">← Back to Agenda</a>
  </div>

  <div class="container">
    <div class="video-container">
      <iframe
        src="{embed_url}"
        referrerpolicy="no-referrer-when-downgrade"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        allowfullscreen>
      </iframe>
    </div>
    <p style="text-align: center; margin-bottom: 1.5rem; font-size: 0.9rem; color: var(--text-light);">
      Video not loading? <a href="{direct_yt_link}" target="_blank">Watch on YouTube</a>
    </p>

    <div class="content-card">
      <div class="toggle-tabs">
        <button class="toggle-tab active" onclick="showPanel('summary')">Summary</button>
        <button class="toggle-tab" onclick="showPanel('transcript')">Full Transcript</button>
      </div>

      <div id="summary" class="content-panel active">
        <p class="summary-text">{summary}</p>
      </div>

      <div id="transcript" class="content-panel">
'''

    for para_text in transcript_paragraphs:
        clean_text = re.sub(r'\s+', ' ', para_text.strip())
        html += f'        <p>{clean_text}</p>\n'

    html += '''      </div>
    </div>

    <div class="nav-links">
'''

    if prev_section:
        prev_slug = prev_section['id'].lower().replace('-', '')
        html += f'''      <a href="section-{prev_slug}.html">← {prev_section['id']}. {prev_section['title'][:30]}...</a>
'''
    else:
        html += '''      <span></span>
'''

    if next_section:
        next_slug = next_section['id'].lower().replace('-', '')
        html += f'''      <a href="section-{next_slug}.html">{next_section['id']}. {next_section['title'][:30]}... →</a>
'''
    else:
        html += '''      <span></span>
'''

    html += f'''    </div>
  </div>

  <footer>
    <p>Transcripts auto-generated from <a href="{meeting['youtubeUrl']}" target="_blank">CB6 YouTube</a>.</p>
    <p>Not an official CB6 website.</p>
  </footer>

  <script>
    function showPanel(panelId) {{
      // Hide all panels
      document.querySelectorAll('.content-panel').forEach(p => p.classList.remove('active'));
      document.querySelectorAll('.toggle-tab').forEach(t => t.classList.remove('active'));

      // Show selected panel
      document.getElementById(panelId).classList.add('active');
      event.target.classList.add('active');
    }}
  </script>
</body>
</html>
'''

    # Write file
    output_path = output_dir / f'section-{sec_slug}.html'
    with open(output_path, 'w') as f:
        f.write(html)

    return output_path


def main():
    project_root = Path(__file__).parent.parent

    # Load processed data
    processed_path = project_root / 'poc' / 'processed_meeting.json'
    with open(processed_path) as f:
        processed_data = json.load(f)

    meeting = processed_data['meeting']
    sections = processed_data['sections']

    print(f"Generating pages for: {meeting['slug']}")

    # Create output directory
    output_dir = project_root / 'poc' / 'meeting'
    output_dir.mkdir(exist_ok=True)

    # Generate agenda page
    agenda_path = generate_agenda_page(processed_data, output_dir)
    print(f"  Created: {agenda_path}")

    # Generate section pages
    for i, section in enumerate(sections):
        section_path = generate_section_page(processed_data, section, i, sections, output_dir)
        if section_path:
            print(f"  Created: {section_path}")

    print(f"\nDone! Open {output_dir / 'index.html'} to view.")


if __name__ == '__main__':
    main()

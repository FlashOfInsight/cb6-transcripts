#!/usr/bin/env python3
"""
Generate a formatted meeting page from processed transcript data.
Chapter-based layout with flowing prose.
"""

import json
import re
from pathlib import Path


def trim_pre_meeting(transcript):
    """Remove pre-meeting chatter, start at 'call to order'."""
    for i, para in enumerate(transcript):
        text_lower = para['text'].lower()
        if 'call to order' in text_lower or 'called to order' in text_lower or 'welcome to the' in text_lower:
            return transcript[i:]
    return transcript


def format_time_link(time_str, youtube_url):
    """Convert HH:MM:SS to YouTube timestamp link."""
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        total_seconds = h * 3600 + m * 60 + s
        if 'youtu.be' in youtube_url:
            return f"{youtube_url}?t={total_seconds}"
        else:
            return f"{youtube_url}&t={total_seconds}"
    return youtube_url


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


def generate_html(processed_data):
    """Generate HTML page from processed meeting data."""
    meeting = processed_data['meeting']
    sections = processed_data['sections']
    transcript = processed_data['transcript']

    # Trim pre-meeting chatter
    transcript = trim_pre_meeting(transcript)

    # Group transcript paragraphs by section
    sections_content = {}
    for para in transcript:
        sec_id = para.get('section', 'I')
        if sec_id not in sections_content:
            sections_content[sec_id] = []
        sections_content[sec_id].append(para)

    # Build HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{meeting['committee']} Meeting - {meeting['date']} | CB6 Transcripts</title>
  <style>
    :root {{
      --primary: #1a365d;
      --primary-light: #2c5282;
      --accent: #3182ce;
      --text: #2d3748;
      --text-light: #718096;
      --bg: #ffffff;
      --bg-alt: #f7fafc;
      --border: #e2e8f0;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      color: var(--text);
      line-height: 1.7;
      background: var(--bg-alt);
    }}
    .container {{
      max-width: 800px;
      margin: 0 auto;
      padding: 2rem;
    }}
    header {{
      background: var(--primary);
      color: white;
      padding: 2rem;
      margin-bottom: 2rem;
    }}
    header .container {{
      padding: 0;
    }}
    h1 {{
      font-size: 1.75rem;
      margin-bottom: 0.5rem;
      font-weight: 600;
    }}
    .meta {{
      opacity: 0.9;
      margin-bottom: 1rem;
    }}
    .header-links {{
      display: flex;
      gap: 1rem;
      margin-top: 1rem;
    }}
    .header-links a {{
      color: white;
      background: rgba(255,255,255,0.15);
      padding: 0.5rem 1rem;
      border-radius: 4px;
      text-decoration: none;
      font-size: 0.9rem;
    }}
    .header-links a:hover {{
      background: rgba(255,255,255,0.25);
    }}

    /* Agenda nav */
    .agenda-nav {{
      background: white;
      border-radius: 8px;
      padding: 1.5rem;
      margin-bottom: 2rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    .agenda-nav h2 {{
      font-size: 1rem;
      color: var(--text-light);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 1rem;
    }}
    .agenda-list {{
      list-style: none;
    }}
    .agenda-list li {{
      margin-bottom: 0.5rem;
    }}
    .agenda-list a {{
      color: var(--accent);
      text-decoration: none;
    }}
    .agenda-list a:hover {{
      text-decoration: underline;
    }}
    .agenda-list .sub-items {{
      margin-left: 1.5rem;
      font-size: 0.9rem;
      color: var(--text-light);
    }}

    /* Transcript sections */
    .section {{
      background: white;
      border-radius: 8px;
      margin-bottom: 1.5rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
      scroll-margin-top: 1rem;
    }}
    .section-header {{
      padding: 1rem 1.5rem;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
    }}
    .section-title {{
      font-size: 1.1rem;
      font-weight: 600;
      color: var(--primary);
    }}
    .section-meta {{
      display: flex;
      gap: 1rem;
      align-items: center;
    }}
    .timestamp {{
      font-size: 0.85rem;
      color: var(--accent);
      text-decoration: none;
      background: var(--bg-alt);
      padding: 0.25rem 0.5rem;
      border-radius: 4px;
    }}
    .timestamp:hover {{
      background: var(--border);
    }}
    .section-content {{
      padding: 1.5rem;
      font-size: 1rem;
      line-height: 1.8;
    }}
    .section-content p {{
      margin-bottom: 1rem;
      text-align: justify;
    }}
    .section-content p:last-child {{
      margin-bottom: 0;
    }}

    /* Footer */
    footer {{
      text-align: center;
      padding: 2rem;
      color: var(--text-light);
      font-size: 0.9rem;
    }}
    footer a {{
      color: var(--accent);
    }}

    /* Back to top */
    .back-to-top {{
      position: fixed;
      bottom: 2rem;
      right: 2rem;
      background: var(--primary);
      color: white;
      padding: 0.75rem 1rem;
      border-radius: 4px;
      text-decoration: none;
      font-size: 0.9rem;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }}
    .back-to-top:hover {{
      background: var(--primary-light);
    }}
  </style>
</head>
<body>
  <header>
    <div class="container">
      <h1>{meeting['committee']} Meeting</h1>
      <div class="meta">Manhattan Community Board 6 · {meeting['date']}</div>
      <div class="header-links">
        <a href="{meeting['youtubeUrl']}" target="_blank">▶ Watch Full Video</a>
        <a href="#agenda">Jump to Agenda</a>
      </div>
    </div>
  </header>

  <div class="container">
    <nav class="agenda-nav" id="agenda">
      <h2>Agenda</h2>
      <ul class="agenda-list">
'''

    # Add agenda sections with links
    for sec in sections:
        sec_id = sec['id']
        title = sec['title']
        has_content = sec_id in sections_content and len(sections_content[sec_id]) > 0

        if has_content:
            html += f'        <li><a href="#section-{sec_id}">{sec_id}. {title}</a></li>\n'
        else:
            html += f'        <li style="color: var(--text-light);">{sec_id}. {title}</li>\n'

    html += '''      </ul>
    </nav>

    <main>
'''

    # Add transcript sections
    for sec in sections:
        sec_id = sec['id']
        title = sec['title']
        paragraphs = sections_content.get(sec_id, [])

        if not paragraphs:
            continue

        # Get first timestamp for the section
        first_time = paragraphs[0]['time']
        time_link = format_time_link(first_time, meeting['youtubeUrl'])
        time_display = format_time_display(first_time)

        # Combine all paragraph text into flowing prose
        # Group into larger paragraphs (roughly every 5-8 utterances)
        all_text = []
        current_para = []

        for i, para in enumerate(paragraphs):
            current_para.append(para['text'])

            # Start new paragraph every ~500 characters or at natural breaks
            combined = ' '.join(current_para)
            if len(combined) > 500 or (i > 0 and i % 6 == 0):
                all_text.append(combined)
                current_para = []

        if current_para:
            all_text.append(' '.join(current_para))

        html += f'''      <section id="section-{sec_id}" class="section">
        <div class="section-header">
          <div class="section-title">{sec_id}. {title}</div>
          <div class="section-meta">
            <a href="{time_link}" target="_blank" class="timestamp">{time_display}</a>
          </div>
        </div>
        <div class="section-content">
'''

        for para_text in all_text:
            # Clean up the text
            clean_text = para_text.strip()
            clean_text = re.sub(r'\s+', ' ', clean_text)  # Normalize whitespace
            html += f'          <p>{clean_text}</p>\n'

        html += '''        </div>
      </section>

'''

    html += '''    </main>
  </div>

  <footer>
    <p>Transcripts auto-generated from <a href="{}" target="_blank">CB6 YouTube</a>.</p>
    <p>Not an official CB6 website.</p>
  </footer>

  <a href="#" class="back-to-top">↑ Top</a>
</body>
</html>
'''.format(meeting['youtubeUrl'])

    return html


def main():
    project_root = Path(__file__).parent.parent

    # Load processed data
    processed_path = project_root / 'poc' / 'processed_meeting.json'
    with open(processed_path) as f:
        processed_data = json.load(f)

    print(f"Generating page for: {processed_data['meeting']['slug']}")

    # Generate HTML
    html = generate_html(processed_data)

    # Save to poc folder for preview
    output_path = project_root / 'poc' / 'meeting_preview.html'
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()

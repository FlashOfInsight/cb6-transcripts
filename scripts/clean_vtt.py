#!/usr/bin/env python3
"""Clean VTT captions to plain text transcript."""

import re
import sys

def clean_vtt(vtt_path):
    with open(vtt_path, 'r') as f:
        content = f.read()

    # Remove VTT header
    content = re.sub(r'^WEBVTT\nKind:.*\nLanguage:.*\n', '', content)

    # Remove timestamps and positioning
    content = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}.*\n', '', content)

    # Remove inline timestamp tags like <00:07:30.800><c>
    content = re.sub(r'<[\d:\.]+>', '', content)
    content = re.sub(r'</?c>', '', content)

    # Replace HTML entities
    content = content.replace('&gt;', '>')
    content = content.replace('&lt;', '<')
    content = content.replace('&amp;', '&')

    # Clean up speaker markers (>> becomes newline + SPEAKER:)
    content = re.sub(r'>>\s*', '\n[SPEAKER]: ', content)

    # Remove duplicate lines (VTT repeats lines for scrolling effect)
    lines = content.split('\n')
    cleaned_lines = []
    prev_line = ''
    for line in lines:
        line = line.strip()
        if line and line != prev_line:
            cleaned_lines.append(line)
            prev_line = line

    # Join and clean up whitespace
    text = '\n'.join(cleaned_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

if __name__ == '__main__':
    vtt_path = sys.argv[1] if len(sys.argv) > 1 else 'test_meeting.en.vtt'
    print(clean_vtt(vtt_path))

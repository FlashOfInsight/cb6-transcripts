#!/usr/bin/env python3
"""
Correct elected official name spellings in transcripts.
Uses fuzzy matching against a reference file of known officials.
"""

import json
import re
from pathlib import Path


def load_officials(officials_path):
    """Load elected officials reference file."""
    with open(officials_path) as f:
        data = json.load(f)

    # Build lookup structures
    officials = {}  # name -> info
    all_names = []  # list of all correct names
    misspellings = data.get('common_misspellings', {})

    for category in ['citywide', 'manhattan_borough_president', 'city_council_manhattan',
                     'state_senate_manhattan', 'state_assembly_manhattan', 'us_congress_manhattan']:
        for official in data.get(category, []):
            name = official['name']
            officials[name.lower()] = official
            all_names.append(name)

            # Also index aliases
            for alias in official.get('aliases', []):
                officials[alias.lower()] = official

    return officials, all_names, misspellings


def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def find_best_match(word, all_names, threshold=0.7):
    """Find best matching official name for a word."""
    word_lower = word.lower()
    best_match = None
    best_score = 0

    for name in all_names:
        name_lower = name.lower()

        # Check each part of the name
        name_parts = name_lower.split()
        for part in name_parts:
            if len(part) < 3:
                continue

            # Calculate similarity
            distance = levenshtein_distance(word_lower, part)
            max_len = max(len(word_lower), len(part))
            similarity = 1 - (distance / max_len)

            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match = name

    return best_match, best_score


def correct_transcript(transcript, officials, all_names, misspellings):
    """Correct name spellings in transcript paragraphs."""
    corrections_made = []

    for para in transcript:
        text = para['text']
        original_text = text

        # First, apply known misspellings
        for wrong, right in misspellings.items():
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            if pattern.search(text):
                text = pattern.sub(right, text)
                corrections_made.append((wrong, right))

        # Find potential name-like words (capitalized or after titles)
        # Look for words after "Senator", "Council Member", "Assembly Member", etc.
        title_patterns = [
            r'\b(Senator|Council\s*Member|Assembly\s*Member|Borough\s*President|'
            r'Representative|Congressman|Comptroller|Mayor)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        ]

        for pattern in title_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                title = match.group(1)
                name_part = match.group(2)

                # Check if this name needs correction
                best_match, score = find_best_match(name_part, all_names, threshold=0.75)
                if best_match and best_match.lower() != name_part.lower():
                    # Extract just the last name from best_match for replacement
                    replacement_name = best_match.split()[-1]
                    if score > 0.8:
                        old_text = match.group(0)
                        new_text = f"{title} {replacement_name}"
                        text = text.replace(old_text, new_text)
                        corrections_made.append((old_text, new_text))

        para['text'] = text

    return transcript, corrections_made


def main():
    project_root = Path(__file__).parent.parent

    # Load officials reference
    officials_path = project_root / 'data' / 'elected_officials.json'
    officials, all_names, misspellings = load_officials(officials_path)
    print(f"Loaded {len(all_names)} official names")

    # Load processed transcript
    processed_path = project_root / 'poc' / 'processed_meeting.json'
    with open(processed_path) as f:
        data = json.load(f)

    transcript = data['transcript']
    print(f"Processing {len(transcript)} paragraphs")

    # Correct names
    transcript, corrections = correct_transcript(transcript, officials, all_names, misspellings)

    if corrections:
        print(f"\nCorrections made:")
        seen = set()
        for wrong, right in corrections:
            if (wrong, right) not in seen:
                print(f"  '{wrong}' -> '{right}'")
                seen.add((wrong, right))
    else:
        print("\nNo corrections needed")

    # Save updated transcript
    data['transcript'] = transcript
    with open(processed_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved to {processed_path}")


if __name__ == '__main__':
    main()

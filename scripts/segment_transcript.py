#!/usr/bin/env python3
"""
Segment transcript into agenda sections using targeted phrase matching.
"""

import json
import re
from pathlib import Path


def find_section_boundaries(transcript):
    """Find where each section actually starts based on context."""

    boundaries = {}

    for i, para in enumerate(transcript):
        text = para['text']
        text_lower = text.lower()
        time = para['time']

        # Skip the opening statement (usually in first 10 paragraphs)
        # which mentions all agenda items

        # Section I: Call to Order - look for actual call to order
        if i < 10 and 'called to order' in text_lower and 'I' not in boundaries:
            boundaries['I'] = i

        # Section II: Adoption of Agenda - usually in same paragraph as call to order
        if i < 10 and 'adopt' in text_lower and 'agenda' in text_lower and 'II' not in boundaries:
            boundaries['II'] = i

        # Section III: Roll Call - look for "Howdy" (secretary) or actual roll call
        if 'howdy folks' in text_lower and 'III' not in boundaries:
            boundaries['III'] = i

        # Section IV: Public Session - look for elected officials speaking
        # Usually starts with Borough President or other elected rep
        if i > 20 and i < 150:
            if ('borough president' in text_lower or 'council member' in text_lower or
                'assembly member' in text_lower or 'senator' in text_lower) and 'IV' not in boundaries:
                # Check if this is an introduction of a speaker
                if 'we have' in text_lower or 'next' in text_lower or 'now' in text_lower:
                    boundaries['IV'] = i

        # Section V: Business Session - look for "chair's report" or "minutes"
        if i > 100:
            if ("chair's report" in text_lower or 'minutes from the' in text_lower) and 'V' not in boundaries:
                boundaries['V'] = i

        # Section VI: Committee Reports header
        if 'VI' not in boundaries:
            if 'committee reports and resolutions' in text_lower:
                boundaries['VI'] = i
            # Also check for "we've now arrived at" which precedes committee reports
            elif "we've now arrived" in text_lower and 'committee' in text_lower:
                boundaries['VI'] = i

        # Committee sections - look for "Next up" or direct committee name mentions
        # with context indicating a transition

        # VI-1: Health & Education - starts right after VI header with first resolution
        # The committee presents without explicit "health and education" announcement
        if 'VI-1' not in boundaries and 'VI' in boundaries:
            vi_start = boundaries['VI']
            if i > vi_start and i < vi_start + 20:
                # Look for first resolution about Bellevue (Health & Ed topic)
                if 'resolution' in text_lower or 'bellevue' in text_lower:
                    boundaries['VI-1'] = i

        # VI-2: Transportation
        if 'VI-2' not in boundaries:
            if 'transportation' in text_lower and ('jason' in text_lower or 'chair' in text_lower):
                if i > 350:  # After Health & Education
                    boundaries['VI-2'] = i

        # VI-3: Business Affairs
        if 'VI-3' not in boundaries:
            if ('business affairs' in text_lower or 'licensing' in text_lower):
                if 'anton' in text_lower or 'next' in text_lower or 'we have' in text_lower:
                    if i > 400:
                        boundaries['VI-3'] = i

        # VI-4: Public Safety
        if 'VI-4' not in boundaries:
            if 'public safety' in text_lower:
                if 'next up' in text_lower or 'we have' in text_lower or 'stu' in text_lower:
                    if i > 450:
                        boundaries['VI-4'] = i

        # VI-5: Parks
        if 'VI-5' not in boundaries:
            if 'parks' in text_lower and 'landmarks' in text_lower:
                if i > 500:
                    boundaries['VI-5'] = i

        # VI-6: Land Use
        if 'VI-6' not in boundaries:
            if 'land use' in text_lower or 'gabe turzo' in text_lower:
                if i > 550:
                    boundaries['VI-6'] = i

        # VI-7: Budget
        if 'VI-7' not in boundaries:
            if 'budget' in text_lower and 'government' in text_lower and 'steve' in text_lower:
                if i > 560:
                    boundaries['VI-7'] = i

        # VI-8: Housing
        if 'VI-8' not in boundaries:
            if 'housing' in text_lower and 'homelessness' in text_lower:
                if i > 565:
                    boundaries['VI-8'] = i

        # Section VIII: Second Roll Call
        if 'second roll call' in text_lower and 'VIII' not in boundaries:
            if i > 580:
                boundaries['VIII'] = i

        # Section IX: Adjournment
        if 'we will adjourn' in text_lower or 'meeting is adjourned' in text_lower:
            if 'IX' not in boundaries and i > 590:
                boundaries['IX'] = i

    return boundaries


def apply_sections(transcript, boundaries, sections):
    """Apply section labels based on boundaries."""

    # Get section order
    section_order = [s['id'] for s in sections]

    # Sort boundaries by paragraph number
    sorted_bounds = sorted(boundaries.items(), key=lambda x: x[1])

    # Add end marker
    sorted_bounds.append(('END', len(transcript)))

    # Apply sections
    for i, (sec_id, start) in enumerate(sorted_bounds[:-1]):
        end = sorted_bounds[i + 1][1]
        for j in range(start, end):
            transcript[j]['section'] = sec_id

    # Fill any gaps at the beginning
    first_boundary = sorted_bounds[0][1] if sorted_bounds else 0
    for j in range(first_boundary):
        transcript[j]['section'] = 'I'

    return transcript


def main():
    project_root = Path(__file__).parent.parent

    # Load processed data
    processed_path = project_root / 'poc' / 'processed_meeting.json'
    with open(processed_path) as f:
        data = json.load(f)

    transcript = data['transcript']
    sections = data['sections']

    print(f"Transcript has {len(transcript)} paragraphs")

    # Find section boundaries
    boundaries = find_section_boundaries(transcript)

    print(f"\nFound boundaries:")
    for sec_id, para_idx in sorted(boundaries.items(), key=lambda x: x[1]):
        time = transcript[para_idx]['time']
        print(f"  {sec_id}: paragraph {para_idx} ({time})")

    # Apply sections
    transcript = apply_sections(transcript, boundaries, sections)

    # Save updated data
    data['transcript'] = transcript
    data['section_boundaries'] = boundaries

    with open(processed_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved to {processed_path}")

    # Show section distribution
    from collections import Counter
    section_counts = Counter(p['section'] for p in transcript)
    print("\nParagraphs per section:")
    for sec_id in [s['id'] for s in sections]:
        count = section_counts.get(sec_id, 0)
        if count > 0:
            print(f"  {sec_id}: {count}")


if __name__ == '__main__':
    main()

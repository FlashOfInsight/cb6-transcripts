# CB6 Transcripts - Project Status

**Last Updated:** 2026-02-10

## What This Project Is

A searchable transcript website for NYC Manhattan Community Board 6 meetings, modeled after citymeetings.nyc. The site has a 3-tier structure:
- **Homepage** → List of meetings with topics from Airtable
- **Meeting page** → Expandable accordion of agenda sections with chapters inside
- **Chapter page** → YouTube embed (timestamped), Summary tab, Full Transcript tab

## Current State

The site is functional and deployed to: https://flashofinsight.github.io/cb6-transcripts/

### What's Working
- YouTube caption download and parsing (VTT → consolidated paragraphs)
- Section boundary detection using regex patterns with constraints
- Agenda items pulled directly from Airtable (via meetings.json)
- **Transition phrase detection** - finds "The second resolution", "Next we have", etc. to align agenda items with transcript
- **Paragraph splitting** - when multiple items are introduced in one breath, splits content at transition phrases
- Summary/Transcript toggle on chapter pages
- Breadcrumb navigation
- Cross-section prev/next navigation
- Accordion state persistence via URL hash
- Pagefind search integration

### Key Files
- `scripts/build_site.py` - Main build script (~1500 lines)
- `src/_data/meetings.json` - Meeting data from Airtable (agenda, topics, videoId)
- `transcripts/*.vtt` - Downloaded YouTube captions
- `dist/` - Generated static site

## Current Challenges

### 1. Timestamp Alignment (Partially Solved)
**Problem:** Matching agenda items to their location in the transcript is hard because:
- Speakers often introduce multiple items in one breath (same paragraph)
- YouTube auto-captions have errors ("Bellevue" → "Bel")
- No clear verbal markers for every agenda item transition

**Current Solution:**
- Transition phrase detection (`find_transition_phrase()`) looks for "The second resolution", etc.
- Fuzzy keyword matching with error mappings for known transcription errors
- Paragraph content splitting when transitions occur mid-paragraph
- Falls back to even division when no markers found

**Remaining Issues:**
- Only works well when speakers say ordinal transitions ("second", "third")
- Some committee sections still rely on keyword detection which can be imprecise
- Item "c) Report" type items rarely have verbal markers

### 2. Transcription Error Mappings
Located in `fuzzy_match_in_text()` around line 388:
```python
error_mappings = {
    'bellevue': ['bel', 'bellview', 'belleview', 'belle view'],
    'helicopter': ['helcopter', 'hellicopter'],  # heliport is a real word, not an error
    'heliport': ['helport', 'heli port'],
    'conedison': ['coned', 'con ed', 'con edison'],
    ...
}
```
Note: "Bellevue" and "heliport" are real words, not errors. The mappings handle cases where YouTube transcribes them incorrectly.

### 3. Section Boundary Detection
The `find_section_boundaries()` function uses regex patterns with paragraph constraints:
- Committee sections (VI-1, VI-2, etc.) must come after the "Committee Reports" section
- "Second Roll Call" and "Adjournment" must be in last 10-15% of transcript
- Enforces chronological ordering

## Build Commands

```bash
# Build site (processes 1 meeting by default)
python scripts/build_site.py 1

# Deploy to GitHub Pages
npx gh-pages -d dist
```

## Potential Future Improvements

1. **Manual timestamps in Airtable** - For high-priority items, allow manual timestamp override
2. **LLM-assisted alignment** - Send transcript + agenda to LLM for semantic boundary detection
3. **Better "Report" item detection** - These items rarely have verbal markers
4. **Multiple meetings** - Currently only processing one meeting; scale to full archive

## Architecture Notes

- Paragraphs are consolidated from VTT segments based on speaker changes (">>" markers)
- Section content is a filtered view of the global paragraph list by section ID
- Chapters are created from agenda items, not auto-extracted from content
- Each chapter stores `para_start`, `para_end` for transcript range and `summary` for display

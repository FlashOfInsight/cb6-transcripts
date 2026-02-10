#!/usr/bin/env node
/**
 * Parse CB6 meeting CSV and generate:
 * 1. src/_data/meetings.json - metadata for all meetings
 * 2. src/meetings/[slug]/index.njk - individual meeting pages
 */

const fs = require('fs');
const path = require('path');

// CSV parser that handles multi-line quoted fields
function parseCSV(content) {
  // Remove BOM if present
  content = content.replace(/^\uFEFF/, '');

  const rows = [];
  let currentRow = [];
  let currentField = '';
  let inQuotes = false;

  for (let i = 0; i < content.length; i++) {
    const char = content[i];
    const nextChar = content[i + 1];

    if (char === '"') {
      if (inQuotes && nextChar === '"') {
        // Escaped quote
        currentField += '"';
        i++;
      } else {
        // Toggle quote mode
        inQuotes = !inQuotes;
      }
    } else if (char === ',' && !inQuotes) {
      currentRow.push(currentField);
      currentField = '';
    } else if ((char === '\n' || (char === '\r' && nextChar === '\n')) && !inQuotes) {
      if (char === '\r') i++; // Skip \n in \r\n
      currentRow.push(currentField);
      if (currentRow.some(f => f.trim())) {
        rows.push(currentRow);
      }
      currentRow = [];
      currentField = '';
    } else if (char !== '\r') {
      currentField += char;
    }
  }

  // Handle last row
  if (currentField || currentRow.length > 0) {
    currentRow.push(currentField);
    if (currentRow.some(f => f.trim())) {
      rows.push(currentRow);
    }
  }

  // Convert to objects using first row as headers
  const headers = rows[0].map(h => h.trim());
  const result = [];

  for (let i = 1; i < rows.length; i++) {
    const row = {};
    headers.forEach((header, index) => {
      row[header] = rows[i][index] || '';
    });
    result.push(row);
  }

  return result;
}

// Extract YouTube video ID from various URL formats
function extractVideoId(url) {
  if (!url) return null;
  const patterns = [
    /youtu\.be\/([^?&]+)/,
    /youtube\.com\/watch\?v=([^&]+)/,
    /youtube\.com\/embed\/([^?&]+)/
  ];
  for (const pattern of patterns) {
    const match = url.match(pattern);
    if (match) return match[1];
  }
  return null;
}

// Parse M/D/YYYY date format to YYYY-MM-DD
function parseDate(dateStr) {
  if (!dateStr) return null;
  const match = dateStr.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/);
  if (match) {
    const [, month, day, year] = match;
    return `${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}`;
  }
  // Fallback to Date parsing
  const date = new Date(dateStr);
  if (isNaN(date.getTime())) return null;
  return date.toISOString().split('T')[0];
}

// Create URL-friendly slug from date and committee
function createSlug(date, committee) {
  const dateStr = parseDate(date);
  if (!dateStr) return null;
  const committeeSlug = (committee || 'general').toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');
  return `${dateStr}-${committeeSlug}`;
}

// Clean VTT content to plain text
function cleanVTT(content) {
  // Remove VTT header
  content = content.replace(/^WEBVTT\nKind:.*\nLanguage:.*\n/m, '');

  // Remove timestamps
  content = content.replace(/\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}.*\n/g, '');

  // Remove inline tags
  content = content.replace(/<[\d:\.]+>/g, '');
  content = content.replace(/<\/?c>/g, '');

  // Replace HTML entities
  content = content.replace(/&gt;/g, '>');
  content = content.replace(/&lt;/g, '<');
  content = content.replace(/&amp;/g, '&');

  // Clean up speaker markers
  content = content.replace(/>>\s*/g, '\n[SPEAKER]: ');

  // Remove duplicate lines (VTT scrolling effect)
  const lines = content.split('\n');
  const cleaned = [];
  let prev = '';
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed && trimmed !== prev) {
      cleaned.push(trimmed);
      prev = trimmed;
    }
  }

  return cleaned.join('\n').replace(/\n{3,}/g, '\n\n').trim();
}

// Main build function
function build() {
  const projectRoot = path.join(__dirname, '..');
  const csvPath = path.join(projectRoot, 'data', 'meetings.csv');
  const transcriptsDir = path.join(projectRoot, 'transcripts');
  const dataOutPath = path.join(projectRoot, 'src', '_data', 'meetings.json');
  const meetingsOutDir = path.join(projectRoot, 'src', 'meetings');

  // Read and parse CSV
  console.log('Reading CSV...');
  const csvContent = fs.readFileSync(csvPath, 'utf-8');
  const rows = parseCSV(csvContent);
  console.log(`Found ${rows.length} meetings in CSV`);

  // Ensure output directories exist
  fs.mkdirSync(path.dirname(dataOutPath), { recursive: true });
  fs.mkdirSync(meetingsOutDir, { recursive: true });

  const meetings = [];
  let transcriptCount = 0;

  for (const row of rows) {
    const dateRaw = row['Meeting Date'];
    const committee = row['Committee'] || 'General';
    const youtubeUrl = row['YouTube Video'];
    const videoId = extractVideoId(youtubeUrl);

    const date = parseDate(dateRaw);
    if (!date) continue;

    const slug = createSlug(dateRaw, committee);
    if (!slug) continue;

    const topics = (row['Topic(s) Covered'] || '')
      .split(',')
      .map(t => t.trim())
      .filter(t => t);

    // Check for transcript
    let transcript = null;
    if (videoId) {
      const vttPath = path.join(transcriptsDir, `${videoId}.vtt`);
      if (fs.existsSync(vttPath)) {
        const vttContent = fs.readFileSync(vttPath, 'utf-8');
        transcript = cleanVTT(vttContent);
        transcriptCount++;
      }
    }

    const meeting = {
      slug,
      date,
      committee,
      topics,
      youtubeUrl: youtubeUrl || null,
      videoId,
      agenda: row['Meeting Agenda'] || null,
      minutesPdf: row['Minutes'] || null,
      hasTranscript: !!transcript
    };

    meetings.push(meeting);

    // Generate meeting page
    const meetingDir = path.join(meetingsOutDir, slug);
    fs.mkdirSync(meetingDir, { recursive: true });

    const pageContent = generateMeetingPage(meeting, transcript);
    fs.writeFileSync(path.join(meetingDir, 'index.njk'), pageContent);
  }

  // Write meetings.json
  fs.writeFileSync(dataOutPath, JSON.stringify(meetings, null, 2));

  console.log(`Generated ${meetings.length} meeting pages`);
  console.log(`Found transcripts for ${transcriptCount} meetings`);
  console.log(`Data written to ${dataOutPath}`);
}

function generateMeetingPage(meeting, transcript) {
  // Escape special characters for YAML and HTML
  const safeTitle = `${meeting.committee} - ${meeting.date}`
    .replace(/"/g, '\\"')
    .replace(/:/g, ' -');

  return `---
layout: base.njk
title: "${safeTitle}"
---

<article class="meeting-page">
  <header class="meeting-header">
    <h1>${meeting.committee} Meeting</h1>
    <div class="meeting-meta">${meeting.date}</div>
    ${meeting.topics.length > 0 ? `
    <div class="meeting-topics">
      ${meeting.topics.map(t => `<span class="topic-tag">${t}</span>`).join('\n      ')}
    </div>
    ` : ''}
    <div class="meeting-links">
      ${meeting.youtubeUrl ? `<a href="${meeting.youtubeUrl}" target="_blank">Watch on YouTube</a>` : ''}
    </div>
  </header>

  ${meeting.agenda ? `
  <section class="transcript-section">
    <h2>Agenda</h2>
    <div class="transcript">${meeting.agenda.replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br>')}</div>
  </section>
  ` : ''}

  ${transcript ? `
  <section class="transcript-section" data-pagefind-body>
    <h2>Transcript</h2>
    <div class="transcript">${transcript.substring(0, 50000).replace(/</g, '&lt;').replace(/>/g, '&gt;')}</div>
  </section>
  ` : `
  <section class="transcript-section">
    <h2>Transcript</h2>
    <p style="color: var(--text-light);">Transcript not yet available for this meeting.</p>
  </section>
  `}
</article>
`;
}

build();

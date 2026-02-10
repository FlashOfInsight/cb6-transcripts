#!/bin/bash
# Deploy CB6 Transcripts to GitHub Pages

set -e

# Build site (adjust number as needed)
echo "Building site..."
python3 scripts/build_site.py "${1:-5}"

# Deploy to gh-pages branch
cd dist
git init
git add -A
git commit -m "Deploy CB6 Transcripts"
git branch -M gh-pages
git push -f git@github.com:YOUR_USERNAME/cb6-transcripts.git gh-pages

echo "Deployed! Visit: https://YOUR_USERNAME.github.io/cb6-transcripts/"

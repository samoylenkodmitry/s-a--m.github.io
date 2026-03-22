#!/usr/bin/env python3
import re
import requests
import time
import os
from PIL import Image
from io import BytesIO
import sys
import hashlib
from datetime import datetime
import json

def download_and_convert_image(url, local_path, quality=75, timeout=15):
    """Download image from URL, convert to webp"""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        
        # Convert to RGB if necessary (handle various formats)
        if img.mode in ("RGBA", "P", "LA", "PA", "1", "L"):
            rgb_img = Image.new("RGB", img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = rgb_img
        
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        img.save(local_path, 'WEBP', quality=quality, method=6)
        return True
    except Exception as e:
        print(f"✗ Error {url[:50]}: {str(e)[:40]}")
        return False

def process_markdown_file(file_path, image_folder, resume_data_file=None):
    """Process markdown file to download images and update links"""
    
    # Load resume data if it exists
    resume_data = {}
    if resume_data_file and os.path.exists(resume_data_file):
        try:
            with open(resume_data_file, 'r') as f:
                resume_data = json.load(f)
                print(f"Loaded {len(resume_data)} previously processed URLs from resume data")
        except:
            resume_data = {}
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all image links
    image_pattern = r'!\[([^\]]*)\]\((https?://[^\)]+)\)'
    matches = list(re.finditer(image_pattern, content))
    
    # Filter only external URLs (not already local)
    to_process = []
    for match in matches:
        alt_text = match.group(1)
        url = match.group(2)
        if 'assets/leetcode_daily_images' not in url:
            to_process.append((alt_text, url, match))
    
    if not to_process:
        print("✓ All images are already processed!")
        return
    
    print(f"Found {len(to_process)} images to process")
    os.makedirs(image_folder, exist_ok=True)
    
    # Track replacements
    replacements = {}
    processed = 0
    skipped = 0
    
    for idx, (alt_text, url, match) in enumerate(to_process, 1):
        # Check if already processed
        if url in resume_data:
            print(f"[{idx}/{len(to_process)}] ⊐ Already processed: {url[:40]}...")
            replacements[url] = resume_data[url]
            skipped += 1
            continue
        
        # Generate local filename
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        local_filename = f"{url_hash}.webp"
        local_path = os.path.join(image_folder, local_filename)
        
        # Skip if file already exists
        if os.path.exists(local_path):
            new_url = os.path.join(image_folder, local_filename)
            resume_data[url] = new_url
            replacements[url] = new_url
            print(f"[{idx}/{len(to_process)}] ✓ File exists: {local_filename}")
            processed += 1
            continue
        
        # Download and convert
        print(f"[{idx}/{len(to_process)}] ↓ Downloading {url[:40]}...", end=' ', flush=True)
        if download_and_convert_image(url, local_path):
            new_url = os.path.join(image_folder, local_filename)
            replacements[url] = new_url
            resume_data[url] = new_url
            print("✓")
            processed += 1
        else:
            print("✗")
        
        # Rate limiting - every 5 images, save progress
        if idx % 5 == 0:
            if resume_data_file:
                with open(resume_data_file, 'w') as f:
                    json.dump(resume_data, f)
                print(f"  [Progress saved: {processed}/{len(to_process)} successful]")
            time.sleep(0.2)  # Small delay between batches
    
    # Update markdown file with all replacements
    print(f"\nUpdating markdown file with {len(replacements)} image replacements...")
    for url, new_path in replacements.items():
        # Find all occurrences of this URL in images
        pattern = f'(!\\[[^\\]]*\\])\\({re.escape(url)}\\)'
        content = re.sub(pattern, f'\\1({new_path})', content)
    
    # Write updated content
    with open(file_path, 'w') as f:
        f.write(content)
    
    # Save final resume data
    if resume_data_file:
        with open(resume_data_file, 'w') as f:
            json.dump(resume_data, f)
    
    print(f"\n✓ Completed!")
    print(f"  - Downloaded: {processed} new images")
    print(f"  - Already existed: {skipped} images")
    print(f"  - Updated markdown with {len(replacements)} replacements")
    print(f"  - Markdown file: {file_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <markdown_file> <image_folder> [--resume]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    image_folder = sys.argv[2]
    resume_data_file = os.path.join(image_folder, '.resume_data.json') if '--resume' in sys.argv else None
    
    process_markdown_file(file_path, image_folder, resume_data_file)

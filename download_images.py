#!/usr/bin/env python3
import re
import requests
import time
import os
from PIL import Image
import sys

def download_and_convert_image(url, local_path, quality=80):
    """Download image from URL, convert to webp, save to local_path"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Open image with PIL
        from io import BytesIO
        img = Image.open(BytesIO(response.content))
        
        # Convert to RGB if necessary
        if img.mode in ("RGBA", "P", "LA", "PA"):
            img = img.convert("RGB")
        
        # Save as webp with optimized settings
        img.save(local_path, 'WEBP', quality=quality, method=6)
        print(f"Downloaded and saved: {local_path}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def process_markdown_file(file_path, image_folder, test_mode=False, max_test=3):
    """Process the markdown file to download images and update links"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all image links
    image_pattern = r'!\[([^\]]*)\]\((https?://[^\)]+)\)'
    matches = re.findall(image_pattern, content)
    
    if test_mode:
        matches = matches[:max_test]
        print(f"Test mode: processing first {max_test} images")
    
    # Create image folder if not exists
    os.makedirs(image_folder, exist_ok=True)
    
    updated_content = content
    processed = 0
    
    for alt_text, url in matches:
        # Generate local filename
        # Use a simple name based on url hash or something
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        ext = '.webp'
        local_filename = f"{url_hash}{ext}"
        local_path = os.path.join(image_folder, local_filename)
        
        # Download and convert
        if download_and_convert_image(url, local_path):
            # Update the link in content
            old_link = f'![{alt_text}]({url})'
            new_link = f'![{alt_text}]({os.path.join(image_folder, local_filename)})'
            print(f"old_link: {repr(old_link)}")
            print(f"in content: {old_link in content}")
            updated_content = updated_content.replace(old_link, new_link)
            processed += 1
            print(f"Updated link: {old_link} -> {new_link}")
        else:
            print(f"Failed to process: {url}")
        
        # Delay to not be spammy
        time.sleep(1)
    
    # Write back the updated content
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print(f"Processed {processed} images")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <markdown_file> <image_folder> [--test]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    image_folder = sys.argv[2]
    test_mode = '--test' in sys.argv
    
    process_markdown_file(file_path, image_folder, test_mode)
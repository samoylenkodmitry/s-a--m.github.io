#!/usr/bin/env python3
import re
import requests
import time
import os
from PIL import Image
from io import BytesIO
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

class ImageDownloader:
    def __init__(self, image_folder, delay=0.5, max_workers=3):
        self.image_folder = image_folder
        self.delay = delay
        self.max_workers = max_workers
        self.lock = Lock()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def download_and_convert_image(self, url):
        """Download image from URL, convert to webp"""
        try:
            time.sleep(self.delay)
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content))
            
            if img.mode in ("RGBA", "P", "LA", "PA"):
                img = img.convert("RGB")
            
            import hashlib
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            local_path = os.path.join(self.image_folder, f"{url_hash}.webp")
            
            img.save(local_path, 'WEBP', quality=75, method=6)
            print(f"✓ Downloaded: {url_hash}.webp")
            return (url, local_path, True)
        except Exception as e:
            print(f"✗ Error: {url[:60]}... {str(e)[:50]}")
            return (url, None, False)

def process_markdown_file(file_path, image_folder):
    """Process the markdown file to download images and update links"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all image links
    image_pattern = r'!\[([^\]]*)\]\((https?://[^\)]+)\)'
    matches = re.findall(image_pattern, content)
    
    # Filter only externa URLs (not already local)
    matches = [(alt, url) for alt, url in matches if 'assets/leetcode_daily_images' not in url]
    
    if not matches:
        print("No external images to download.")
        return
    
    print(f"Found {len(matches)} external images to download")
    
    # Create image folder if not exists
    os.makedirs(image_folder, exist_ok=True)
    
    downloader = ImageDownloader(image_folder, delay=0.3, max_workers=3)
    updated_content = content
    processed = 0
    
    # Download images with threading
    with ThreadPoolExecutor(max_workers=downloader.max_workers) as executor:
        futures = {executor.submit(downloader.download_and_convert_image, url): (alt, url) 
                   for alt, url in matches}
        
        for future in as_completed(futures):
            alt, url = futures[future]
            try:
                orig_url, local_path, success = future.result()
                if success:
                    old_link = f'![{alt}]({url})'
                    new_link = f'![{alt}]({os.path.join(image_folder, os.path.basename(local_path))})'
                    updated_content = updated_content.replace(old_link, new_link)
                    processed += 1
                    if processed % 10 == 0:
                        print(f"Progress: {processed}/{len(matches)} images processed")
            except Exception as e:
                print(f"Error processing future: {e}")
    
    # Write back the updated content
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print(f"✓ Completed: {processed}/{len(matches)} images downloaded and linked")
    print(f"Updated markdown file: {file_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <markdown_file> <image_folder>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    image_folder = sys.argv[2]
    
    process_markdown_file(file_path, image_folder)

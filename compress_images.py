#!/usr/bin/env python3
import os
from PIL import Image
import sys

def compress_images(image_folder, compression_level='2x'):
    """
    Recompress WebP images
    compression_level options:
    - '2x': quality=50 (target ~75KB average)
    - '4x': quality=25 (target ~37KB average)
    - '8x': quality=12 (target ~18KB average)
    """
    
    quality_map = {
        '2x': 50,
        '4x': 25,
        '8x': 12,
    }
    
    if compression_level not in quality_map:
        print(f"Invalid compression level. Use '2x' or '4x'")
        sys.exit(1)
    
    target_quality = quality_map[compression_level]
    
    webp_files = []
    for f in os.listdir(image_folder):
        if f.endswith('.webp'):
            webp_files.append(os.path.join(image_folder, f))
    
    if not webp_files:
        print(f"No WebP files found in {image_folder}")
        return
    
    print(f"Recompressing {len(webp_files)} images to {compression_level} compression (quality={target_quality})...")
    print(f"This may take a few minutes...")
    print()
    
    total_before = sum(os.path.getsize(f) for f in webp_files)
    processed = 0
    errors = 0
    
    for idx, filepath in enumerate(webp_files, 1):
        try:
            before_size = os.path.getsize(filepath)
            filename = os.path.basename(filepath)
            
            # Load image
            img = Image.open(filepath)
            
            # Recompress
            img.save(filepath, 'WEBP', quality=target_quality, method=6)
            
            after_size = os.path.getsize(filepath)
            reduction = (1 - after_size/before_size) * 100
            
            if idx % 50 == 0:
                print(f"[{idx}/{len(webp_files)}] Processed {filename}: {reduction:.1f}% reduction")
            
            processed += 1
            
        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")
            errors += 1
    
    total_after = sum(os.path.getsize(f) for f in webp_files)
    total_reduction = (1 - total_after/total_before) * 100
    
    print()
    print(f"✓ Compression complete!")
    print(f"  Files processed: {processed}/{len(webp_files)}")
    print(f"  Errors: {errors}")
    print(f"  Before: {total_before / (1024**2):.1f} MB")
    print(f"  After: {total_after / (1024**2):.1f} MB")
    print(f"  Total reduction: {total_reduction:.1f}%")
    print(f"  Space saved: {(total_before - total_after) / (1024**2):.1f} MB")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compress_images.py <image_folder> <compression_level>")
        print("compression_level: 2x, 4x, or 8x")
        sys.exit(1)
    
    image_folder = sys.argv[1]
    compression_level = sys.argv[2]
    
    compress_images(image_folder, compression_level)

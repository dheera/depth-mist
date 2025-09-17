#!/usr/bin/env python3
"""
Mist Effect Pipeline using Depth Anything V2

This script creates an artistic mist/fog effect by:
1. Running Depth Anything V2 on input images
2. Applying depth-dependent contrast reduction
3. Saving the processed images

Usage:
    python mist_effect.py --input image.jpg --output misty_image.jpg
    python mist_effect.py --input-dir ./photos --output-dir ./misty_photos
"""

import argparse
import cv2
import glob
import numpy as np
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt

import sys
sys.path.append('Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2


def load_depth_model(encoder='vitl', checkpoint_path='depth_anything_v2_vitl.pth'):
    """Load the Depth Anything V2 model"""
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    model = DepthAnythingV2(**model_configs[encoder])
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(DEVICE).eval()
    
    return model, DEVICE


def apply_mist_effect(image, depth_map, contrast_reduction_factor=0.5, brightness_adjustment=0.0):
    """
    Apply mist effect based on depth information
    
    Args:
        image: Input image (numpy array, BGR format)
        depth_map: Depth map from Depth Anything V2 (numpy array)
        contrast_reduction_factor: How much to reduce contrast for distant objects (0.0-1.0)
        brightness_adjustment: Brightness adjustment for distant objects (-1.0 to 1.0)
    
    Returns:
        Processed image with mist effect
    """
    # Normalize depth map to 0-1 range
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    # Convert image to float for processing
    image_float = image.astype(np.float32) / 255.0
    
    # Create mist effect mask - distant objects get more effect
    # Invert depth_normalized so that distant objects (lower depth values) get higher mist effect
    mist_mask = 1.0 - depth_normalized
    
    # Apply contrast reduction based on depth
    # For each pixel, reduce contrast more for distant objects
    processed_image = image_float.copy()
    
    for c in range(3):  # Process each color channel
        channel = image_float[:, :, c]
        
        # Calculate mean for contrast adjustment
        channel_mean = np.mean(channel)
        
        # Apply contrast reduction: closer to mean for distant pixels
        contrast_factor = 1.0 - (mist_mask * contrast_reduction_factor)
        processed_channel = channel_mean + (channel - channel_mean) * contrast_factor
        
        # Apply brightness adjustment for distant objects
        brightness_factor = 1.0 + (mist_mask * brightness_adjustment)
        processed_channel = processed_channel * brightness_factor
        
        # Clamp values to valid range
        processed_channel = np.clip(processed_channel, 0.0, 1.0)
        
        processed_image[:, :, c] = processed_channel
    
    # Convert back to uint8
    processed_image = (processed_image * 255).astype(np.uint8)
    
    return processed_image


def process_single_image(model, device, image_path, output_path, 
                        contrast_reduction_factor=0.5, brightness_adjustment=0.0,
                        input_size=518):
    """Process a single image with mist effect"""
    print(f"Processing: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return False
    
    # Get depth map
    depth_map = model.infer_image(image, input_size)
    
    # Apply mist effect
    misty_image = apply_mist_effect(image, depth_map, contrast_reduction_factor, brightness_adjustment)
    
    # Save result
    cv2.imwrite(output_path, misty_image)
    print(f"Saved: {output_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Apply mist effect to images using Depth Anything V2')
    
    # Input/Output options
    parser.add_argument('--input', type=str, help='Input image file')
    parser.add_argument('--input-dir', type=str, help='Input directory containing images')
    parser.add_argument('--output', type=str, help='Output image file (required if --input is used)')
    parser.add_argument('--output-dir', type=str, default='./misty_output', 
                       help='Output directory (default: ./misty_output)')
    
    # Model options
    parser.add_argument('--encoder', type=str, default='vitl', 
                       choices=['vits', 'vitb', 'vitl', 'vitg'],
                       help='Depth Anything V2 encoder (default: vitl)')
    parser.add_argument('--checkpoint', type=str, default='depth_anything_v2_vitl.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--input-size', type=int, default=518,
                       help='Input size for depth estimation (default: 518)')
    
    # Effect parameters
    parser.add_argument('--contrast-reduction', type=float, default=0.5,
                       help='Contrast reduction factor for distant objects (0.0-1.0, default: 0.5)')
    parser.add_argument('--brightness-adjustment', type=float, default=0.0,
                       help='Brightness adjustment for distant objects (-1.0 to 1.0, default: 0.0)')
    
    # File filtering
    parser.add_argument('--extensions', type=str, nargs='+', 
                       default=['.jpg', '.jpeg', '.png', '.tiff', '.tif'],
                       help='File extensions to process')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input and not args.output:
        print("Error: --output is required when using --input")
        return
    
    if not args.input and not args.input_dir:
        print("Error: Either --input or --input-dir must be specified")
        return
    
    # Load model
    print("Loading Depth Anything V2 model...")
    model, device = load_depth_model(args.encoder, args.checkpoint)
    print(f"Model loaded on device: {device}")
    
    # Create output directory if needed
    if args.input_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Process images
    if args.input:
        # Single image
        success = process_single_image(
            model, device, args.input, args.output,
            args.contrast_reduction, args.brightness_adjustment, args.input_size
        )
        if not success:
            return
    else:
        # Directory of images
        image_files = []
        for ext in args.extensions:
            pattern = os.path.join(args.input_dir, f'**/*{ext}')
            image_files.extend(glob.glob(pattern, recursive=True))
            pattern = os.path.join(args.input_dir, f'**/*{ext.upper()}')
            image_files.extend(glob.glob(pattern, recursive=True))
        
        if not image_files:
            print(f"No images found in {args.input_dir}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        for i, image_path in enumerate(image_files):
            # Create output filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(args.output_dir, f"{base_name}_misty.jpg")
            
            success = process_single_image(
                model, device, image_path, output_path,
                args.contrast_reduction, args.brightness_adjustment, args.input_size
            )
            
            if not success:
                continue
            
            print(f"Progress: {i+1}/{len(image_files)}")
    
    print("Processing complete!")


if __name__ == '__main__':
    main()

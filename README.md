# Mist Effect Pipeline

This project creates artistic mist/fog and Orton effects on photographs using Depth Anything V2 for depth estimation and depth-dependent processing.

## How it Works

1. **Input**: Takes a photograph (JPG, TIFF, PNG, etc.)
2. **Depth Estimation**: Uses Depth Anything V2 to generate a depth map
3. **Mist Effect**: Applies depth-dependent contrast reduction - distant objects get more contrast reduction, creating a natural mist/fog effect
4. **Variable Orton Effect**: Applies depth-aware blur and overlay effects that respect depth ordering - distant objects get more blur/overlay, but close objects are never blurred by distant objects behind them
5. **Output**: Saves the processed image with the combined effects

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r Depth-Anything-V2/requirements.txt
```

3. Download the Depth Anything V2 model checkpoint:
```bash
# The checkpoint file should be placed in the root directory
# depth_anything_v2_vitl.pth (already present in this repository)
```

## Usage

### Single Image Processing

```bash
python mist_effect.py --input path/to/image.jpg --output misty_image.jpg
```

### Batch Processing

```bash
python mist_effect.py --input-dir path/to/images --output-dir path/to/output
```

### Parameters

- `--contrast-reduction`: How much to reduce contrast for distant objects (0.0-1.0, default: 0.5)
- `--brightness-adjustment`: Brightness adjustment for distant objects (-1.0 to 1.0, default: 0.0)
- `--orton-blur`: Maximum blur strength for Orton effect (0.0-1.0, default: 0.0)
- `--orton-overlay`: Maximum overlay opacity for Orton effect (0.0-1.0, default: 0.0)
- `--encoder`: Depth Anything V2 encoder ('vits', 'vitb', 'vitl', 'vitg', default: 'vitl')
- `--input-size`: Input size for depth estimation (default: 518)

### Examples

```bash
# Basic mist effect
python mist_effect.py --input photo.jpg --output misty_photo.jpg

# Strong mist effect
python mist_effect.py --input photo.jpg --output misty_photo.jpg --contrast-reduction 0.8 --brightness-adjustment 0.1

# Orton effect only
python mist_effect.py --input photo.jpg --output orton_photo.jpg --orton-blur 0.6 --orton-overlay 0.3

# Combined mist and Orton effects
python mist_effect.py --input photo.jpg --output combined_photo.jpg --contrast-reduction 0.3 --orton-blur 0.4 --orton-overlay 0.2

# Process all images in a directory
python mist_effect.py --input-dir ./photos --output-dir ./processed_photos

# Use different encoder for faster processing
python mist_effect.py --input photo.jpg --output misty_photo.jpg --encoder vits
```

## Effect Parameters

- **Contrast Reduction (0.0-1.0)**: 
  - 0.0 = No contrast reduction (no mist effect)
  - 0.5 = Moderate mist effect (default)
  - 1.0 = Maximum contrast reduction (strong mist effect)

- **Brightness Adjustment (-1.0 to 1.0)**:
  - 0.0 = No brightness change (default)
  - Positive values = Brighten distant objects
  - Negative values = Darken distant objects

- **Orton Blur Strength (0.0-1.0)**:
  - 0.0 = No blur (default)
  - 0.5 = Moderate blur for distant objects
  - 1.0 = Maximum blur for distant objects

- **Orton Overlay Opacity (0.0-1.0)**:
  - 0.0 = No overlay effect (default)
  - 0.3 = Subtle overlay with brightness and saturation boost
  - 1.0 = Maximum overlay effect
  - **Note**: Uses "lighten only" blending - only brightens, never darkens

## Technical Details

### Mist Effect
The mist effect is achieved by:

1. **Depth Normalization**: The depth map is normalized to 0-1 range
2. **Contrast Reduction**: For each pixel, contrast is reduced based on its depth value
3. **Brightness Adjustment**: Optional brightness adjustment for distant objects
4. **Channel Processing**: Each color channel (R, G, B) is processed independently

The algorithm uses the formula:
```
processed_pixel = mean + (original_pixel - mean) * contrast_factor
```

Where `contrast_factor = 1.0 - (depth_normalized * contrast_reduction_factor)`

### Variable Orton Effect
The Orton effect is achieved by:

1. **Sharp Base Image**: Keeps the original image crisp as the foundation
2. **Single Blur Pass**: Creates one blurred version of the entire image
3. **Depth-Aware Blending**: Blends blurred and sharp versions based on depth
4. **Natural Depth Ordering**: Distant objects get more blur, close objects stay sharp
5. **Lighten-Only Overlay**: Brightness and saturation enhancement that only brightens, never darkens

**Key Benefits**:
- **No bleeding**: Dark areas never bleed into light areas
- **Natural depth-of-field**: Mimics real camera behavior
- **Crisp foregrounds**: Close objects remain sharp
- **Ethereal backgrounds**: Distant objects get soft, dreamy appearance

This creates a natural depth-aware effect where distant objects appear softer and more ethereal while maintaining sharpness in foreground elements.

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- PIL/Pillow

## Model

This project uses Depth Anything V2 (ViT-L encoder) for depth estimation. The model checkpoint should be placed in the root directory as `depth_anything_v2_vitl.pth`.
# depth-mist

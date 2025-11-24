"""Generate synthetic corridor images for training."""

import numpy as np
import cv2
from pathlib import Path
import argparse
from typing import Tuple
import json


def generate_corridor_image(
    width: int = 224,
    height: int = 224,
    heading_angle: float = 0.0,
    corridor_width: float = 0.6,
    noise_level: float = 0.1,
) -> Tuple[np.ndarray, float]:
    """Generate a synthetic corridor image.
    
    Args:
        width: Image width
        height: Image height
        heading_angle: Heading angle in radians (0 = centered)
        corridor_width: Width of corridor (0-1, relative to image)
        noise_level: Amount of noise to add
        
    Returns:
        Tuple of (image, heading_angle)
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Compute vanishing point based on heading
    center_x = width / 2
    center_y = height / 2
    vanishing_y = height * 0.3  # Vanishing point Y position
    
    # Offset vanishing point based on heading
    # Positive heading = turn right (vanishing point moves left)
    offset = heading_angle * width * 0.3
    vanishing_x = center_x - offset
    
    # Draw corridor walls
    wall_color = (100, 100, 100)  # Gray walls
    floor_color = (50, 50, 50)  # Dark gray floor
    
    # Left wall
    left_top = (int(width * (1 - corridor_width) / 2), 0)
    left_bottom = (int(vanishing_x - corridor_width * width / 2), int(vanishing_y))
    
    # Right wall
    right_top = (int(width * (1 + corridor_width) / 2), 0)
    right_bottom = (int(vanishing_x + corridor_width * width / 2), int(vanishing_y))
    
    # Draw floor
    floor_points = np.array([
        left_bottom,
        right_bottom,
        (width, height),
        (0, height),
    ], np.int32)
    cv2.fillPoly(image, [floor_points], floor_color)
    
    # Draw left wall
    left_wall_points = np.array([
        left_top,
        left_bottom,
        (0, height),
        (0, 0),
    ], np.int32)
    cv2.fillPoly(image, [left_wall_points], wall_color)
    
    # Draw right wall
    right_wall_points = np.array([
        right_top,
        right_bottom,
        (width, height),
        (width, 0),
    ], np.int32)
    cv2.fillPoly(image, [right_wall_points], wall_color)
    
    # Add texture/noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * 255, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add some lighting variation
    brightness = np.random.uniform(0.8, 1.2)
    image = np.clip(image * brightness, 0, 255).astype(np.uint8)
    
    return image, heading_angle


def generate_dataset(
    output_dir: Path,
    num_samples: int = 10000,
    train_split: float = 0.8,
    width: int = 224,
    height: int = 224,
):
    """Generate synthetic dataset.
    
    Args:
        output_dir: Output directory
        num_samples: Number of samples to generate
        train_split: Fraction for training set
        width: Image width
        height: Image height
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    num_train = int(num_samples * train_split)
    num_val = num_samples - num_train
    
    metadata_train = []
    metadata_val = []
    
    print(f"Generating {num_train} training samples...")
    for i in range(num_train):
        # Random heading angle between -pi/4 and pi/4
        heading = np.random.uniform(-np.pi / 4, np.pi / 4)
        image, _ = generate_corridor_image(
            width=width, height=height, heading_angle=heading
        )
        
        image_path = train_dir / f"image_{i:06d}.png"
        cv2.imwrite(str(image_path), image)
        
        metadata_train.append({
            "image_path": str(image_path.relative_to(output_dir)),
            "heading": float(heading),
        })
        
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_train} samples")
    
    print(f"Generating {num_val} validation samples...")
    for i in range(num_val):
        heading = np.random.uniform(-np.pi / 4, np.pi / 4)
        image, _ = generate_corridor_image(
            width=width, height=height, heading_angle=heading
        )
        
        image_path = val_dir / f"image_{i:06d}.png"
        cv2.imwrite(str(image_path), image)
        
        metadata_val.append({
            "image_path": str(image_path.relative_to(output_dir)),
            "heading": float(heading),
        })
        
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_val} samples")
    
    # Save metadata
    with open(output_dir / "train_metadata.json", "w") as f:
        json.dump(metadata_train, f, indent=2)
    
    with open(output_dir / "val_metadata.json", "w") as f:
        json.dump(metadata_val, f, indent=2)
    
    print(f"\nDataset generated successfully!")
    print(f"  Training samples: {num_train}")
    print(f"  Validation samples: {num_val}")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic corridor dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic",
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Total number of samples to generate",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data for training",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="Image width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="Image height",
    )
    
    args = parser.parse_args()
    
    generate_dataset(
        output_dir=Path(args.output_dir),
        num_samples=args.num_samples,
        train_split=args.train_split,
        width=args.width,
        height=args.height,
    )


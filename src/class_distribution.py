import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import argparse
from dataset.video_dataset import collect_video_samples


def analyze_class_distribution(data_dir: str, num_classes: int = 33):
    """
    Analyzes and visualizes the class distribution from a dataset directory.
    
    Args:
        data_dir: Path to the dataset directory
        num_classes: Number of classes in the dataset
    """
    print(f"\n📊 Analyzing class distribution from: {data_dir}")
    
    # Collect all samples
    samples = collect_video_samples(Path(data_dir))
    print(f"✓ Found {len(samples)} samples")
    
    # Extract class labels from samples
    class_labels = [sample[1] for sample in samples]  # sample format: (video_path, class_id)
    
    # Count distribution
    distribution = Counter(class_labels)
    total_samples = len(class_labels)
    
    # Print statistics
    print(f"\n📈 Class Distribution Statistics:")
    print(f"   Total samples: {total_samples}")
    print(f"   Number of classes present: {len(distribution)}")
    print(f"   Min samples per class: {min(distribution.values())}")
    print(f"   Max samples per class: {max(distribution.values())}")
    print(f"   Mean samples per class: {np.mean(list(distribution.values())):.2f}")
    
    print(f"\n📋 Detailed Distribution:")
    for class_id in sorted(distribution.keys()):
        count = distribution[class_id]
        percentage = (count / total_samples) * 100
        bar = "█" * int(percentage / 2)  # Scale bar to 50 chars max
        print(f"   Class {class_id:2d}: {count:4d} samples ({percentage:5.2f}%) {bar}")
    
    # Create visualization
    print(f"\n📉 Generating visualization...")
    class_ids = sorted(distribution.keys())
    counts = [distribution[cid] for cid in class_ids]
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(class_ids, counts, color='steelblue', alpha=0.8, edgecolor='navy')
    
    # Color bars differently to show balance
    max_count = max(counts)
    min_count = min(counts)
    for bar, count in zip(bars, counts):
        if count < min_count * 1.5:
            bar.set_color('salmon')
        elif count > max_count * 0.75:
            bar.set_color('lightgreen')
    
    plt.xlabel("Class ID", fontsize=12, fontweight='bold')
    plt.ylabel("Number of Samples", fontsize=12, fontweight='bold')
    plt.title(f"Class Distribution - {Path(data_dir).name}", fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Save figure
    save_path = f"class_distribution_{Path(data_dir).name}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved to '{save_path}'")
    
    return distribution


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze class distribution in a dataset")
    parser.add_argument(
        "data_dir",
        type=str,
        help="Path to the dataset directory (e.g., processed_data/val2/val)"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=33,
        help="Number of classes in the dataset (default: 33)"
    )
    
    args = parser.parse_args()
    
    analyze_class_distribution(args.data_dir, args.num_classes)

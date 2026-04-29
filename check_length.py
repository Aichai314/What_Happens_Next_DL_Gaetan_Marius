from pathlib import Path
import statistics

def analyze_video_lengths(folder_path: str | Path):
    path = Path(folder_path)
    if not path.exists():
        print(f"⚠️ Directory not found: {path}")
        return

    frame_counts = []
    
    # Recursively find all directories
    for subdir in path.rglob('*'):
        if subdir.is_dir():
            # Count standard image formats (matching your video_dataset.py)
            images = (
                list(subdir.glob("*.jpg")) + 
                list(subdir.glob("*.jpeg")) + 
                list(subdir.glob("*.png")) + 
                list(subdir.glob("*.webp"))
            )
            count = len(images)
            
            # If the folder has images, we treat it as a video sequence
            if count > 0:
                frame_counts.append(count)

    if not frame_counts:
        print(f"❌ No video frames found in {path.name}")
        return

    avg_len = statistics.mean(frame_counts)
    min_len = min(frame_counts)
    max_len = max(frame_counts)
    
    print(f"--- 📊 Stats for '{path.name}' ---")
    print(f"  Total Videos : {len(frame_counts):,}")
    print(f"  Avg Frames   : {avg_len:.1f}")
    print(f"  Min Frames   : {min_len}")
    print(f"  Max Frames   : {max_len}\n")

if __name__ == "__main__":
    # Adjust these paths to match where your data actually lives
    directories = [
        "processed_data/val2/train", 
        "processed_data/val2/val", 
        "processed_data/val2/test"
    ]
    
    for d in directories:
        analyze_video_lengths(d)
import json
from rapidfuzz import process
import os
import glob
from PIL import Image
import pandas as pd
from datasets import load_dataset

# --- Step 1: Load video metadata (run this externally first):
# yt-dlp --flat-playlist -J "https://www.youtube.com/@SynthesisWorkshopVideos" > channel_videos.json
with open("channel_videos.json") as f:
    data = json.load(f)
    videos = data["entries"]

video_dicts = []
for v in videos:
    video_id = v["id"]
    title = v["title"]
    link = f"https://www.youtube.com/watch?v={video_id}"
    video_dicts.append({
        "id": video_id,
        "title_video": title,
        "link_video": link
    })

# --- Step 2: Load triplets (speaker, caption, img_path, timestamp) from local directory
base_dir = "data/img_t_pairs"

triplets = []
speaker_dirs = glob.glob(os.path.join(base_dir, "*"))
for speaker_dir in speaker_dirs:
    speaker = os.path.basename(speaker_dir)
    image_dir = os.path.join(speaker_dir, "image")
    text_dir = os.path.join(speaker_dir, "text")
    timestamps_path = os.path.join(speaker_dir, "timestamps.csv")

    if not (os.path.isdir(image_dir) and os.path.isdir(text_dir) and os.path.isfile(timestamps_path)):
        continue

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))])
    text_files = sorted([f for f in os.listdir(text_dir) if f.endswith('.txt')])
    timestamps = pd.read_csv(timestamps_path)["timestamp"].tolist()

    n = min(len(image_files), len(text_files), len(timestamps))
    for i in range(n):
        img_path = os.path.join(image_dir, image_files[i])
        txt_path = os.path.join(text_dir, text_files[i])
        timestamp = timestamps[i]
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
            triplets.append((speaker, caption, img_path, timestamp))
        except Exception as e:
            print(f"Error loading {img_path} or {txt_path}: {e}")

print(f"Loaded {len(triplets)} slide-caption-timestamp triplets")

# --- Step 3: Fuzzy-match speaker to video title and build quadruplets with dynamic thresholds
quadruplets = []
no_match_count = 0
total_triplets = len(triplets)

for speaker, caption, img_path, timestamp in triplets:
    # Try different thresholds to ensure we always get a match
    thresholds = [80, 70, 60, 50, 40, 30, 20, 10, 0]  # Start high, go lower
    youtube_url = None
    best_match = None
    best_score = 0
    best_idx = 0
    
    for threshold in thresholds:
        match_result = process.extractOne(
        speaker, [v["title_video"] for v in video_dicts]
    )
        
        if match_result is not None:
            match, score, idx = match_result
            if score > threshold:
                best_match = match
                best_score = score
                best_idx = idx
                break
    
    # If we found a match at any threshold, use it
    if best_match is not None:
        video = video_dicts[best_idx]
        t = int(round(float(timestamp)))
        youtube_url = f"{video['link_video']}&t={t}s"
        print(f"✅ Matched '{speaker}' to '{best_match}' (score: {best_score:.1f})")
    else:
        # If still no match, take the best available match regardless of score
        match_result = process.extractOne(
            speaker, [v["title_video"] for v in video_dicts]
        )
        if match_result is not None:
            match, score, idx = match_result
            video = video_dicts[idx]
            t = int(round(float(timestamp)))
            youtube_url = f"{video['link_video']}&t={t}s"
            print(f"⚠️  Forced match '{speaker}' to '{match}' (score: {score:.1f})")
            no_match_count += 1
        else:
            # This should never happen, but just in case
            print(f"❌ No match found for '{speaker}' - using first video")
            video = video_dicts[0]
            t = int(round(float(timestamp)))
            youtube_url = f"{video['link_video']}&t={t}s"
            no_match_count += 1
    
    quadruplets.append((speaker, caption, img_path, timestamp, youtube_url))

print(f"✅ Processed {len(triplets)} triplets")
print(f"⚠️  {no_match_count} required forced matching (low scores)")
print(f"📊 Success rate: {((total_triplets - no_match_count) / total_triplets * 100):.1f}%")

# --- Step 4: Save quadruplets to JSON
with open("triplets_with_youtube.json", "w") as f:
    json.dump([
        {
            "speaker": q[0],
            "caption": q[1],
            "img_path": q[2],
            "timestamp": q[3],
            "youtube_url": q[4]
        }
        for q in quadruplets
    ], f, indent=2)

print(f"Saved {len(quadruplets)} quadruplets with YouTube links to triplets_with_youtube.json")

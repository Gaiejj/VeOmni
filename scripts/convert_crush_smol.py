"""Convert finetrainers/crush-smol dataset to VeOmni parquet format.

Expected output columns:
  - prompt: str (video description)
  - video_bytes: bytes (raw mp4 file content)
  - source: str (dataset name)
"""

import os
import pyarrow as pa
import pyarrow.parquet as pq

DATASET_DIR = "/mnt/users/jiayi/.cache/veomni/crush-smol"
OUTPUT_DIR = "/mnt/users/jiayi/.cache/veomni/crush-smol-parquet"
SOURCE_NAME = "crush-smol"
ROWS_PER_FILE = 50  # all 47 rows fit in one file

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read videos.txt and prompt.txt (line-by-line correspondence)
with open(os.path.join(DATASET_DIR, "videos.txt")) as f:
    video_paths = [line.strip() for line in f if line.strip()]

with open(os.path.join(DATASET_DIR, "prompt.txt")) as f:
    prompts = [line.strip() for line in f if line.strip()]

assert len(video_paths) == len(prompts), (
    f"Mismatch: {len(video_paths)} videos vs {len(prompts)} prompts"
)

print(f"Found {len(video_paths)} video-prompt pairs")

# Build lists for the table
all_prompts = []
all_video_bytes = []
all_sources = []

for i, (vpath, prompt) in enumerate(zip(video_paths, prompts)):
    full_path = os.path.join(DATASET_DIR, vpath)
    if not os.path.exists(full_path):
        print(f"WARNING: missing {full_path}, skipping")
        continue
    with open(full_path, "rb") as vf:
        video_data = vf.read()
    all_prompts.append(prompt)
    all_video_bytes.append(video_data)
    all_sources.append(SOURCE_NAME)
    if (i + 1) % 10 == 0:
        print(f"  Processed {i + 1}/{len(video_paths)}")

print(f"Writing {len(all_prompts)} rows to parquet...")

# Write as a single parquet file
table = pa.table({
    "prompt": pa.array(all_prompts, type=pa.string()),
    "video_bytes": pa.array(all_video_bytes, type=pa.binary()),
    "source": pa.array(all_sources, type=pa.string()),
})

output_path = os.path.join(OUTPUT_DIR, "train-00000-of-00001.parquet")
pq.write_table(table, output_path)

print(f"Done! Written to {output_path}")
print(f"  Rows: {len(table)}")
print(f"  Schema: {table.schema}")
file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"  File size: {file_size_mb:.1f} MB")

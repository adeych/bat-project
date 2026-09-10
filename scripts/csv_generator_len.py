import os
import pandas as pd
import librosa


def get_duration_seconds(filepath):
    """Return audio duration in seconds, or None if the file can't be read."""
    try:
        return librosa.get_duration(path=filepath)
    except Exception as e:
        print(f"  Warning: could not read duration for {filepath} ({e})")
        return None


def consolidate_bat_dataset(root_dir, output_csv="final_metadata.csv"):
    # Class mapping: [TypeA, TypeB, TypeC, TypeD, Echo]
    # We removed the 6th 'Noise' column entirely.
    class_map = {
        'type A': [1, 0, 0, 0, 0],
        'type B': [0, 1, 0, 0, 0],
        'type C': [0, 0, 1, 0, 0],
        'type D': [0, 0, 0, 1, 0],
        'Echolocation': [0, 0, 0, 0, 1]
    }

    class_cols = ['type_a', 'type_b', 'type_c', 'type_d', 'echo']
    all_records = []

    # 1. Crawl all folders
    for folder_name, vector in class_map.items():
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.exists(folder_path):
            print(f"Skipping missing folder: {folder_name}")
            continue

        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.wav', '.mp3', '.flac')):
                full_path = os.path.join(folder_path, fname)

                # Create a temporary record
                record = {
                    'filename': fname,
                    'folder_ref': folder_name,
                    'duration': get_duration_seconds(full_path),
                }
                for i, col in enumerate(class_cols):
                    record[col] = vector[i]
                all_records.append(record)

    if not all_records:
        print("No audio files found! Check your folder paths.")
        return

    # 2. MERGE DUPLICATES (The A + C merge)
    # We group by filename and take the 'max' of the 1s and 0s.
    df = pd.DataFrame(all_records)
    df_merged = df.groupby('filename').agg({
        'type_a': 'max',
        'type_b': 'max',
        'type_c': 'max',
        'type_d': 'max',
        'echo': 'max',
        'duration': 'first',   # same file -> same duration regardless of which folder copy we saw
        'folder_ref': 'first'  # Keep one folder as the path reference
    }).reset_index()

    # 3. Create the path for the DataLoader
    df_merged['relative_path'] = df_merged.apply(
        lambda x: os.path.join(x['folder_ref'], x['filename']), axis=1
    )

    # 4. Global Sort by filename, and put columns in the desired final order
    df_merged = df_merged.sort_values(by='filename').drop(columns=['folder_ref'])
    df_merged = df_merged[
        ['filename', 'type_a', 'type_b', 'type_c', 'type_d', 'echo', 'duration', 'relative_path']
    ]

    # 5. Save
    df_merged.to_csv(output_csv, index=False)

    print("-" * 30)
    print(f"Unique files found: {len(df_merged)}")
    print(f"Total multi-label files (e.g., A+C): {sum(df_merged[class_cols].sum(axis=1) > 1)}")
    n_missing_duration = df_merged['duration'].isna().sum()
    if n_missing_duration:
        print(f"Warning: {n_missing_duration} file(s) had unreadable duration (see warnings above)")
    print(f"CSV saved as: {output_csv}")


# Run it
consolidate_bat_dataset(
    r"/idiap/temp/adeych/data/xenocanto-dataset"
)
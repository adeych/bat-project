"""
Quick sanity check for a Slurm job:
1. Confirm we can see/read a file from the node's temp/scratch directory
2. Confirm the conda environment can import + load the encoder

Run this AFTER copying at least one data file into your temp dir
(see the cp step in your .slurm script), e.g.:

    cp /home/$USER/data/bat_clips/some_clip.wav $WORKDIR/data/

Usage (inside your Slurm script, after activating the conda env):
    python test_env.py --data_dir $WORKDIR/data
"""

import argparse
import os
import sys


def check_temp_folder(data_dir):
    print(f"\n--- Checking temp folder: {data_dir} ---")
    if not os.path.isdir(data_dir):
        print(f"[FAIL] Directory does not exist: {data_dir}")
        sys.exit(1)

    files = os.listdir(data_dir)
    if not files:
        print(f"[FAIL] Directory exists but is empty: {data_dir}")
        sys.exit(1)

    print(f"[OK] Found {len(files)} item(s) in {data_dir}")
    first_file = os.path.join(data_dir, files[0])
    print(f"[OK] Example file: {first_file}")

    # Try to actually open it, to confirm it's readable (not just listed)
    try:
        with open(first_file, "rb") as f:
            chunk = f.read(1024)
        print(f"[OK] Successfully read {len(chunk)} bytes from {files[0]}")
    except Exception as e:
        print(f"[FAIL] Could not open/read {first_file}: {e}")
        sys.exit(1)

    return first_file


def check_encoder_import():
    print("\n--- Checking encoder import (avex) ---")
    try:
        import avex  # noqa: F401
        print("[OK] `avex` imported successfully")
        print(f"     Module location: {avex.__file__}")
        return avex
    except ImportError as e:
        print(f"[FAIL] Could not import `avex`: {e}")
        print("       Check: is it installed in this conda env? "
              "Try `pip show avex` or `conda list | grep avex`.")
        sys.exit(1)

def check_torch_import():
    print("\n--- Checking PyTorch import ---")
    try:
        import torch  # noqa: F401
        print("[OK] `torch` imported successfully")
        print(f"     Module location: {torch.__file__}")
        return torch
    except ImportError as e:
        print(f"[FAIL] Could not import `torch`: {e}")
        print("       Check: is it installed in this conda env? "
              "Try `pip show torch` or `conda list | grep torch`.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Path to the temp/scratch directory containing your data")
    args = parser.parse_args()

    print("=" * 50)
    print("ENVIRONMENT + DATA ACCESS TEST")
    print("=" * 50)

    example_file = check_temp_folder(args.data_dir)
    torch = check_torch_import()
    avex_module = check_encoder_import()

    print("\n" + "=" * 50)
    print("ALL CHECKS PASSED")
    print("=" * 50)
#!/usr/bin/env python3
"""
Quick setup script for .tar.xz dataset files

Usage:
    python setup_tar_xz_dataset.py <your_dataset.tar.xz>
    
Example:
    python setup_tar_xz_dataset.py data/azure_dataset.tar.xz
    python setup_tar_xz_dataset.py ~/Downloads/cloud_traces.tar.xz
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from extract_compressed import CompressedDatasetHandler, quick_extract_and_load


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    tar_xz_file = sys.argv[1]
    nrows = int(sys.argv[2]) if len(sys.argv) > 2 else None

    # Check if file exists
    if not os.path.exists(tar_xz_file):
        print(f"✗ File not found: {tar_xz_file}")
        print("\nUsage: python setup_tar_xz_dataset.py <path_to_tar_xz>")
        sys.exit(1)

    print(f"File: {tar_xz_file}")
    print(f"Size: {os.path.getsize(tar_xz_file) / (1024*1024):.2f} MB")
    print()

    # Extract and load
    try:
        df = quick_extract_and_load(tar_xz_file, nrows=nrows)

        print("\n" + "=" * 70)
        print("SETUP COMPLETE!")
        print("=" * 70)
        print("\nNow you can train with your dataset:")
        print()
        print("  # Option 1: Auto-detect extracted CSV")
        print("  python train_with_azure_data.py")
        print()
        print("  # Option 2: Specify timesteps")
        print("  python train_with_azure_data.py --timesteps 50000")
        print()
        print("  # Option 3: Quick test")
        print("  python train_with_azure_data.py --timesteps 10000 --episodes 2")
        print()

    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

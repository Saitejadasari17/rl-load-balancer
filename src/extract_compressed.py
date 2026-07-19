"""
Handle .tar.xz compressed dataset files
Supports extraction and processing of compressed dataset archives
"""

import tarfile
import os
import pandas as pd
import numpy as np
from pathlib import Path


class CompressedDatasetHandler:
    """Extract and process .tar.xz and other compressed dataset files."""

    def __init__(self, tar_xz_path):
        """
        Initialize with path to .tar.xz file.
        
        Args:
            tar_xz_path: Path to .tar.xz file (e.g., "data/azure_dataset.tar.xz")
        """
        self.tar_xz_path = tar_xz_path
        self.extract_dir = None
        self.csv_files = []

    def extract(self, output_dir="data/extracted"):
        """
        Extract .tar.xz file.
        
        Args:
            output_dir: Directory to extract files to
            
        Returns:
            Path to extraction directory
        """
        if not os.path.exists(self.tar_xz_path):
            raise FileNotFoundError(f"File not found: {self.tar_xz_path}")

        print(f"Extracting {self.tar_xz_path}...")
        print(f"This may take a few minutes...")

        os.makedirs(output_dir, exist_ok=True)

        try:
            with tarfile.open(self.tar_xz_path, "r:xz") as tar:
                tar.extractall(path=output_dir)
                print(f"✓ Extraction complete: {output_dir}")

            self.extract_dir = output_dir
            self._find_csv_files(output_dir)
            return output_dir

        except Exception as e:
            print(f"✗ Extraction failed: {e}")
            raise

    def _find_csv_files(self, directory):
        """Find all CSV files in extracted directory."""
        csv_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.csv'):
                    full_path = os.path.join(root, file)
                    csv_files.append(full_path)
                    print(f"   Found: {full_path}")

        self.csv_files = csv_files
        return csv_files

    def load_first_csv(self, nrows=None):
        """
        Load the first CSV file found.
        
        Args:
            nrows: Limit rows to load (useful for large files)
            
        Returns:
            pandas DataFrame
        """
        if not self.csv_files:
            if not self.extract_dir:
                raise ValueError("Must extract first")
            self._find_csv_files(self.extract_dir)

        if not self.csv_files:
            raise FileNotFoundError("No CSV files found after extraction")

        first_csv = self.csv_files[0]
        print(f"\nLoading: {first_csv}")

        if nrows:
            print(f"Loading first {nrows} rows...")
            df = pd.read_csv(first_csv, nrows=nrows)
        else:
            df = pd.read_csv(first_csv)

        print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    def load_all_csv_files(self, nrows=None):
        """
        Load and combine all CSV files.
        
        Args:
            nrows: Limit rows per file
            
        Returns:
            Combined pandas DataFrame
        """
        if not self.csv_files:
            if not self.extract_dir:
                raise ValueError("Must extract first")
            self._find_csv_files(self.extract_dir)

        dataframes = []
        for csv_file in self.csv_files:
            print(f"Loading: {csv_file}")
            df = pd.read_csv(csv_file, nrows=nrows)
            dataframes.append(df)
            print(f"  ✓ {len(df)} rows")

        combined = pd.concat(dataframes, ignore_index=True)
        print(f"\n✓ Combined: {len(combined)} total rows")
        return combined

    def get_statistics(self, df):
        """Print statistics about the dataset."""
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nData Types:\n{df.dtypes}")
        print(f"\nNumeric Summary:\n{df.describe()}")
        print("=" * 60 + "\n")


def quick_extract_and_load(tar_xz_path, nrows=None):
    """
    Quick utility: Extract and load .tar.xz file in one command.
    
    Args:
        tar_xz_path: Path to .tar.xz file
        nrows: Limit rows to load
        
    Returns:
        Loaded pandas DataFrame
    """
    handler = CompressedDatasetHandler(tar_xz_path)
    handler.extract()
    df = handler.load_first_csv(nrows=nrows)
    handler.get_statistics(df)
    return df


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extract_compressed.py <tar_xz_file> [nrows]")
        print("\nExample:")
        print("  python extract_compressed.py data/dataset.tar.xz")
        print("  python extract_compressed.py data/dataset.tar.xz 10000")
        sys.exit(1)

    tar_file = sys.argv[1]
    nrows = int(sys.argv[2]) if len(sys.argv) > 2 else None

    df = quick_extract_and_load(tar_file, nrows=nrows)

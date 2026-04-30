"""
Azure Serverless Functions Dataset Integration
Loads and processes the Microsoft Azure Functions Invocation dataset (July 2019)
for realistic load balancing simulation.

Supports: .csv files and .tar.xz compressed archives

Dataset: https://github.com/Azure/AzurePublicDataset
Paper: "Serverless in the Wild: Characterizing and Optimizing the Serverless Paradigm"
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path

try:
    from .extract_compressed import CompressedDatasetHandler
except ImportError:
    from extract_compressed import CompressedDatasetHandler

class AzureDatasetLoader:
    """Load and process Azure Functions dataset traces."""

    def __init__(self, data_path=None):
        """
        Initialize dataset loader.
        
        Args:
            data_path: Path to Azure dataset CSV file. If None, will look for it in data/ folder.
        """
        self.data_path = data_path
        self.traces = None
        self.invocations = None

    def download_dataset(self, output_dir="data"):
        """
        Download the Azure Functions dataset.
        Note: Manual download from GitHub may be required.
        """
        print("Azure dataset download instructions:")
        print("1. Download from: https://github.com/Azure/AzurePublicDataset")
        print("2. Look for 'azurefunctions_trace' files")
        print("3. Place CSV files in the 'data/' directory")
        print("4. Recommended: Use invocations_per_function_md5_90min.csv")

    def load_traces(self, csv_file=None):
        """
        Load Azure Functions invocation traces.
        
        Supports:
        - .csv files directly
        - .tar.xz compressed archives (auto-extracts)
        
        Expected columns (or compatible format):
        - timestamp: Unix timestamp
        - function_name / function_id: Function identifier
        - execution_time: Duration of function execution (ms)
        - memory_used: Memory consumption (MB)
        - trigger_type: HTTP, Timer, Queue, etc.
        """
        if csv_file is None:
            csv_file = self.data_path or self._find_dataset_file()

        if csv_file is None or not os.path.exists(csv_file):
            print(f"Dataset not found at {csv_file}")
            print("Using synthetic data instead. To use real data:")
            print("  1. Download dataset from https://github.com/Azure/AzurePublicDataset")
            print("  2. Place it in data/ folder (.csv or .tar.xz format)")
            return None

        print(f"Loading dataset from {csv_file}")

        # Handle .tar.xz compressed files
        if csv_file.endswith('.tar.xz'):
            print("Detected .tar.xz format - extracting...")
            handler = CompressedDatasetHandler(csv_file)
            handler.extract(output_dir="data/extracted")
            df = handler.load_first_csv()
        elif csv_file.endswith('.csv'):
            # Load CSV directly
            try:
                df = pd.read_csv(csv_file)
                print(f"Loaded {len(df)} invocation records")
            except Exception as e:
                print(f"Error loading dataset: {e}")
                return None
        else:
            print(f"Unsupported file format: {csv_file}")
            return None

        self.traces = df
        return df

    def _find_dataset_file(self):
        """Search for dataset files in common locations."""
        search_patterns = [
            "data/*.tar.xz",          # .tar.xz files first (larger archives)
            "data/invocations_per_function_md5_90min.csv",
            "data/azure_functions.csv",
            "data/*.csv",             # Any CSV
            "../data/*.tar.xz",
            "../data/invocations_per_function_md5_90min.csv",
        ]
        
        from glob import glob
        
        for pattern in search_patterns:
            files = glob(pattern)
            if files:
                return files[0]  # Return first match
        return None

    def get_traffic_pattern(self, duration_seconds=3600, n_servers=3):
        """
        Convert Azure dataset into server load pattern.
        
        Args:
            duration_seconds: Simulation duration (default 1 hour)
            n_servers: Number of servers to distribute load across
            
        Returns:
            List of request loads for each time step
        """
        if self.traces is None:
            traces = self.load_traces()
            if traces is None:
                return self._generate_synthetic_pattern(duration_seconds, n_servers)

        # Extract execution times (proxy for request size/load)
        if 'execution_time' in self.traces.columns:
            exec_times = self.traces['execution_time'].values
        elif 'duration' in self.traces.columns:
            exec_times = self.traces['duration'].values
        else:
            # Use any numeric column as proxy
            numeric_cols = self.traces.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                exec_times = self.traces[numeric_cols[0]].values
            else:
                return self._generate_synthetic_pattern(duration_seconds, n_servers)

        # Normalize execution times to [0.05, 0.3] range (request load)
        exec_times = np.array(exec_times, dtype=float)
        exec_times = np.clip(exec_times, 0, np.percentile(exec_times, 99))  # Remove outliers
        normalized = 0.05 + (exec_times / np.max(exec_times)) * 0.25

        # Create time series by resampling
        n_steps = min(len(normalized), duration_seconds)
        traffic_pattern = normalized[:n_steps]

        # If we need more steps, repeat pattern
        if len(traffic_pattern) < duration_seconds:
            repeats = duration_seconds // len(traffic_pattern) + 1
            traffic_pattern = np.tile(traffic_pattern, repeats)[:duration_seconds]

        return traffic_pattern.tolist()

    def _generate_synthetic_pattern(self, duration_seconds, n_servers):
        """Generate synthetic traffic pattern when dataset unavailable."""
        # Realistic pattern: normal traffic with periodic bursts
        pattern = []
        for t in range(duration_seconds):
            base_load = np.random.uniform(0.05, 0.15)

            # Periodic burst (like business hours or scheduled jobs)
            hour_of_day = (t // 3600) % 24
            if 9 <= hour_of_day <= 17:  # Business hours
                burst_factor = 1.5
            else:
                burst_factor = 1.0

            # Random spike (cloud events, viral content)
            if np.random.random() < 0.01:  # 1% chance per second
                burst_factor *= 2.0

            load = base_load * burst_factor
            pattern.append(np.clip(load, 0.05, 0.3))

        return pattern

    def get_statistics(self):
        """Print dataset statistics."""
        if self.traces is None:
            self.load_traces()

        if self.traces is None:
            print("No dataset loaded")
            return

        print("\n=== Azure Dataset Statistics ===")
        print(f"Total invocations: {len(self.traces)}")

        if 'execution_time' in self.traces.columns:
            exec_times = self.traces['execution_time']
            print(f"Execution time (ms): mean={exec_times.mean():.2f}, "
                  f"median={exec_times.median():.2f}, "
                  f"p99={exec_times.quantile(0.99):.2f}")

        if 'memory_used' in self.traces.columns:
            memory = self.traces['memory_used']
            print(f"Memory used (MB): mean={memory.mean():.2f}, "
                  f"median={memory.median():.2f}, "
                  f"p99={memory.quantile(0.99):.2f}")

        if 'trigger_type' in self.traces.columns:
            print(f"Trigger types: {self.traces['trigger_type'].value_counts().to_dict()}")

        print()


class TraceReplayEnvironment:
    """Environment that replays real Azure traces instead of synthetic data."""

    def __init__(self, traces, n_servers=3):
        """
        Initialize trace replay.
        
        Args:
            traces: List of request loads from real dataset
            n_servers: Number of servers
        """
        self.traces = traces
        self.n_servers = n_servers
        self.current_step = 0

    def get_request_load(self):
        """Get next request load from traces."""
        if self.current_step >= len(self.traces):
            self.current_step = 0  # Loop back to start

        load = self.traces[self.current_step]
        self.current_step += 1
        return load

    def reset(self):
        """Reset trace playback."""
        self.current_step = 0


def compare_with_azure_data():
    """Utility function to compare algorithms on real Azure data."""
    loader = AzureDatasetLoader()
    traffic = loader.get_traffic_pattern(duration_seconds=1000, n_servers=3)
    loader.get_statistics()
    return traffic


if __name__ == "__main__":
    # Example usage
    loader = AzureDatasetLoader()
    loader.get_statistics()
    traffic = loader.get_traffic_pattern()
    print(f"Generated traffic pattern with {len(traffic)} steps")
    print(f"Sample loads: {traffic[:10]}")

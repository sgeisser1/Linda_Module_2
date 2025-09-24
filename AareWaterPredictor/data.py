import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

def get_all_station_names(data_dir='data/hydrology'):
    """Quickly scans all CSVs to return a list of unique station names."""
    print(f"[*] Pre-scanning for all available station names in '{data_dir}'...")
    csv_files = glob.glob(f"{data_dir}/aare_*.csv")
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found at '{data_dir}/aare_*.csv'.")
    
    all_names = set()
    for f in csv_files:
        df = pd.read_csv(f, usecols=['station_name'])
        all_names.update(df['station_name'].unique())
        
    print(f"[*] Found {len(all_names)} unique stations in total.")
    return sorted(list(all_names))

def load_data(data_dir='data/hydrology', stations_to_load=None, debug_mode=False):
    """Loads all yearly CSVs and filters them for specific stations and date ranges."""
    print(f"[*] Loading data for {len(stations_to_load)} specified stations...")
    csv_files = glob.glob(f"{data_dir}/aare_*.csv")
        
    df_list = [pd.read_csv(f) for f in csv_files]
    full_df = pd.concat(df_list, ignore_index=True)
    
    full_df = full_df[full_df['station_name'].isin(stations_to_load)]
    
    full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])

    if debug_mode:
        print("[!] Running in DEBUG mode: Filtering data to the first 2 years.")
        min_date = full_df['timestamp'].min()
        five_years_later = min_date + pd.DateOffset(years=1)
        full_df = full_df[full_df['timestamp'] < five_years_later]

    full_df['timestamp'] = full_df['timestamp'].astype(np.int64) // 10**9
    
    print(f"[*] Data loading complete.")
    return full_df.groupby('station_name')

def get_station_splits(grouped_data, test_stations, val_split=0.2, seed=42):
    print("[*] Splitting stations into training, validation, and test sets...")
    np.random.seed(seed)
    station_names = np.array([name for name, _ in grouped_data])
    test_mask = np.isin(station_names, test_stations)
    test_station_names = station_names[test_mask]
    train_val_station_names = station_names[~test_mask]
    np.random.shuffle(train_val_station_names)
    val_size = int(len(train_val_station_names) * val_split)
    val_station_names = train_val_station_names[:val_size]
    train_station_names = train_val_station_names[val_size:]
    print(f"    - Training stations: {len(train_station_names)}")
    print(f"    - Validation stations: {len(val_station_names)}")
    print(f"    - Test stations: {len(test_station_names)}")
    return train_station_names, val_station_names, test_station_names

class RiverFlowDataset(Dataset):
    def __init__(self, data, stations, sequence_length=30, prediction_horizon=1):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.sequences = self._create_sequences(data, stations)

    def _create_sequences(self, data, stations):
        sequences = []
        # --- THIS IS THE FIX ---
        # Check the length, which works for both lists and numpy arrays
        if len(stations) == 0:
            return sequences
        # -----------------------
        for station in tqdm(stations, desc="Creating sequences", leave=False):
            station_data = data.get_group(station)['value'].values
            for i in range(len(station_data) - self.sequence_length - self.prediction_horizon + 1):
                seq = station_data[i:i + self.sequence_length]
                label = station_data[i + self.sequence_length + self.prediction_horizon - 1]
                sequences.append((torch.tensor(seq, dtype=torch.float32).unsqueeze(1), torch.tensor(label, dtype=torch.float32)))
        return sequences

    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx]

def get_dataloaders(grouped_data, train_stations, val_stations, test_stations, batch_size=64, sequence_length=30):
    print("[*] Creating PyTorch DataLoaders...")
    print("    - Building training dataset...")
    train_dataset = RiverFlowDataset(grouped_data, train_stations, sequence_length)
    print("    - Building validation dataset...")
    val_dataset = RiverFlowDataset(grouped_data, val_stations, sequence_length)
    print("    - Building test dataset...")
    test_dataset = RiverFlowDataset(grouped_data, test_stations, sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("[*] DataLoaders created successfully.")
    return train_loader, val_loader, test_loader
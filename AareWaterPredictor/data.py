import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

def extract_fft_features(data_window, n_top_freqs, cutoff_frequency):
    """
    Calculates FFT on a window of data, applies a low-pass filter, and
    returns the top N frequencies with the highest magnitudes from the
    filtered set.
    """
    if len(data_window) == 0:
        return np.zeros(n_top_freqs)
        
    fft_vals = np.fft.fft(data_window)
    fft_freq = np.fft.fftfreq(len(data_window))
    
    # --- Low-Pass Filter Logic ---
    # We only care about positive frequencies below the cutoff
    positive_mask = (fft_freq > 0) & (fft_freq <= cutoff_frequency)
    
    fft_vals = fft_vals[positive_mask]
    fft_freq = fft_freq[positive_mask]
    
    # If no frequencies are left after filtering, return zeros
    if len(fft_freq) == 0:
        return np.zeros(n_top_freqs)
    
    # Get magnitudes and sort to find top frequencies
    magnitudes = np.abs(fft_vals)
    
    # Ensure we don't try to get more freqs than available
    num_to_select = min(n_top_freqs, len(fft_freq))
    top_indices = np.argsort(magnitudes)[-num_to_select:]
    
    top_freqs = fft_freq[top_indices]
    
    # Pad with zeros if we found fewer than n_top_freqs
    if len(top_freqs) < n_top_freqs:
        padded_freqs = np.zeros(n_top_freqs)
        padded_freqs[:len(top_freqs)] = top_freqs
        return padded_freqs
        
    return top_freqs

def load_data(data_dir='data/hydrology', station_name='Hagneck'):
    """
    Loads, resamples, and splits data for a single specified station.
    """
    print(f"[*] Loading data for station: '{station_name}'...")
    csv_files = glob.glob(f"{data_dir}/aare_*.csv")
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found at '{data_dir}/aare_*.csv'.")
    
    full_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    
    station_df = full_df[full_df['station_name'] == station_name].copy()
    if station_df.empty:
        raise ValueError(f"No data found for station '{station_name}'.")

    station_df['timestamp'] = pd.to_datetime(station_df['timestamp'])
    station_df = station_df.set_index('timestamp').sort_index()

    print("[*] Resampling data to a 1-hour interval...")
    value_df = station_df[['value']]
    resampled_df = value_df.resample('1h').interpolate(method='linear')
    print("[*] Resampling complete.")
    
    print("[*] Splitting data into train, validation, and test sets...")
    train_df = resampled_df.loc['2010-01-01':'2018-12-31']
    val_df = resampled_df.loc['2019-01-01':'2021-12-31']
    test_df = resampled_df.loc['2022-01-01':'2024-12-31']
    
    print(f"[*] Train set size: {len(train_df)} samples")
    print(f"[*] Validation set size: {len(val_df)} samples")
    print(f"[*] Test set size: {len(test_df)} samples")
    
    print(f"[*] Data loading and splitting complete.")
    return train_df, val_df, test_df

class RiverFlowDataset(Dataset):
    def __init__(self, data_df, sequence_length=30, prediction_horizon=1, fft_window_size=100, n_top_freqs=10, fft_cutoff_freq=0.05):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.fft_window_size = fft_window_size
        self.n_top_freqs = n_top_freqs
        self.fft_cutoff_freq = fft_cutoff_freq
        self.sequences = self._create_sequences(data_df)
        if len(self.sequences) > 0:
            input_features = self.sequences[0][0].shape[1]
            print(f"[*] Dataset created with {len(self.sequences)} sequences. Input feature shape: (sequence_length, {input_features})")

    def _create_sequences(self, data_df):
        sequences = []
        station_data = data_df['value'].values
        timestamps = data_df.index
        
        start_offset = self.fft_window_size
        end_point = len(station_data) - self.sequence_length - self.prediction_horizon + 1
        
        for i in tqdm(range(start_offset, end_point), desc="Creating sequences with all features", leave=False):
            # 1. FFT Features (constant for the sequence)
            fft_window = station_data[i - self.fft_window_size : i]
            fft_features = extract_fft_features(fft_window, self.n_top_freqs, self.fft_cutoff_freq)
            fft_features_tensor = torch.tensor(fft_features, dtype=torch.float32).repeat(self.sequence_length, 1)
            
            # 2. Main Sequence Value
            seq_values = station_data[i : i + self.sequence_length]
            seq_tensor = torch.tensor(seq_values, dtype=torch.float32).unsqueeze(1)
            
            # 3. Cyclical Date Features (variable for each step in the sequence)
            seq_timestamps = timestamps[i : i + self.sequence_length]
            day_of_year = seq_timestamps.dayofyear.values
            month = seq_timestamps.month.values
            
            day_sin = np.sin(2 * np.pi * day_of_year / 365.25)
            day_cos = np.cos(2 * np.pi * day_of_year / 365.25)
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            
            date_features_tensor = torch.tensor(np.stack([day_sin, day_cos, month_sin, month_cos], axis=1), dtype=torch.float32)
            
            # 4. Label
            label = station_data[i + self.sequence_length + self.prediction_horizon - 1]
            
            # 5. Combine all features
            # Shape: (sequence_length, 1 + n_top_freqs + 4)
            combined_input = torch.cat((seq_tensor, fft_features_tensor, date_features_tensor), dim=1)
            
            sequences.append((combined_input, torch.tensor(label, dtype=torch.float32)))
            
        return sequences

    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx]

def get_dataloaders(train_df, val_df, test_df, config):
    """Creates datasets and dataloaders for train, validation, and test sets."""
    train_dataset = RiverFlowDataset(train_df, config['sequence_length'], fft_window_size=config['fft_window_size'], n_top_freqs=config['n_top_freqs'], fft_cutoff_freq=config['fft_cutoff_freq'])
    val_dataset = RiverFlowDataset(val_df, config['sequence_length'], fft_window_size=config['fft_window_size'], n_top_freqs=config['n_top_freqs'], fft_cutoff_freq=config['fft_cutoff_freq'])
    test_dataset = RiverFlowDataset(test_df, config['sequence_length'], fft_window_size=config['fft_window_size'], n_top_freqs=config['n_top_freqs'], fft_cutoff_freq=config['fft_cutoff_freq'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    return train_loader, val_loader, test_loader
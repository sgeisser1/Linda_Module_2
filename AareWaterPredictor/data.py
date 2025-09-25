import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

def extract_fft_features(data_window, n_top_freqs, cutoff_frequency):
    if len(data_window) == 0: return np.zeros(n_top_freqs)
    fft_vals = np.fft.fft(data_window)
    fft_freq = np.fft.fftfreq(len(data_window))
    positive_mask = (fft_freq > 0) & (fft_freq <= cutoff_frequency)
    fft_vals, fft_freq = fft_vals[positive_mask], fft_freq[positive_mask]
    if len(fft_freq) == 0: return np.zeros(n_top_freqs)
    magnitudes = np.abs(fft_vals)
    num_to_select = min(n_top_freqs, len(fft_freq))
    top_indices = np.argsort(magnitudes)[-num_to_select:]
    top_freqs = fft_freq[top_indices]
    if len(top_freqs) < n_top_freqs:
        padded_freqs = np.zeros(n_top_freqs)
        padded_freqs[:len(top_freqs)] = top_freqs
        return padded_freqs
    return top_freqs

def load_data(data_dir='data/hydrology', station_name='Hagneck'):
    print(f"[*] Loading data for station: '{station_name}'...")
    csv_files = glob.glob(f"{data_dir}/aare_*.csv")
    if not csv_files: raise FileNotFoundError(f"No CSV files found at '{data_dir}/aare_*.csv'.")
    full_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    station_df = full_df[full_df['station_name'] == station_name].copy()
    if station_df.empty: raise ValueError(f"No data found for station '{station_name}'.")
    station_df['timestamp'] = pd.to_datetime(station_df['timestamp'])
    station_df = station_df.set_index('timestamp').sort_index()
    print("[*] Resampling data to a 1-hour interval...")
    resampled_df = station_df[['value']].resample('1h').interpolate(method='linear')
    print("[*] Resampling complete.")
    train_df = resampled_df.loc['2010-01-01':'2018-12-31']
    val_df = resampled_df.loc['2019-01-01':'2021-12-31']
    test_df = resampled_df.loc['2022-01-01':'2024-12-31']
    print(f"[*] Data loading and splitting complete.")
    return train_df, val_df, test_df

class RiverFlowDataset(Dataset):
    def __init__(self, data_df, config):
        self.config = config
        self.sequence_length = config['sequence_length']
        self.prediction_horizon = 1
        self.sequences = self._create_sequences(data_df)
        if len(self.sequences) > 0:
            input_features = self.sequences[0][0].shape[1]
            print(f"[*] Dataset created with {len(self.sequences)} sequences. Input feature shape: (sequence_length, {input_features})")

    def _create_sequences(self, data_df):
        sequences = []
        station_data = data_df['value'].values
        timestamps = data_df.index
        
        use_fft = self.config.get('fft_window_size', 0) > 0
        use_date = self.config.get('use_date_features', False)
        
        start_offset = self.config.get('fft_window_size', 0) if use_fft else 0
        end_point = len(station_data) - self.sequence_length - self.prediction_horizon + 1
        
        desc = "Creating sequences"
        if use_fft and use_date: desc += " with FFT & Date"
        elif use_fft: desc += " with FFT"
        elif use_date: desc += " with Date"

        for i in tqdm(range(start_offset, end_point), desc=desc, leave=False):
            tensors_to_combine = []
            
            tensors_to_combine.append(torch.tensor(station_data[i : i + self.sequence_length], dtype=torch.float32).unsqueeze(1))
            
            if use_fft:
                fft_window = station_data[i - self.config['fft_window_size'] : i]
                fft_features = extract_fft_features(fft_window, self.config['n_top_freqs'], self.config['fft_cutoff_freq'])
                tensors_to_combine.append(torch.tensor(fft_features, dtype=torch.float32).repeat(self.sequence_length, 1))
            
            if use_date:
                seq_timestamps = timestamps[i : i + self.sequence_length]
                day_sin = np.sin(2 * np.pi * seq_timestamps.dayofyear.values / 365.25)
                day_cos = np.cos(2 * np.pi * seq_timestamps.dayofyear.values / 365.25)
                month_sin = np.sin(2 * np.pi * seq_timestamps.month.values / 12)
                month_cos = np.cos(2 * np.pi * seq_timestamps.month.values / 12)
                date_features = np.stack([day_sin, day_cos, month_sin, month_cos], axis=1)
                tensors_to_combine.append(torch.tensor(date_features, dtype=torch.float32))
            
            label = station_data[i + self.sequence_length + self.prediction_horizon - 1]
            combined_input = torch.cat(tensors_to_combine, dim=1)
            sequences.append((combined_input, torch.tensor(label, dtype=torch.float32)))
            
        return sequences

    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx]

def get_dataloaders(train_df, val_df, test_df, config):
    train_dataset = RiverFlowDataset(train_df, config)
    val_dataset = RiverFlowDataset(val_df, config)
    test_dataset = RiverFlowDataset(test_df, config)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    return train_loader, val_loader, test_loader
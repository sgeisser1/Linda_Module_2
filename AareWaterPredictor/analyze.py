import argparse
import json
import os
import torch
import numpy as np
import pandas as pd


# Explicitly set the Matplotlib backend *before* importing pyplot.
# This ensures that a GUI backend is used, allowing plots to be displayed.
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


import seaborn as sns
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import necessary components from your existing scripts
from data import RiverFlowDataset, load_data
from model import LSTMModel

def analyze_run(run_name, data_split='test', station_name='Hagneck'):
    """
    Loads a trained model and its config, runs inference on a data split for a
    specific station, and generates/displays analysis plots and metrics.
    """
    print(f"--- Starting Analysis for Run: {run_name} ---")
    print(f"--- Station: {station_name} | Data Split: {data_split} ---")
    
    # --- 1. Load Model and Configuration ---
    model_path = f"models/best_{run_name}.pth"
    config_path = f"models/best_{run_name}.json"

    if not os.path.exists(model_path) or not os.path.exists(config_path):
        raise FileNotFoundError(f"Model or config file not found for run '{run_name}'. "
                              f"Expected '{model_path}' and '{config_path}'.")

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("[*] Configuration loaded successfully.")

    # --- 2. Setup Device and Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    input_size = 1
    if config.get('fft_window_size', 0) > 0:
        input_size += config.get('n_top_freqs', 10)
    if config.get('use_date_features', False):
        input_size += 4

    model = LSTMModel(
        input_size=input_size,
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        hidden_layer_size=config['hidden_layer_size']
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("[*] Model loaded and set to evaluation mode.")

    # --- 3. Load and Prepare Data ---
    print(f"[*] Loading and preparing '{data_split}' data for station '{station_name}'...")
    try:
        train_df, val_df, test_df = load_data(station_name=station_name)
    except ValueError as e:
        print(f"\n[!] ERROR: {e}")
        return

    if data_split == 'test': eval_df = test_df
    elif data_split == 'validation': eval_df = val_df
    else: eval_df = train_df
        
    dataset = RiverFlowDataset(eval_df, config)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # --- 4. Run Inference ---
    print("[*] Running inference...")
    predictions, actuals = [], []
    with torch.no_grad():
        for seq, labels in tqdm(loader, desc="Generating Predictions"):
            seq, labels = seq.to(device), labels.to(device)
            y_pred = model(seq)
            predictions.append(y_pred.item())
            actuals.append(labels.item())
    
    predictions, actuals = np.array(predictions), np.array(actuals)

    # --- 5. Calculate Metrics ---
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    print("\n--- Analysis Metrics ---")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  R-squared (R²):           {r2:.4f}")

    # --- 6. Generate and Save Visualizations ---
    print("\n[*] Generating and saving plots...")
    output_dir = "analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    offset = (dataset.config.get('fft_window_size', 0) if dataset.config.get('fft_window_size', 0) > 0 else 0) + dataset.sequence_length
    timestamps = eval_df.index[offset:]

    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot 1: Predictions vs. Actuals over Time
    fig1, ax1 = plt.subplots(figsize=(15, 6))
    ax1.plot(timestamps, actuals, label='Actual Values', color='dodgerblue', alpha=0.8)
    ax1.plot(timestamps, predictions, label='Predicted Values', color='orangered', linestyle='--')
    ax1.set_title(f'Predictions vs. Actuals on "{station_name}" ({run_name})', fontsize=16)
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Water Flow Value')
    ax1.legend()
    ax1.grid(True)
    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, f"{run_name}_{station_name}_timeseries.png"))

    # Plot 2: Scatter Plot of Predictions vs. Actuals
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.scatter(actuals, predictions, alpha=0.5, edgecolors='k', s=40)
    ax2.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2, label='Perfect Prediction')
    ax2.set_title(f'Prediction vs. Actual Scatter ({station_name})', fontsize=16)
    ax2.set_xlabel('Actual Values')
    ax2.set_ylabel('Predicted Values')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, f"{run_name}_{station_name}_scatter.png"))

    # Plot 3: Distribution of Errors (Residuals)
    errors = actuals - predictions
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.histplot(errors, kde=True, ax=ax3, bins=50)
    ax3.axvline(errors.mean(), color='red', linestyle='--', label=f'Mean Error: {errors.mean():.2f}')
    ax3.set_title(f'Error Distribution ({station_name})', fontsize=16)
    ax3.set_xlabel('Error (Actual - Predicted)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True)
    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, f"{run_name}_{station_name}_error_dist.png"))

    print(f"[*] Analysis complete. Plots saved to '{output_dir}/'")
    
    # --- 7. Display all created figures ---
    # print("[*] Displaying interactive plots. Close all plot windows to exit.")
    # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analysis script for Aare Water Predictor')
    parser.add_argument('run_name', type=str, help='The name of the W&B run to analyze (e.g., "dulcet-sweep-1").')
    parser.add_argument('--data_split', type=str, default='test', choices=['train', 'validation', 'test'],
                        help='The data split to run analysis on.')
    parser.add_argument('--station', type=str, default='Hagneck',
                        help='The station name to load data for. Defaults to "Hagneck".')
    
    args = parser.parse_args()
    analyze_run(args.run_name, args.data_split, args.station)
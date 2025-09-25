import argparse
import numpy as np
import torch
import wandb
import sys
import os
from tqdm.auto import tqdm
from data import load_data, get_dataloaders
from model import LSTMModel, train_model, set_seeds

def run_pipeline(config):
    """Main pipeline for training or evaluation, callable from other scripts."""
    project_name = "aare-predictor-debug" if config['debug'] else "aare-water-predictor"
    
    if wandb.run is None:
        wandb.init(project=project_name, config=config)
    
    config.update(wandb.config)
    
    print(f"[*] Logging run '{wandb.run.name}' to project '{project_name}'")
    print(f"[*] Run config: {config}")

    # --- Data Loading and Preparation ---
    data_path = 'data/hydrology'
    train_df, val_df, test_df = load_data(data_path, station_name='Hagneck')
    
    train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df, config)

    # --- Model Training ---
    print("\n--- Training Model ---")
    # FIX: Update input_size to include the 4 new cyclical date features
    input_size = 1 + config['n_top_freqs'] + 4
    
    model = LSTMModel(
        input_size=input_size,
        num_layers=config['num_layers'], 
        dropout=config['dropout'],
        hidden_layer_size=config.get('hidden_layer_size', 100)
    )
    wandb.watch(model, log='all', log_freq=100)
    
    train_model(model, train_loader, val_loader, config)
    
    # --- Final Model Evaluation ---
    print("\n--- Evaluating Model on Test Set ---")
    best_model_path = f"models/best_{wandb.run.name}.pth"
    try:
        model.load_state_dict(torch.load(best_model_path))
        print(f"[*] Successfully loaded best model from '{best_model_path}' for final evaluation.")
    except Exception as e:
        print(f"[!] WARNING: Could not load best model. Evaluating with the final epoch's model. Error: {e}")

    model.eval()
    test_loss = 0
    loss_function = torch.nn.MSELoss()
    with torch.no_grad():
        for seq, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            y_pred = model(seq)
            test_loss += loss_function(y_pred, labels.unsqueeze(1)).item()
            
    final_mse = test_loss / len(test_loader) if len(test_loader) > 0 else 0
    print(f"\n--- Evaluation Complete ---")
    print(f"[*] Final Test Loss (MSE): {final_mse:.4f}")
    wandb.log({"final_test_mse": final_mse})
    
    if wandb.run:
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aare River Water Flow Prediction')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode and log to a debug project.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs.')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--sequence_length', type=int, default=30, help='Sequence length.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience.')
    parser.add_argument('--hidden_layer_size', type=int, default=100, help='Size of LSTM hidden layer.')
    parser.add_argument('--fft_window_size', type=int, default=100, help='Number of past datapoints for FFT.')
    parser.add_argument('--n_top_freqs', type=int, default=10, help='Number of top frequencies to use as features.')
    # New argument for the low-pass filter cutoff
    parser.add_argument('--fft_cutoff_freq', type=float, default=0.05, help='Cutoff for the FFT low-pass filter.')
    
    args = parser.parse_args()
    config = vars(args)
    
    os.makedirs("models", exist_ok=True)
    set_seeds(config['seed'])
    run_pipeline(config)
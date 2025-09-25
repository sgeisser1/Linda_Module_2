import argparse
import torch
import wandb
import os
import json
from tqdm.auto import tqdm
from data import load_data, get_dataloaders
from model import LSTMModel, train_model, set_seeds

def run_pipeline(config):
    project_name = "aare-predictor-debug" if config.get('debug') else "aare-water-predictor"
    
    if wandb.run is None:
        wandb.init(project=project_name, config=config)
    
    # Combine initial config with sweep config from wandb
    full_config = {**config, **wandb.config}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    print(f"[*] Logging run '{wandb.run.name}' to project '{project_name}'")
    print(f"[*] Run config: {full_config}")

    data_path = 'data/hydrology'
    train_df, val_df, test_df = load_data(data_path, station_name='Hagneck')
    train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df, full_config)

    input_size = 1
    if full_config.get('fft_window_size', 0) > 0:
        input_size += full_config.get('n_top_freqs', 10)
    if full_config.get('use_date_features', False):
        input_size += 4
    
    print(f"[*] Model input size set to: {input_size}")

    model = LSTMModel(
        input_size=input_size,
        num_layers=full_config['num_layers'], 
        dropout=full_config['dropout'],
        hidden_layer_size=full_config.get('hidden_layer_size', 100)
    ).to(device)
    
    wandb.watch(model, log='all', log_freq=100)
    
    train_model(model, train_loader, val_loader, full_config, device)
    
    print("\n--- Evaluating Model on Test Set ---")
    best_model_path = f"models/best_{wandb.run.name}.pth"
    try:
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"[*] Successfully loaded best model for final evaluation.")
    except Exception as e:
        print(f"[!] WARNING: Could not load best model. Error: {e}")

    model.eval()
    test_loss = 0
    loss_function = torch.nn.MSELoss()
    with torch.no_grad():
        for seq, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            seq, labels = seq.to(device), labels.to(device)
            y_pred = model(seq)
            test_loss += loss_function(y_pred, labels.unsqueeze(1)).item()
            
    final_mse = test_loss / len(test_loader) if len(test_loader) > 0 else 0
    print(f"\n--- Evaluation Complete ---")
    print(f"[*] Final Test Loss (MSE): {final_mse:.4f}")
    wandb.log({"final_test_mse": final_mse})

    # --- SAVE FINAL CONFIG ---
    config_save_path = f"models/best_{wandb.run.name}.json"
    with open(config_save_path, 'w') as f:
        json.dump(full_config, f, indent=4)
    print(f"[*] Final configuration saved to '{config_save_path}'")

    if wandb.run:
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aare River Water Flow Prediction')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    # ... (rest of the arguments are the same)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--sequence_length', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--hidden_layer_size', type=int, default=100)
    parser.add_argument('--fft_window_size', type=int, default=168)
    parser.add_argument('--n_top_freqs', type=int, default=10)
    parser.add_argument('--fft_cutoff_freq', type=float, default=0.05)
    parser.add_argument('--use_date_features', action='store_true')
    
    args = parser.parse_args()
    config = vars(args)
    
    os.makedirs("models", exist_ok=True)
    set_seeds(config['seed'])
    run_pipeline(config)
import argparse
import numpy as np
import torch
import wandb
import sys
import os
from tqdm.auto import tqdm
from data import get_all_station_names, load_data, get_dataloaders
from model import LSTMModel, train_model, cross_validate, set_seeds

def run_training(config):
    """Handles the complete training and final evaluation workflow."""
    # Set W&B project based on debug mode
    project_name = "aare-predictor-debug" if config['debug'] else "aare-water-predictor"
    wandb.init(project=project_name, config=config)
    print(f"[*] Logging run '{wandb.run.name}' to project '{project_name}'")

    data_path = 'data/hydrology'

    # --- Step 1: Determine the global, permanent test set ---
    all_stations = np.array(get_all_station_names(data_path))
    rng = np.random.default_rng(config['seed'])
    test_stations = rng.choice(all_stations, size=3, replace=False)
    train_val_pool = np.setdiff1d(all_stations, test_stations)
    
    print("\n--- Global Station Set Assignments ---")
    print(f"[*] Permanent Test Set (Hold-out): {sorted(test_stations)}")
    print(f"[*] Available Train/Validation Pool: {len(train_val_pool)} stations")
    print("--------------------------------------")

    # --- Step 2: Define stations for this run ---
    if config['debug']:
        num_stations_for_debug_cv = min(config['n_splits'], len(train_val_pool))
        stations_for_cv = train_val_pool[:num_stations_for_debug_cv]
        stations_to_load = np.concatenate([stations_for_cv, test_stations])
    else:
        stations_for_cv = train_val_pool
        stations_to_load = all_stations

    # --- Step 3: Load data and perform CV ---
    grouped_data = load_data(data_path, stations_to_load=stations_to_load, debug_mode=config['debug'])
    
    if len(stations_for_cv) < 2:
        print("\n[!] ERROR: Not enough stations for CV. Exiting.")
        sys.exit(1)
    config['n_splits'] = min(config['n_splits'], len(stations_for_cv))

    mean_cv_loss, final_cv_samples = cross_validate(grouped_data, stations_for_cv, config)
    print(f"\n[*] Mean Cross-Validation Loss: {mean_cv_loss:.4f}")

    # --- Step 4: Train Final Model ---
    print("\n--- Training Final Model ---")
    train_loader, _, _ = get_dataloaders(grouped_data, stations_for_cv, np.array([]), np.array([]),
                                         batch_size=config['batch_size'],
                                         sequence_length=config['sequence_length'])
    
    final_model = LSTMModel(num_layers=config['num_layers'], dropout=config['dropout'])
    wandb.watch(final_model, log='all', log_freq=100)
    
    # Start training from where CV left off to keep the x-axis continuous
    train_model(final_model, train_loader, torch.utils.data.DataLoader([]), config, start_samples=final_cv_samples)
    
    # --- Step 5: Save the final model ---
    os.makedirs("models", exist_ok=True)
    save_path = f"models/{wandb.run.name}.pth"
    torch.save(final_model.state_dict(), save_path)
    print(f"[*] Final model saved to '{save_path}'")
    
    wandb.finish()
    print("\n--- Training Finished ---")
    print(f"To evaluate this model, run:\npython {sys.argv[0]} --evaluate_model_path {save_path}")


def evaluate_model(config):
    """Handles loading a saved model and evaluating it on the test set."""
    print(f"--- Evaluating Model: {config['evaluate_model_path']} ---")
    data_path = 'data/hydrology'

    # --- Step 1: Recreate the exact same test set ---
    all_stations = np.array(get_all_station_names(data_path))
    rng = np.random.default_rng(config['seed'])
    test_stations = rng.choice(all_stations, size=3, replace=False)
    print(f"[*] Evaluating on permanent test set: {sorted(test_stations)}")

    # --- Step 2: Load only the test data ---
    grouped_data = load_data(data_path, stations_to_load=test_stations)
    _, _, test_loader = get_dataloaders(grouped_data, [], [], test_stations,
                                        batch_size=config['batch_size'],
                                        sequence_length=config['sequence_length'])

    # --- Step 3: Load the model ---
    model = LSTMModel(num_layers=config['num_layers'], dropout=config['dropout'])
    try:
        model.load_state_dict(torch.load(config['evaluate_model_path']))
        print("[*] Model state loaded successfully.")
    except Exception as e:
        print(f"[!] ERROR: Could not load model. Ensure architecture parameters match. Error: {e}")
        sys.exit(1)
        
    # --- Step 4: Run evaluation ---
    model.eval()
    test_loss = 0
    loss_function = torch.nn.MSELoss()
    with torch.no_grad():
        for seq, labels in tqdm(test_loader, desc="Evaluating"):
            y_pred = model(seq)
            test_loss += loss_function(y_pred, labels.unsqueeze(1)).item()
    
    final_mse = test_loss / len(test_loader) if len(test_loader) > 0 else 0
    print(f"\n--- Evaluation Complete ---")
    print(f"[*] Final Test Loss (MSE): {final_mse:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aare River Water Flow Prediction')
    # Mode selection
    parser.add_argument('--evaluate_model_path', type=str, default=None, help='Path to a saved .pth model file to evaluate.')
    # General args
    parser.add_argument('--debug', action='store_true', help='Run in debug mode and log to a debug project.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    # Training args
    parser.add_argument('--n_splits', type=int, default=3, help='Number of folds for cross-validation.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs.')
    # Model architecture (needed for both training and evaluation)
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate.')
    # Data args
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--sequence_length', type=int, default=30, help='Sequence length.')
    # Optimizer args
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience.')
    
    args = parser.parse_args()
    config = vars(args)
    
    set_seeds(config['seed'])

    if config['evaluate_model_path']:
        evaluate_model(config)
    else:
        run_training(config)
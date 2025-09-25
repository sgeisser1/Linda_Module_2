import wandb
import argparse
import os
from train import run_pipeline, set_seeds

# 1. Define the sweep configuration using the new flag-based logic
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'epoch_val_loss',
        'goal': 'minimize'   
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 1e-2
        },
        'dropout': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.3
        },
        'num_layers': {
            'values': [1, 2, 3]
        },
        'batch_size': {
            'values': [32, 64]
        },
        'sequence_length': {
            'values': [24, 168, 720] # 1 day, 1 week, 1 month
        },
        'hidden_layer_size': {
            'values': [50, 100, 150]
        },
        # --- NEW FEATURE CONTROL LOGIC ---
        # fft_window_size = 0 means NO FFT features.
        'fft_window_size': {
            'values': [0, 168, 336, 720] # 0, 1 week, 2 weeks, 1 month
        },
        # use_date_features is a simple boolean flag.
        'use_date_features': {
            'values': [True, False]
        },
        # These parameters will be selected by the agent, but we will ignore them
        # in our code if fft_window_size is 0.
        'n_top_freqs': {
            'values': [5, 10, 20]
        },
        'fft_cutoff_freq': {
            'values': [0.01, 0.05]
        }
    }
}

def main():
    parser = argparse.ArgumentParser(description="Run or resume a W&B sweep.")
    parser.add_argument('--project', type=str, default="aare-predictor", help="W&B project name.")
    parser.add_argument('--count', type=int, default=50, help="Number of runs to execute in this agent.")
    parser.add_argument('--sweep_id', type=str, default=None, help="Optional: ID of an existing sweep to resume.")
    
    args = parser.parse_args()

    # If a sweep_id is provided, use it. Otherwise, create a new sweep.
    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"[*] Resuming sweep with ID '{sweep_id}' on project '{args.project}'.")
    else:
        print(f"[*] Starting a new sweep on project '{args.project}'.")
        sweep_id = wandb.sweep(sweep_config, project=args.project)
        print(f"[*] New sweep created with ID: {sweep_id}")
        print(f"[*] To resume this sweep later, run: python {__file__} --sweep_id {sweep_id}")

    def train_with_sweep():
        run = wandb.init()
        
        if wandb.config.fft_window_size == 0:
            wandb.config.update({
                'n_top_freqs': None,
                'fft_cutoff_freq': None
            }, allow_val_change=True)

        config = {
            'debug': True,
            'seed': 42,
            'epochs': 25,
            'patience': 5
        }
        
        set_seeds(config['seed'])
        run_pipeline(config)
        
        run.finish()

    os.makedirs("models", exist_ok=True)
    
    # The agent will now run on either the new or the existing sweep_id
    wandb.agent(sweep_id, function=train_with_sweep, count=args.count, project=args.project)

if __name__ == '__main__':
    main()
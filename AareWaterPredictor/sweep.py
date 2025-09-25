import wandb
import argparse
import os
from train import run_pipeline, set_seeds

# 1. Define the sweep configuration
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
            'values': [32, 64, 128]
        },
        'sequence_length': {
            'values': [24, 7*24, 30*24] # 1 day, 1 week, 1 month
        },
        # 'hidden_layer_size': {
        #     'values': [50, 100, 150]
        # },
        'n_top_freqs': {
            'values': [5, 10, 20]
        },
        'fft_window_size': {
            'values': [7*24, 14*24, 30*24, 365*24, 2*365*24] # 1 week, 2 weeks, 1 month, 1 year, 2 years
        },
        'fft_cutoff_freq': {
            'values': [0.01, 0.05] # 0.01 ~ weekly+, 0.05 ~ daily+
        }
    }
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default="aare-predictor-debug", help="W&B project name for the sweep.")
    parser.add_argument('--count', type=int, default=30, help="Number of runs in the sweep.")
    args = parser.parse_args()

    print(f"[*] Starting sweep on project '{args.project}' for {args.count} runs.")
    
    sweep_id = wandb.sweep(sweep_config, project=args.project)

    def train_with_sweep():
        # Fixed parameters for all sweep runs
        config = {
            'debug': True,
            'seed': 42,
            'epochs': 20,
            'patience': 5,
            'hidden_layer_size': 100,
        }
        set_seeds(config['seed'])
        run_pipeline(config)

    os.makedirs("models", exist_ok=True)
    wandb.agent(sweep_id, function=train_with_sweep, count=args.count)

if __name__ == '__main__':
    main()
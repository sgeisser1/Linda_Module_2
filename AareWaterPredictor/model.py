import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm.auto import tqdm

from data import get_dataloaders

def set_seeds(seed=42):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

class LSTMModel(nn.Module):
    """LSTM model for time series prediction."""
    # FIX: Added 'input_size' to the constructor
    def __init__(self, input_size, hidden_layer_size=100, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        
        lstm_dropout = dropout if num_layers > 1 else 0
        
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, dropout=lstm_dropout, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

def train_model(model, train_loader, val_loader, config):
    """Trains the LSTM model with sample-based W&B logging."""
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    samples_seen = 0

    epoch_pbar = tqdm(range(config['epochs']), desc="Epochs")

    for epoch in epoch_pbar:
        model.train()
        epoch_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
        for seq, labels in train_pbar:
            samples_seen += len(seq)
            
            optimizer.zero_grad()
            y_pred = model(seq)
            loss = loss_function(y_pred, labels.unsqueeze(1))
            loss.backward()
            
            wandb.log({"batch_loss": loss.item()}, step=samples_seen)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        epoch_val_loss = 0
        if len(val_loader) > 0:
            with torch.no_grad():
                for seq, labels in val_loader:
                    y_pred = model(seq)
                    epoch_val_loss += loss_function(y_pred, labels.unsqueeze(1)).item()
        
        epoch_val_loss /= len(val_loader) if len(val_loader) > 0 else 1
        epoch_train_loss /= len(train_loader) if len(train_loader) > 0 else 1
        
        total_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        
        wandb.log({
            "epoch": epoch, 
            "epoch_train_loss": epoch_train_loss, 
            "epoch_val_loss": epoch_val_loss,
            "gradient_norm_L2": total_norm
        }, step=samples_seen)
        
        epoch_pbar.set_postfix(val_loss=f"{epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            if wandb.run:
                torch.save(model.state_dict(), f"models/best_{wandb.run.name}.pth")
        else:
            epochs_no_improve += 1

        if len(val_loader) > 0 and epochs_no_improve >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch+1}!")
            break
            
    return best_val_loss
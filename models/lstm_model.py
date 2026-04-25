import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import joblib
import os


class StackedLSTMRegressor(nn.Module):
    """
    Stacked LSTM for RUL Prediction.
    
    Based on: Yilma et al. (2026) - Sequential metaheuristic optimization
    of stacked-LSTM hyperparameters for RUL prediction.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        super(StackedLSTMRegressor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output
    
    def predict(self, X: np.ndarray, device: str = 'cpu') -> np.ndarray:
        """Predict RUL for input sequences."""
        self.eval()
        X_tensor = torch.FloatTensor(X).to(device)
        
        with torch.no_grad():
            predictions = self.forward(X_tensor)
        
        return predictions.cpu().numpy()
    
    def save(self, filepath: str):
        """Save model to file."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'output_size': 1
            }
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str, device: str = 'cpu') -> 'StackedLSTMRegressor':
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(
            input_size=checkpoint['config']['input_size'],
            hidden_size=checkpoint['config']['hidden_size'],
            num_layers=checkpoint['config']['num_layers'],
            dropout=checkpoint['config']['dropout']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model


def create_sequences(
    data: np.ndarray,
    labels: np.ndarray,
    sequence_length: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for LSTM training.
    
    Args:
        data: Feature array of shape (n_samples, n_features)
        labels: RUL labels of shape (n_samples,)
        sequence_length: Length of sliding window
    
    Returns:
        X: Sequences of shape (n_sequences, sequence_length, n_features)
        y: Labels of shape (n_sequences,)
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(labels[i + sequence_length])
    
    return np.array(X), np.array(y)


def build_model(
    input_size: int,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    device: str = 'cpu'
) -> StackedLSTMRegressor:
    """Build and return a StackedLSTM model."""
    model = StackedLSTMRegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    return model.to(device)


def train_model(
    model: StackedLSTMRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    early_stopping_patience: int = 10
) -> Dict:
    """
    Train the LSTM model.
    
    Returns:
        Dict with training history and best validation score
    """
    model.train()
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    
    if X_val is not None:
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    history = {'train_loss': []}
    if X_val is not None:
        history['val_loss'] = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        n_batches = 0
        
        for i in range(0, len(X_train_t), batch_size):
            batch_X = X_train_t[i:i + batch_size]
            batch_y = y_train_t[i:i + batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = train_loss / n_batches
        history['train_loss'].append(avg_train_loss)
        
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
            history['val_loss'].append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return {
        'history': history,
        'best_val_loss': best_val_loss if X_val is not None else avg_train_loss
    }


def evaluate_model(
    model: StackedLSTMRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = 'cpu'
) -> Dict:
    """Evaluate model on test set."""
    model.eval()
    
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)
    
    with torch.no_grad():
        predictions = model(X_test_t)
        mse = nn.MSELoss()(predictions, y_test_t.unsqueeze(1)).item()
        mae = torch.mean(torch.abs(predictions - y_test_t.unsqueeze(1))).item()
        rmse = np.sqrt(mse)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'predictions': predictions.cpu().numpy()
    }
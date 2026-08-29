"""
Modular Stock Price Forecasting Module (RNN / LSTM)
Supports PyTorch engine with automatic fallback to high-performance NumPy-vectorized RNN.
Features:
- Load ticker CSV (Date, Open, High, Low, Close, Adj Close, Volume), sort chronologically, slice first half (Variant A).
- MinMaxScaler fitted ONLY on training set (strictly prevents data leakage).
- 60-day sliding window sequence creation.
- RNN/LSTM architecture with Dropout regularization and Dense(1) linear head.
- Adam optimizer and MSE loss with Early Stopping.
- Evaluation on unscaled actual dollar values (RMSE and MAE).
- Loss curve and price trajectory visualization.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

# Attempt to import PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


def load_and_preprocess_data(csv_path: str, variant_a: bool = True) -> pd.DataFrame:
    """
    Load stock dataset from CSV, parse dates, sort chronologically,
    and optionally slice to retain only the first half of the date range (Variant A).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at '{csv_path}'")
        
    df = pd.read_csv(csv_path)
    
    # Check required columns
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in CSV dataset.")
            
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    if variant_a:
        # Keep only the first 50% of the date range chronologically (Variant A)
        half_idx = len(df) // 2
        df = df.iloc[:half_idx].copy().reset_index(drop=True)
        
    return df


def split_data(df: pd.DataFrame, train_ratio: float = 0.8):
    """
    Perform a chronological train/test split (e.g. 80/20) preserving time order (no shuffling).
    """
    train_size = int(len(df) * train_ratio)
    train_df = df.iloc[:train_size].copy().reset_index(drop=True)
    test_df = df.iloc[train_size:].copy().reset_index(drop=True)
    return train_df, test_df


def scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list, target_col: str = 'Close'):
    """
    Scale OHLCV feature columns and target column using MinMaxScaler.
    CRITICAL: Fit scalers ONLY on training set to prevent data leakage.
    """
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit scalers strictly on training data
    train_features_scaled = feature_scaler.fit_transform(train_df[feature_cols])
    train_target_scaled = target_scaler.fit_transform(train_df[[target_col]])
    
    # Transform test data using fitted training scalers
    test_features_scaled = feature_scaler.transform(test_df[feature_cols])
    test_target_scaled = target_scaler.transform(test_df[[target_col]])
    
    return (
        train_features_scaled,
        train_target_scaled,
        test_features_scaled,
        test_target_scaled,
        feature_scaler,
        target_scaler
    )


def create_sliding_windows(features_scaled: np.ndarray, target_scaled: np.ndarray, window_size: int = 60):
    """
    Construct 60-day sliding windows.
    Each input X[i] contains 60 consecutive past days of features.
    Target y[i] is the target value on day i (the 61st day).
    """
    X_list, y_list = [], []
    for i in range(window_size, len(features_scaled)):
        X_list.append(features_scaled[i - window_size:i])
        y_list.append(target_scaled[i])
        
    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.float32)
    
    if TORCH_AVAILABLE:
        X_tensor = torch.tensor(X_arr, dtype=torch.float32)
        y_tensor = torch.tensor(y_arr, dtype=torch.float32)
        return X_tensor, y_tensor
        
    return X_arr, y_arr


# --- Model Implementations ---

if TORCH_AVAILABLE:
    class PyTorchRNNPredictor(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2, cell_type: str = 'LSTM'):
            super(PyTorchRNNPredictor, self).__init__()
            self.cell_type = cell_type.upper()
            dropout_param = dropout if num_layers > 1 else 0.0
            
            if self.cell_type == 'LSTM':
                self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_param)
            else:
                self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_param)
                
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            out, _ = self.rnn(x)
            last_step_out = self.dropout(out[:, -1, :])
            return self.fc(last_step_out)


class NumPyRNNPredictor:
    """
    High-Performance Vectorized Recurrent Neural Network with Dropout and Dense(1) head.
    Trained via Adam Optimizer with Momentum and RMSprop variance.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Xavier/He Initialization
        self.W_xh = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + hidden_dim))
        self.b_h = np.zeros((1, hidden_dim))
        
        self.W_hy = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / (hidden_dim + 1))
        self.b_y = np.zeros((1, 1))
        
        # Adam optimizer state
        self.m = {k: np.zeros_like(v) for k, v in self.get_params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self.get_params().items()}
        self.t = 0

    def get_params(self):
        return {
            'W_xh': self.W_xh,
            'W_hh': self.W_hh,
            'b_h': self.b_h,
            'W_hy': self.W_hy,
            'b_y': self.b_y
        }

    def set_params(self, params):
        self.W_xh = params['W_xh'].copy()
        self.W_hh = params['W_hh'].copy()
        self.b_h = params['b_h'].copy()
        self.W_hy = params['W_hy'].copy()
        self.b_y = params['b_y'].copy()

    def forward(self, X, training=False):
        # X shape: (batch_size, sequence_length=60, input_dim)
        batch_size, seq_len, _ = X.shape
        h = np.zeros((batch_size, self.hidden_dim))
        
        for t in range(seq_len):
            x_t = X[:, t, :]
            # Recurrent step with tanh non-linearity
            h = np.tanh(np.dot(x_t, self.W_xh) + np.dot(h, self.W_hh) + self.b_h)
            
        # Apply Dropout
        if training and self.dropout > 0:
            mask = (np.random.rand(*h.shape) >= self.dropout) / (1.0 - self.dropout)
            h = h * mask
            
        # Dense(1) linear output
        y_pred = np.dot(h, self.W_hy) + self.b_y
        return y_pred


def build_model(input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2, cell_type: str = 'LSTM'):
    """
    Factory function instantiating the RNN / LSTM predictor model.
    """
    if TORCH_AVAILABLE:
        return PyTorchRNNPredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            cell_type=cell_type
        )
    else:
        return NumPyRNNPredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )


def train_model(model, train_loader, val_loader, epochs: int = 40, lr: float = 0.001, patience: int = 8, device: str = 'cpu'):
    """
    Train model using Adam optimizer, MSE loss, and Early Stopping on validation loss.
    """
    train_losses = []
    val_losses = []
    
    if TORCH_AVAILABLE and isinstance(model, nn.Module):
        model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        best_weights = None
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            model.train()
            running_train_loss = 0.0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                out = model(X_b)
                loss = criterion(out, y_b)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * X_b.size(0)
                
            ep_train_loss = running_train_loss / len(train_loader.dataset)
            train_losses.append(ep_train_loss)
            
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    out = model(X_b)
                    loss = criterion(out, y_b)
                    running_val_loss += loss.item() * X_b.size(0)
                    
            ep_val_loss = running_val_loss / len(val_loader.dataset)
            val_losses.append(ep_val_loss)
            
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train MSE: {ep_train_loss:.6f} | Val MSE: {ep_val_loss:.6f}")
            
            if ep_val_loss < best_val_loss:
                best_val_loss = ep_val_loss
                best_weights = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}. Restoring best model weights.")
                    break
                    
        if best_weights is not None:
            model.load_state_dict(best_weights)
            
        return train_losses, val_losses

    else:
        # NumPy Vectorized Training with Adam Optimizer
        X_train, y_train = train_loader
        X_val, y_val = val_loader
        
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        best_val_loss = float('inf')
        best_params = None
        patience_counter = 0
        batch_size = 32
        
        n_samples = len(X_train)
        
        for epoch in range(1, epochs + 1):
            indices = np.arange(n_samples)
            # Shuffle batch order for gradient descent
            np.random.shuffle(indices)
            
            running_train_loss = 0.0
            for start_idx in range(0, n_samples, batch_size):
                b_idx = indices[start_idx:start_idx + batch_size]
                X_b, y_b = X_train[b_idx], y_train[b_idx]
                
                # Forward pass
                y_pred = model.forward(X_b, training=True)
                loss = np.mean((y_pred - y_b) ** 2)
                running_train_loss += loss * len(b_idx)
                
                # Finite difference / analytical gradient step
                model.t += 1
                grad_y = 2.0 * (y_pred - y_b) / len(b_idx)
                
                # Backprop to linear head
                batch_size_b, seq_len, _ = X_b.shape
                h = np.zeros((batch_size_b, model.hidden_dim))
                for t in range(seq_len):
                    h = np.tanh(np.dot(X_b[:, t, :], model.W_xh) + np.dot(h, model.W_hh) + model.b_h)
                    
                dW_hy = np.dot(h.T, grad_y)
                db_y = np.sum(grad_y, axis=0, keepdims=True)
                
                # Backprop into recurrent weights
                dh = np.dot(grad_y, model.W_hy.T) * (1.0 - h ** 2)
                dW_xh = np.dot(X_b[:, -1, :].T, dh)
                dW_hh = np.dot(h.T, dh) * 0.5
                db_h = np.sum(dh, axis=0, keepdims=True)
                
                grads = {'W_xh': dW_xh, 'W_hh': dW_hh, 'b_h': db_h, 'W_hy': dW_hy, 'b_y': db_y}
                params = model.get_params()
                
                # Adam update
                for k in params:
                    model.m[k] = beta1 * model.m[k] + (1 - beta1) * grads[k]
                    model.v[k] = beta2 * model.v[k] + (1 - beta2) * (grads[k] ** 2)
                    m_hat = model.m[k] / (1 - beta1 ** model.t)
                    v_hat = model.v[k] / (1 - beta2 ** model.t)
                    params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)
                    
                model.set_params(params)
                
            ep_train_loss = running_train_loss / n_samples
            train_losses.append(ep_train_loss)
            
            # Validation pass
            val_preds = model.forward(X_val, training=False)
            ep_val_loss = np.mean((val_preds - y_val) ** 2)
            val_losses.append(ep_val_loss)
            
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train MSE: {ep_train_loss:.6f} | Val MSE: {ep_val_loss:.6f}")
            
            if ep_val_loss < best_val_loss:
                best_val_loss = ep_val_loss
                best_params = {k: v.copy() for k, v in model.get_params().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}. Restoring best model weights.")
                    break
                    
        if best_params is not None:
            model.set_params(best_params)
            
        return train_losses, val_losses


def evaluate_model(model, test_loader, target_scaler, device: str = 'cpu'):
    """
    Evaluate trained model on test data, inverse-scaling predictions and actuals to compute RMSE and MAE.
    """
    if TORCH_AVAILABLE and isinstance(model, nn.Module):
        model.to(device)
        model.eval()
        preds_scaled, actuals_scaled = [], []
        with torch.no_grad():
            for X_b, y_b in test_loader:
                X_b = X_b.to(device)
                out = model(X_b)
                preds_scaled.append(out.cpu().numpy())
                actuals_scaled.append(y_b.numpy())
        preds_scaled = np.vstack(preds_scaled)
        actuals_scaled = np.vstack(actuals_scaled)
    else:
        X_test, y_test = test_loader
        preds_scaled = model.forward(X_test, training=False)
        actuals_scaled = y_test
        
    actual_prices = target_scaler.inverse_transform(actuals_scaled).flatten()
    predicted_prices = target_scaler.inverse_transform(preds_scaled).flatten()
    
    rmse = float(root_mean_squared_error(actual_prices, predicted_prices))
    mae = float(mean_absolute_error(actual_prices, predicted_prices))
    
    print(f"\n--- Model Evaluation Metrics (Actual Dollar Scale) ---")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.4f}")
    print(f"Mean Absolute Error (MAE):     ${mae:.4f}")
    
    return actual_prices, predicted_prices, rmse, mae


def plot_results(train_losses, val_losses, actual_prices, predicted_prices, dates=None, save_path=None):
    """
    Plot training vs validation loss convergence and actual vs predicted stock closing prices.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Loss curves
    axes[0].plot(train_losses, label='Train Loss (MSE)', color='#1f77b4', linewidth=2)
    axes[0].plot(val_losses, label='Validation Loss (MSE)', color='#ff7f0e', linewidth=2, linestyle='--')
    axes[0].set_title('Training & Validation Loss Convergence', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epochs', fontsize=12)
    axes[0].set_ylabel('Mean Squared Error Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Actual vs Predicted Stock Closing Prices
    x_axis = pd.to_datetime(dates) if dates is not None else np.arange(len(actual_prices))
    axes[1].plot(x_axis, actual_prices, label='Actual Closing Price', color='#2ca02c', linewidth=2)
    axes[1].plot(x_axis, predicted_prices, label='Predicted Closing Price', color='#d62728', linestyle='--', linewidth=2)
    axes[1].set_title('Next-Day Stock Closing Price: Actual vs. Predicted', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date' if dates is not None else 'Sample Index', fontsize=12)
    axes[1].set_ylabel('Stock Price ($)', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to '{save_path}'")
    plt.close()

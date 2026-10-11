"""
End-to-End Stock Forecasting Pipeline Runner
Executes data loading (Variant A), windowing, scaling, model training, evaluation, and plot rendering.
"""

import os
from scripts.generate_sample_data import generate_stock_data
from src.rnn_stock_model import (
    load_and_preprocess_data,
    split_data,
    scale_features,
    create_sliding_windows,
    build_model,
    train_model,
    evaluate_model,
    plot_results,
    TORCH_AVAILABLE
)

if TORCH_AVAILABLE:
    import torch
    from torch.utils.data import TensorDataset, DataLoader


def main():
    print("=" * 65)
    print("   STOCK PRICE FORECASTING USING RNN / LSTM (VARIANT A)")
    print(f"   Engine: {'PyTorch (' + torch.__version__ + ')' if TORCH_AVAILABLE else 'Vectorized RNN with Adam'}")
    print("=" * 65)
    
    # 1. Dataset setup
    csv_path = 'data/sample_stock_data.csv'
    if not os.path.exists(csv_path):
        print("CSV dataset not found. Generating sample stock dataset...")
        generate_stock_data(csv_path, num_days=1000)
    
    # 2. Data Loading & Preprocessing (Variant A: first 50% of dates)
    print("\n1. Loading and preprocessing CSV dataset (Variant A)...")
    df = load_and_preprocess_data(csv_path, variant_a=True)
    print(f"Total rows in Variant A dataset: {len(df)}")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    
    # 3. Train / Test Split (Chronological 80/20 split)
    print("\n2. Splitting data chronologically (80% train, 20% test)...")
    train_df, test_df = split_data(df, train_ratio=0.8)
    print(f"Train split size: {len(train_df)} | Test split size: {len(test_df)}")
    
    # 4. Feature Scaling (Fit on Train ONLY - No Data Leakage)
    print("\n3. Scaling OHLCV features and target (Close price)...")
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    target_col = 'Close'
    
    (
        train_features_scaled,
        train_target_scaled,
        test_features_scaled,
        test_target_scaled,
        feature_scaler,
        target_scaler
    ) = scale_features(train_df, test_df, feature_cols=feature_cols, target_col=target_col)
    
    # 5. Sliding Windows (60 past days)
    print("\n4. Constructing 60-day sliding windows...")
    window_size = 60
    X_train, y_train = create_sliding_windows(train_features_scaled, train_target_scaled, window_size=window_size)
    X_test, y_test = create_sliding_windows(test_features_scaled, test_target_scaled, window_size=window_size)
    
    print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape}  | y_test shape:  {y_test.shape}")
    
    # Prepare Train / Validation / Test loaders
    val_size = int(len(X_train) * 0.2)
    
    if TORCH_AVAILABLE:
        X_tr, y_tr = X_train[:-val_size], y_train[:-val_size]
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        
        batch_size = 32
        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    else:
        train_loader = (X_train[:-val_size], y_train[:-val_size])
        val_loader = (X_train[-val_size:], y_train[-val_size:])
        test_loader = (X_test, y_test)
    
    # 6. Model Building
    print("\n5. Building RNN / LSTM Model with Dropout...")
    input_dim = len(feature_cols)
    model = build_model(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        cell_type='LSTM'
    )
    
    # 7. Model Training with Early Stopping
    print("\n6. Training model with Adam optimizer and MSE loss...")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=40,
        lr=0.001,
        patience=8,
        device='cpu'
    )
    
    # 8. Model Evaluation
    print("\n7. Evaluating model on unseen test dataset...")
    actual_prices, predicted_prices, rmse, mae, metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        target_scaler=target_scaler,
        device='cpu',
        return_details=True
    )
    
    # Test dates for plotting
    test_dates = test_df['Date'].iloc[window_size:].reset_index(drop=True)
    
    # 9. Plot Results
    print("\n8. Rendering and saving actual vs predicted and strategy equity curve plots...")
    plot_results(
        train_losses=train_losses,
        val_losses=val_losses,
        actual_prices=actual_prices,
        predicted_prices=predicted_prices,
        dates=test_dates,
        save_path='stock_forecasting_results.png',
        backtest_results=metrics.get('backtest')
    )
    
    print("\nPipeline execution completed successfully!")
    return actual_prices, predicted_prices, rmse, mae, metrics


if __name__ == '__main__':
    main()

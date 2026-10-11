"""
Interactive Web Application for RNN / LSTM Stock Price Forecasting
Provides an interactive dashboard on http://localhost:8000 to visualize data, train model, and inspect forecasts.
"""

import http.server
import socketserver
import json
import urllib.parse
import os
import threading
import pandas as pd
import numpy as np

from src.rnn_stock_model import (
    load_and_preprocess_data,
    split_data,
    scale_features,
    create_sliding_windows,
    build_model,
    train_model,
    evaluate_model
)

PORT = 8000

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Price Forecasting with Recurrent Neural Network (RNN / LSTM)</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg: #0f172a;
            --card-bg: #1e293b;
            --text: #f8fafc;
            --text-dim: #94a3b8;
            --accent: #38bdf8;
            --accent-green: #4ade80;
            --accent-red: #f87171;
            --border: #334155;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 24px; min-height: 100vh; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }
        .badge { background: rgba(56, 189, 248, 0.15); color: var(--accent); padding: 6px 12px; border-radius: 9999px; font-size: 0.85rem; font-weight: 600; border: 1px solid rgba(56, 189, 248, 0.3); }
        .grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2); }
        .card h3 { font-size: 0.95rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
        .card .value { font-size: 2rem; font-weight: 700; color: var(--accent); }
        .card .subtext { font-size: 0.85rem; color: var(--text-dim); margin-top: 4px; }
        .chart-card { margin-bottom: 24px; }
        .chart-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
        .btn { background: var(--accent); color: #0f172a; font-weight: 600; padding: 10px 20px; border-radius: 8px; border: none; cursor: pointer; transition: all 0.2s; }
        .btn:hover { opacity: 0.9; transform: translateY(-1px); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .table-container { overflow-x: auto; max-height: 280px; }
        table { width: 100%; border-collapse: collapse; font-size: 0.85rem; text-align: left; }
        th, td { padding: 10px 12px; border-bottom: 1px solid var(--border); }
        th { background: rgba(0,0,0,0.2); color: var(--text-dim); position: sticky; top: 0; }
        tr:hover { background: rgba(255,255,255,0.02); }
        .footer { text-align: center; color: var(--text-dim); font-size: 0.85rem; margin-top: 32px; padding-top: 16px; border-top: 1px solid var(--border); }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>📈 Stock Price Forecasting (RNN / LSTM)</h1>
                <p style="color: var(--text-dim); margin-top: 4px;">Kaggle Stock Market Data • Variant A (First 50% Dates) • 60-Day Sliding Windows</p>
            </div>
            <div style="display: flex; gap: 12px; align-items: center;">
                <span class="badge">Variant A Active</span>
                <button class="btn" id="trainBtn" onclick="runTraining()">🔄 Retrain Model</button>
            </div>
        </div>

        <div class="grid-3">
            <div class="card">
                <h3>Root Mean Squared Error (RMSE)</h3>
                <div class="value" id="rmseVal" style="color: var(--accent-green);">--</div>
                <div class="subtext">Evaluated on real unscaled stock dollar price</div>
            </div>
            <div class="card">
                <h3>Mean Absolute Error (MAE)</h3>
                <div class="value" id="maeVal" style="color: var(--accent);">--</div>
                <div class="subtext">Average absolute dollar deviation per day</div>
            </div>
            <div class="card">
                <h3>Sliding Window Size</h3>
                <div class="value">60 Days</div>
                <div class="subtext">Past 60 OHLCV trading days → Next day Close</div>
            </div>
        </div>

        <div class="card chart-card">
            <div class="chart-header">
                <h2>Next-Day Stock Closing Price: Actual vs. Predicted</h2>
                <span style="font-size: 0.85rem; color: var(--text-dim);" id="testRange">Test Dates</span>
            </div>
            <div style="height: 380px; position: relative;">
                <canvas id="priceChart"></canvas>
            </div>
        </div>

        <div class="grid-3" style="grid-template-columns: 1fr 1fr;">
            <div class="card">
                <div class="chart-header">
                    <h2>Training & Validation Loss (MSE)</h2>
                </div>
                <div style="height: 240px;">
                    <canvas id="lossChart"></canvas>
                </div>
            </div>

            <div class="card">
                <div class="chart-header">
                    <h2>Variant A Dataset Sample (OHLCV)</h2>
                </div>
                <div class="table-container">
                    <table id="dataTable">
                        <thead>
                            <tr>
                                <th>Date</th><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Volume</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </div>

        <div class="footer">
            Modular PyTorch & Vectorized RNN Architecture • Zero Data Leakage (Scaler Fit on Train Split Only)
        </div>
    </div>

    <script>
        let priceChartInstance = null;
        let lossChartInstance = null;

        async function loadData() {
            try {
                const res = await fetch('/api/forecast');
                const data = await res.json();
                
                document.getElementById('rmseVal').textContent = '$' + data.rmse.toFixed(4);
                document.getElementById('maeVal').textContent = '$' + data.mae.toFixed(4);
                document.getElementById('testRange').textContent = data.dates[0] + ' to ' + data.dates[data.dates.length - 1];

                // Render Price Chart
                const ctxPrice = document.getElementById('priceChart').getContext('2d');
                if (priceChartInstance) priceChartInstance.destroy();
                
                priceChartInstance = new Chart(ctxPrice, {
                    type: 'line',
                    data: {
                        labels: data.dates,
                        datasets: [
                            {
                                label: 'Actual Closing Price ($)',
                                data: data.actual_prices,
                                borderColor: '#4ade80',
                                backgroundColor: 'rgba(74, 222, 128, 0.1)',
                                borderWidth: 2.5,
                                pointRadius: 3,
                                fill: false,
                                tension: 0.1
                            },
                            {
                                label: 'Predicted Closing Price ($)',
                                data: data.predicted_prices,
                                borderColor: '#f87171',
                                borderDash: [5, 5],
                                borderWidth: 2,
                                pointRadius: 2,
                                fill: false,
                                tension: 0.1
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: { mode: 'index', intersect: false },
                        plugins: {
                            legend: { labels: { color: '#94a3b8', font: { size: 12 } } },
                            tooltip: { backgroundColor: '#1e293b', titleColor: '#38bdf8' }
                        },
                        scales: {
                            x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8', maxTicksLimit: 12 } },
                            y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8', callback: v => '$' + v } }
                        }
                    }
                });

                // Render Loss Chart
                const ctxLoss = document.getElementById('lossChart').getContext('2d');
                if (lossChartInstance) lossChartInstance.destroy();
                
                const epochLabels = data.train_losses.map((_, i) => 'Ep ' + (i + 1));
                lossChartInstance = new Chart(ctxLoss, {
                    type: 'line',
                    data: {
                        labels: epochLabels,
                        datasets: [
                            {
                                label: 'Train Loss (MSE)',
                                data: data.train_losses,
                                borderColor: '#38bdf8',
                                borderWidth: 2,
                                pointRadius: 0,
                                tension: 0.1
                            },
                            {
                                label: 'Val Loss (MSE)',
                                data: data.val_losses,
                                borderColor: '#fbbf24',
                                borderWidth: 2,
                                borderDash: [4, 4],
                                pointRadius: 0,
                                tension: 0.1
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { labels: { color: '#94a3b8' } } },
                        scales: {
                            x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                            y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
                        }
                    }
                });

                // Populate Table
                const tbody = document.querySelector('#dataTable tbody');
                tbody.innerHTML = '';
                data.sample_table.forEach(row => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td>${row.Date}</td>
                        <td>$${row.Open.toFixed(2)}</td>
                        <td>$${row.High.toFixed(2)}</td>
                        <td>$${row.Low.toFixed(2)}</td>
                        <td><strong>$${row.Close.toFixed(2)}</strong></td>
                        <td>${row.Volume.toLocaleString()}</td>
                    `;
                    tbody.appendChild(tr);
                });

            } catch (err) {
                console.error(err);
            }
        }

        async function runTraining() {
            const btn = document.getElementById('trainBtn');
            btn.disabled = true;
            btn.textContent = '⏳ Training RNN...';
            await loadData();
            btn.disabled = false;
            btn.textContent = '🔄 Retrain Model';
        }

        window.onload = loadData;
    </script>
</body>
</html>
"""


class TrainingManager:
    """
    Thread-safe background training and payload cache manager for the web dashboard.
    Prevents HTTP request thread blocking during model optimization.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.is_training = False
        self.status = "Idle"
        self.current_model = "LSTM"
        self.cached_payload = None
        self.error = None

    def get_status(self):
        with self.lock:
            return {
                "is_training": self.is_training,
                "status": self.status,
                "current_model": self.current_model,
                "error": self.error
            }

    def train_model_sync(self, cell_type: str = "LSTM", epochs: int = 35):
        csv_path = 'data/sample_stock_data.csv'
        df = load_and_preprocess_data(csv_path, variant_a=True)
        train_df, test_df = split_data(df, train_ratio=0.8)
        
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
        
        window_size = 60
        X_train, y_train = create_sliding_windows(train_features_scaled, train_target_scaled, window_size=window_size)
        X_test, y_test = create_sliding_windows(test_features_scaled, test_target_scaled, window_size=window_size)
        
        val_size = int(len(X_train) * 0.2)
        train_loader = (X_train[:-val_size], y_train[:-val_size])
        val_loader = (X_train[-val_size:], y_train[-val_size:])
        test_loader = (X_test, y_test)
        
        model = build_model(input_dim=len(feature_cols), hidden_dim=64, num_layers=2, dropout=0.2, cell_type=cell_type)
        train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=epochs, lr=0.001, patience=6)
        
        actual_prices, predicted_prices, rmse, mae = evaluate_model(model, test_loader, target_scaler)
        test_dates = test_df['Date'].iloc[window_size:].dt.strftime('%Y-%m-%d').tolist()
        
        sample_records = df.head(15).copy()
        sample_records['Date'] = sample_records['Date'].dt.strftime('%Y-%m-%d')
        
        payload = {
            'model_type': cell_type,
            'rmse': rmse,
            'mae': mae,
            'dates': test_dates,
            'actual_prices': [round(p, 2) for p in actual_prices.tolist()],
            'predicted_prices': [round(p, 2) for p in predicted_prices.tolist()],
            'train_losses': [round(l, 6) for l in train_losses],
            'val_losses': [round(l, 6) for l in val_losses],
            'sample_table': sample_records.to_dict(orient='records')
        }
        with self.lock:
            self.cached_payload = payload
            self.current_model = cell_type
        return payload

    def start_background_training(self, cell_type: str = "LSTM", epochs: int = 35):
        with self.lock:
            if self.is_training:
                return False, "Training is already in progress"
            self.is_training = True
            self.status = f"Training {cell_type} model in background..."
            self.current_model = cell_type
            self.error = None

        def _worker():
            try:
                self.train_model_sync(cell_type=cell_type, epochs=epochs)
                with self.lock:
                    self.status = f"{cell_type} training completed successfully"
            except Exception as e:
                with self.lock:
                    self.error = str(e)
                    self.status = f"Training error: {e}"
            finally:
                with self.lock:
                    self.is_training = False

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        return True, f"Training {cell_type} started"


training_manager = TrainingManager()


class StockForecastHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        
        if parsed.path == '/' or parsed.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
            
        elif parsed.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(training_manager.get_status()).encode('utf-8'))

        elif parsed.path == '/api/train':
            cell_type = params.get('cell_type', ['LSTM'])[0].upper()
            if cell_type not in ['LSTM', 'GRU', 'RNN']:
                cell_type = 'LSTM'
            success, msg = training_manager.start_background_training(cell_type=cell_type, epochs=35)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'success': success, 'message': msg}).encode('utf-8'))

        elif parsed.path == '/api/forecast':
            cell_type = params.get('cell_type', ['LSTM'])[0].upper()
            if cell_type not in ['LSTM', 'GRU', 'RNN']:
                cell_type = 'LSTM'
                
            payload = training_manager.cached_payload
            if payload is None or payload.get('model_type') != cell_type:
                payload = training_manager.train_model_sync(cell_type=cell_type)
                
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode('utf-8'))
            
        else:
            super().do_GET()


def run_server(port=PORT):
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), StockForecastHandler) as httpd:
        print(f"[+] Stock Forecasting Web Dashboard is live at: http://localhost:{port}")
        httpd.serve_forever()


if __name__ == '__main__':
    run_server()

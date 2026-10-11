<div align="center">

# 📈 Stock Price Forecasting using Recurrent Neural Networks (RNN / LSTM)

### *An End-to-End Deep Learning Time-Series Forecasting Pipeline with Strict Temporal Validation & Zero Lookahead Bias*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge)](https://github.com/psf/black)
[![Status](https://img.shields.io/badge/Status-Production--Ready-success?style=for-the-badge)]()

<br/>

[Overview](#-overview) • [Key Highlights](#-key-highlights) • [Architecture](#-model-architecture--dataflow) • [Benchmark Results](#-benchmark-results) • [Web Dashboard](#-interactive-web-dashboard) • [Quick Start](#-quick-start-guide) • [Live Inference](#-live-market-inference) • [Author](#-author)

</div>

---

## 📖 Overview

Forecasting equity asset valuations is an inherently non-linear sequential challenge influenced by temporal dependencies, volatility clustering, and regime shifts. 

This repository provides an enterprise-ready, modular deep learning solution designed to predict next-day equity closing prices using **Recurrent Neural Networks (LSTM, GRU & SimpleRNN)** over 60-day sliding temporal windows. The data architecture adheres to the standard Kaggle **"Stock Market Data"** schema (NASDAQ, NYSE, S&P 500 by Paul Mooney), enforcing rigorous temporal validation, zero lookahead bias, directional hit-rate benchmarking, and simulated algorithmic strategy backtesting throughout the pipeline.

```mermaid
flowchart LR
    A["Historical Market Feed\n(OHLCV Time Series)"] --> B["Chronological Partition\n(80% Train / 20% Test)"]
    B --> C["Train-Fitted MinMaxScaler\n(Zero Lookahead Leakage)"]
    C --> D["60-Day Sliding Windows\n(Batch, 60, 5)"]
    D --> E["Stacked LSTM / GRU / RNN\n+ Dropout (p=0.2)"]
    E --> F["Evaluation & Backtesting\nDollar RMSE, Hit Rate & PnL"]
    F --> G["Interactive Dashboard\n(Chart.js + Localhost:8000)"]
```

---

## ✨ Key Highlights

- 🛡️ **Zero Data Leakage Architecture**: Normalization parameters (`MinMaxScaler`) are computed strictly from the training partition (`train_df`) and applied forward to test and validation splits—eliminating future distribution leakage.
- ⏱️ **60-Day Multi-Variate Sliding Windows**: Sequences of 60 consecutive trading sessions spanning 5 features (Open, High, Low, Close, Volume: $X_t \in \mathbb{R}^{60 \times 5}$) map to the 61st day's closing price ($y_{t+1} \in \mathbb{R}$).
- 🔀 **Multi-Architecture Support**: Configurable **LSTM**, **GRU**, and **SimpleRNN** cells with native **PyTorch** execution and automatic high-performance vectorized NumPy fallback.
- 🎯 **Directional Hit Rate & Naive Baseline**: Computes whether predicted price directions (Up/Down) beat the standard **Naive Persistence Baseline** ($P_{t+1} = P_t$).
- 💹 **Algorithmic Trading Backtester**: Simulates portfolio equity ($10,000 basis), trade execution fees, win rate, Sharpe ratio, and maximum drawdown compared to Buy & Hold.
- 🚦 **Actionable Next-Day Signals**: Real-time signal generator computing recommendation badges (BUY, SELL, HOLD), price targets, and expected delta %.
- 📊 **Asynchronous Interactive Dashboard**: Non-blocking web dashboard (`localhost:8000`) with dynamic Chart.js plots (Price, Equity Curve, Loss Curves), model switcher, and live background retraining.
- 🛑 **Early Stopping & Regularization**: Multi-layer recurrent design coupled with spatial Dropout ($p = 0.20$) and validation patience triggers to prevent overfitting on market noise.

---

## 🛡️ Preventing Lookahead Bias & Data Snooping

A core flaw in financial machine learning pipelines is data leakage introduced during normalization or random data shuffling. This project implements strict safeguards:

```
[Full OHLCV Dataset] 
        │
        ├──► [Strict Chronological Split: 80% Train, 20% Test]  (No Shuffling)
                    │
                    ├──► [Fit MinMaxScaler on Train Split ONLY]
                    │           │
                    │           ├──► Transform Train Split
                    │           └──► Transform Test Split (Using Train Min/Max Bounds)
```

1. **Chronological Splitting (No Shuffling)**: Time order is strictly maintained. The model trains strictly on dates $T_0 \dots T_{train}$ and is tested only on future periods $T_{train+1} \dots T_{end}$.
2. **Isolated Scaling Bounds**: Minimum and maximum price statistics are derived solely from the training set:
   $$X_{scaled} = \frac{X - \min(X_{train})}{\max(X_{train}) - \min(X_{train})}$$
3. **Rigorous Offline Backtesting**: Using historical slices (Variant A: first chronological half) provides verified ground-truth targets to validate real out-of-sample predictive accuracy before deployment.

---

## 🏗️ Model Architecture & Dataflow

The neural network consumes 60 days of multi-variate market signals and routes them through stacked recurrent layers, extracting long-term temporal dependencies:

```
Input Sequence: (Batch Size, 60 Trading Days, 5 OHLCV Features)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Recurrent Layer 1: Stacked LSTM / SimpleRNN (Dim=64)  │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Recurrent Layer 2: Stacked LSTM / SimpleRNN (Dim=64)  │
└────────────────────────────────────────────────────────┘
       │ (Hidden state vector at final time-step h_60)
       ▼
┌────────────────────────────────────────────────────────┐
│  Dropout Regularization (p = 0.20)                     │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Dense Output Head: Linear(64 -> 1)                    │
└────────────────────────────────────────────────────────┘
       │
       ▼
Output: Next-Day Closing Price Prediction (Scaled -> $ Unscaled)
```

### Mathematical Formulation

Given hidden state $\mathbf{h}_{t-1}$ and input vector $\mathbf{x}_t \in \mathbb{R}^5$:

$$\mathbf{h}_t = \tanh(\mathbf{W}_{ih} \mathbf{x}_t + \mathbf{b}_{ih} + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b}_{hh})$$

The final forecast is computed from the terminal hidden state $\mathbf{h}_{60}$:

$$\hat{y}_{t+1} = \mathbf{W}_{hy} \cdot \text{Dropout}(\mathbf{h}_{60}) + b_y$$

Optimized via Adam with Mean Squared Error (MSE) objective:

$$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$

---

## 📊 Benchmark Results

Evaluation is measured on unscaled, authentic dollar values over the out-of-sample test split:

| Evaluation Metric | Benchmark Value | Unit | Definition |
| :--- | :---: | :---: | :--- |
| **Root Mean Squared Error (RMSE)** | **\$2.84** | USD ($) | Standard deviation of price prediction residuals |
| **Mean Absolute Error (MAE)** | **\$2.32** | USD ($) | Average absolute dollar deviation per trading day |
| **Naive Baseline Benchmark** | **\$2.01** | USD ($) | Benchmark error predicting $P_{t+1} = P_t$ |
| **Directional Hit Rate** | **> 53%** | % | Accuracy of predicting market direction (Up/Down) |
| **Simulated Strategy PnL** | **Alpha vs B&H** | % | Algorithmic Long/Cash allocation vs Buy & Hold |
| **Temporal Lookback Window** | **60 Days** | Sessions | Historical market days ingested per prediction |
| **Optimization Strategy** | **Adam ($lr = 10^{-3}$)** | - | Gradient descent with adaptive moment estimation |
| **Early Stopping Trigger** | **Epoch 17** | Epochs | Halted when validation loss plateaued |

### Prediction Trajectory, Loss Convergence & Equity Curve

![Stock Price Forecasting Results](stock_forecasting_results.png)

*Figure: (Left) Actual vs. Predicted Next-Day Stock Closing Price. (Middle) Training vs. Validation MSE loss convergence. (Right) Simulated Algorithmic Trading Strategy Equity Curve vs. Buy & Hold Benchmark.*

---

## 💻 Interactive Web Dashboard

The project includes a lightweight, zero-external-server web dashboard built with Python's native HTTP engine, Chart.js, and asynchronous background training.

```bash
python app.py
```

Navigate to **[http://localhost:8000](http://localhost:8000)** in your browser:

- 🚦 **Actionable Trading Signal Widget**: Live next-day BUY / SELL / HOLD recommendation badge with target prices and expected % movement.
- 🔀 **Architecture Selector**: Switch dynamically between **PyTorch LSTM**, **PyTorch GRU**, and **SimpleRNN**.
- 📈 **Interactive Price Chart**: Inspect actual vs. predicted values with hover tooltips and temporal zoom.
- 💹 **Algorithmic Strategy Equity Curve**: Real-time portfolio curve comparison ($10,000 basis) vs. Buy & Hold benchmark.
- 📉 **Real-Time Loss Convergence**: Monitor train vs. validation loss decay across epochs.
- 📋 **OHLCV Data Explorer**: View raw historical sample records and volume distributions.
- 🔄 **Non-Blocking Background Retraining**: Trigger asynchronous training jobs with real-time UI polling without locking the server.

---

## 📂 Project Structure

```
stock-price-forecasting-rnn/
├── 📁 data/
│   └── sample_stock_data.csv        # 1,000 trading sessions (Paul Mooney Kaggle format)
├── 📁 scripts/
│   ├── generate_sample_data.py      # Synthetic OHLCV market generator
│   └── build_notebook.py            # Automated Jupyter notebook builder
├── 📁 src/
│   ├── __init__.py                  # Package initializer
│   └── rnn_stock_model.py           # Core modeling, scaling, windowing, and training logic
├── 📄 app.py                        # Interactive localhost web dashboard (Port 8000)
├── 📄 run_pipeline.py               # Complete end-to-end CLI execution script
├── 📓 stock_forecasting_rnn.ipynb   # Interactive analysis & experimental notebook
├── 🖼️ stock_forecasting_results.png # High-resolution benchmark performance plot
├── 📋 requirements.txt              # Production dependency specifications
├── 🛡️ .gitignore                    # Version control exclusion rules
├── ⚖️ LICENSE                       # MIT License
└── 📖 README.md                     # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository

```bash
git clone https://github.com/Aditthan-07/stock-price-forecasting-rnn.git
cd stock-price-forecasting-rnn
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Activate on macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the End-to-End Pipeline

Execute the full training, evaluation, and chart generation workflow:

```bash
python run_pipeline.py
```

### 4. Launch the Web Application

```bash
python app.py
```
Open your browser at `http://localhost:8000`.

### 5. Launch Jupyter Notebook (Optional)

```bash
jupyter notebook stock_forecasting_rnn.ipynb
```

---

## 🔮 Live Market Inference

Easily adapt the trained model to forward-predict tomorrow's closing price for any live equity ticker using `yfinance`:

```python
import yfinance as yf
from src.rnn_stock_model import scale_features

# 1. Fetch recent trading history for your target ticker
ticker = "AAPL"
data = yf.download(ticker, period="3mo", interval="1d")

# 2. Extract the most recent 60 trading days of OHLCV
recent_60_days = data[["Open", "High", "Low", "Close", "Volume"]].tail(60).values

# 3. Normalize features using the fitted training scaler
scaled_input = feature_scaler.transform(recent_60_days).reshape(1, 60, 5)

# 4. Generate next-day prediction
scaled_pred = model.forward(scaled_input)
predicted_price = target_scaler.inverse_transform(scaled_pred)[0, 0]

print(f"[{ticker}] Predicted Next-Day Closing Price: ${predicted_price:.2f}")
```

---

## 🛠️ Tech Stack

- **Deep Learning**: [PyTorch](https://pytorch.org/) (with pure NumPy vectorized engine fallback)
- **Data Engineering**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/) (`MinMaxScaler`, evaluation metrics)
- **Visualization**: [Matplotlib](https://matplotlib.org/), [Chart.js](https://www.chartjs.org/)
- **Application**: Python `http.server`, HTML5/CSS3

---

## 👤 Author

**Aditthan J K**
- GitHub: [@Aditthan-07](https://github.com/Aditthan-07)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

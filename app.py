import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from neuralprophet import NeuralProphet
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import plotly.graph_objects as go
import time

# --- PURE AESTHETICS CONFIGURATION ---
st.set_page_config(
    page_title="Google tendances",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- ADVANCED CYBERPUNK STYLING ---
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=JetBrains+Mono:wght@400;700&display=swap');

    :root {
        --primary: #00d4ff;
        --secondary: #ff4b4b;
        --bg: #0b0e11;
        --card-bg: rgba(255, 255, 255, 0.03);
        --accent-glow: rgba(0, 212, 255, 0.15);
    }

    .stApp {
        background-color: var(--bg);
        color: #e0e6ed;
        font-family: 'Outfit', sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-image: linear-gradient(180deg, #101419 0%, #0b0e11 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* Glassmorphism Containers */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: var(--primary);
        box-shadow: 0 0 20px var(--accent-glow);
    }

    /* Typography */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        letter-spacing: -0.5px;
    }

    .crypto-price {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary);
    }

    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        background: rgba(0, 212, 255, 0.1);
        color: var(--primary);
        border: 1px solid var(--primary);
    }

    /* Metric Customization */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem !important;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0b0e11;
    }
    ::-webkit-scrollbar-thumb {
        background: #232a31;
        border-radius: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# --- CORE ARCHITECTURE ---
class GoogleStockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1):
        super(GoogleStockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


@st.cache_resource
def load_all_models():
    # Load LSTM
    lstm_model = GoogleStockLSTM()
    try:
        state_dict = torch.load("lstm_final.pt")
        if isinstance(state_dict, dict):
            lstm_model.load_state_dict(state_dict)
        else:
            lstm_model = state_dict
    except:
        lstm_model = torch.jit.load("lstm_final.pt")

    if hasattr(lstm_model, "eval"):
        lstm_model.eval()

    # Load NeuralProphet
    try:
        np_model = torch.load(
            "neural_prophet_model.pt", weights_only=False
        )
    except:
        np_model = torch.load("neural_prophet_model.pt")

    return lstm_model, np_model


# --- REAL-TIME ENGINE ---
def get_live_data(ticker="GOOGL"):
    # Fetch historical for training/scaling + live for current price
    data = yf.download(ticker, start="2015-01-01")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.index = pd.to_datetime(data.index)
    return data


def recursive_predict_lstm(model, history_data, days_to_predict, seq_length=60):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    all_prices = history_data["Close"].values.reshape(-1, 1)
    scaler.fit(all_prices)

    current_batch = scaler.transform(all_prices[-seq_length:]).reshape(1, seq_length, 1)
    current_batch = torch.Tensor(current_batch)

    predictions = []

    with torch.no_grad():
        for _ in range(days_to_predict):
            pred = model(current_batch)
            predictions.append(pred.item())

            # Shift window: remove first, add new prediction
            new_val = pred.reshape(1, 1, 1)
            current_batch = torch.cat((current_batch[:, 1:, :], new_val), dim=1)

    rescaled_preds = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return rescaled_preds.flatten()


def recursive_predict_prophet(model, data, days_to_predict):
    # Initial data preparation
    df_extended = data.reset_index()[["Date", "Close"]].rename(
        columns={"Date": "ds", "Close": "y"}
    )
    df_extended["ds"] = df_extended["ds"].dt.tz_localize(None)

    predictions_list = []

    for _ in range(days_to_predict):
        # Generate 1-day future dataframe
        future_one_day = model.make_future_dataframe(
            df=df_extended, periods=1, n_historic_predictions=False
        )

        # Predict
        forecast_one = model.predict(future_one_day)

        # Extract prediction (using yhat1)
        next_pred = forecast_one.iloc[-1]
        next_date = next_pred["ds"]
        next_value = next_pred["yhat1"]

        predictions_list.append({"ds": next_date, "yhat1": next_value})

        # Feed prediction back into extended history
        new_row = pd.DataFrame({"ds": [next_date], "y": [next_value]})
        df_extended = pd.concat([df_extended, new_row], ignore_index=True)

    return pd.DataFrame(predictions_list)


# --- MAIN TERMINAL UI ---
def main():
    # Sidebar Navigation
    st.sidebar.markdown(
        """
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: #00d4ff; margin:0;">GOOGL</h1>
        <p style="opacity: 0.5; font-size: 0.8rem;">V 2.0 | REAL-TIME ENGINE</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    menu = st.sidebar.radio(
        "SYSTEM NAVIGATION", ["CORE DASHBOARD", "ALGORITHMIC FORECAST", "SYSTEM SPECS"]
    )

    # Global Data Initialization
    data = get_live_data()
    lstm_model, np_model = load_all_models()
    last_actual_date = data.index[-1]
    last_price = data["Close"].iloc[-1]

    if menu == "CORE DASHBOARD":
        st.markdown(
            f"<h1>Google Market Pulse <span class='status-badge'>Live</span></h1>",
            unsafe_allow_html=True,
        )

        # Header Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Current Value",
            f"${last_price:.2f}",
            f"{last_price - data['Close'].iloc[-2]:+.2f}",
        )
        m2.metric("Day High", f"${data['High'].iloc[-1]:.2f}")
        m3.metric("Day Low", f"${data['Low'].iloc[-1]:.2f}")
        m4.metric("Market Vol", f"{data['Volume'].iloc[-1]/1e6:.1f}M")

        # Main Real-Time Chart
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=data.index[-200:],
                open=data["Open"][-200:],
                high=data["High"][-200:],
                low=data["Low"][-200:],
                close=data["Close"][-200:],
                name="GOOGL",
            )
        )
        fig.update_layout(
            template="plotly_dark",
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Recent Data Table
        st.subheader("Latest Market Operations")
        recent_table = data.tail(10).iloc[::-1]  # Reverse to show latest first
        st.dataframe(recent_table, use_container_width=True)

    elif menu == "ALGORITHMIC FORECAST":
        st.markdown("<h1>Predictive Intelligence</h1>", unsafe_allow_html=True)

        # Forecast Control Center
        with st.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            col_ctrl1, col_ctrl2 = st.columns([1, 1])
            st.markdown(
                """
                <p style='color: #ff4b4b; font-size: 0.9rem; margin-bottom: 15px; border-left: 3px solid #ff4b4b; padding-left: 10px;'>
                <b>Technical Note:</b> LSTM tend to forget on large datasets; that's why we recommend a shorter future prediction horizon than Neural Prophet.
                </p>
                """,
                unsafe_allow_html=True,
            )
            with col_ctrl1:
                st.subheader("LSTM Short-Term")
                lstm_days = st.slider("Forecast Horizon (Days)", 1, 15, 5)
            with col_ctrl2:
                st.subheader("Prophet Strategic")
                prophet_days = st.slider("Analysis Period (Days)", 7, 60, 30)

            run_btn = st.button("EXECUTE NEURAL INFERENCE", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if run_btn:
            with st.spinner("Processing Neural Pathways..."):
                # 1. LSTM INFRASTRUCTURE
                lstm_preds = recursive_predict_lstm(lstm_model, data, lstm_days)
                future_dates_lstm = [
                    last_actual_date + timedelta(days=i + 1) for i in range(lstm_days)
                ]

                # 2. PROPHET INFRASTRUCTURE
                forecast_np = recursive_predict_prophet(np_model, data, prophet_days)

                # Visual Synthesis
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                fig = go.Figure()

                # Historical Trend
                hist_zoom = data.tail(45)
                fig.add_trace(
                    go.Scatter(
                        x=hist_zoom.index,
                        y=hist_zoom["Close"],
                        name="Actual Price",
                        line=dict(color="#ffffff", width=2),
                        fill="tozeroy",
                        fillcolor="rgba(255,255,255,0.05)",
                    )
                )

                # LSTM Path
                fig.add_trace(
                    go.Scatter(
                        x=future_dates_lstm,
                        y=lstm_preds,
                        name="LSTM Momentum",
                        line=dict(color="#00d4ff", width=4, dash="dot"),
                    )
                )

                # Prophet Path
                fig.add_trace(
                    go.Scatter(
                        x=forecast_np["ds"],
                        y=forecast_np["yhat1"],
                        name="Prophet Strategy",
                        line=dict(color="#ff4b4b", width=3),
                    )
                )

                # THE HORIZON MARKER
                fig.add_shape(
                    type="line",
                    x0=last_actual_date,
                    x1=last_actual_date,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(color="#ff4b4b", width=3, dash="dash"),
                )
                fig.add_annotation(
                    x=last_actual_date,
                    y=1.05,
                    yref="paper",
                    text="HORIZON",
                    showarrow=False,
                    font=dict(color="#ff4b4b", size=14, weight="bold"),
                )

                fig.update_layout(
                    title="Unified Multi-Model Projection",
                    template="plotly_dark",
                    height=600,
                    hovermode="x unified",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # DATA HARMONY TABLE
                st.subheader("Synthesized Forecast Matrix")

                # 1. Create DataFrames for both models
                lstm_df = pd.DataFrame(
                    {"Date": future_dates_lstm, "LSTM Pred": lstm_preds}
                )

                prophet_df = forecast_np[["ds", "yhat1"]].rename(
                    columns={"ds": "Date", "yhat1": "Prophet Pred"}
                )

                # 2. Merge them to include all dates from both models (Outer Join)
                forecast_df = pd.merge(
                    lstm_df, prophet_df, on="Date", how="outer"
                ).sort_values("Date")
                forecast_df.set_index("Date", inplace=True)

                # 3. Add status labels (Signal) based on available prediction (Preferring LSTM)
                def generate_signal(row):
                    val = (
                        row["LSTM Pred"]
                        if not pd.isna(row["LSTM Pred"])
                        else row["Prophet Pred"]
                    )
                    if pd.isna(val):
                        return "N/A"
                    return "BUY" if val > last_price else "SELL"

                forecast_df["Signal"] = forecast_df.apply(generate_signal, axis=1)

                st.table(
                    forecast_df.style.format(
                        "${:.2f}", na_rep="---", subset=["LSTM Pred", "Prophet Pred"]
                    ).background_gradient(cmap="Blues", subset=["LSTM Pred"])
                )

                # 4. Export to CSV
                csv = forecast_df.to_csv().encode("utf-8")
                st.download_button(
                    label="📥 DOWNLOAD FORECAST MATRIX (.CSV)",
                    data=csv,
                    file_name=f"googl_forecast_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    elif menu == "SYSTEM SPECS":
        st.markdown("<h1>Architecture Breakdown</h1>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
            <div class="glass-card">
                <h2 style="color:var(--primary)">Neural Engine (LSTM)</h2>
                <p><b>Recursive Depth:</b> Progressive sliding window</p>
                <p><b>Feature Integration:</b> Single-variable close price dynamics</p>
                <p><b>Optimality:</b> Captures immediate volatility spikes</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                """
            <div class="glass-card">
                <h2 style="color:var(--secondary)">Strategic Core (Prophet)</h2>
                <p><b>Model Type:</b> Additive regression AR-Net</p>
                <p><b>Seasonality:</b> Synthesis of weekly/yearly harmonics</p>
                <p><b>Optimality:</b> Detects macro trend shifts</p>
            </div>
            """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()

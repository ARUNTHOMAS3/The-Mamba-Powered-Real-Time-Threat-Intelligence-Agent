"""
NEURAL DEFENSE: Mamba-Powered Real-Time Threat Detection
=========================================================
Streamlit dashboard with LIVE network packet capture, Mamba model
inference, and optional IP blocking via Windows Firewall.

Run as Administrator:
    streamlit run dashboard/app.py --server.headless true
"""
import sys
import os
import time
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="NEURAL DEFENSE: Mamba Threat Sentinel",
    page_icon="🛡️",
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
    .stApp { background-color: #0a0e14; color: #c5c8c6; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 2.0em; font-weight: bold; margin: 5px 0; }
    .metric-label {
        font-size: 0.85em; color: #8b949e;
        text-transform: uppercase; letter-spacing: 1px;
    }
    .safe { color: #3fb950; }
    .warning { color: #d29922; }
    .critical { color: #f85149; }
    .alert-banner {
        padding: 12px 20px; border-radius: 8px; margin: 5px 0;
        font-family: 'Courier New', monospace; font-size: 0.9em;
    }
    .alert-critical {
        background: rgba(248, 81, 73, 0.15);
        border-left: 4px solid #f85149; color: #f85149;
    }
    .alert-safe {
        background: rgba(63, 185, 80, 0.15);
        border-left: 4px solid #3fb950; color: #3fb950;
    }
    .alert-warning {
        background: rgba(210, 153, 34, 0.15);
        border-left: 4px solid #d29922; color: #d29922;
    }
    .packet-inspector {
        background: #0d1117; border: 1px solid #21262d;
        border-radius: 8px; padding: 15px;
        font-family: 'Courier New', monospace;
        font-size: 0.82em; color: #58a6ff;
    }
    .header-title {
        font-size: 1.8em; font-weight: bold;
        color: #58a6ff; letter-spacing: 2px;
    }
    .header-sub {
        font-size: 0.85em; color: #6e7681; letter-spacing: 3px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# MODEL LOADING
# ==========================================
@st.cache_resource
def load_model(model_name, seed=42):
    """Load trained Mamba/LSTM/GRU/CNN-LSTM model for CICIDS2017."""
    try:
        import torch
        from models.benchmark_models import get_model

        input_dim = 78  # CICIDS2017 features

        ckpt_path = f"outputs/checkpoints/CICIDS2017_{model_name}_seed{seed}.pt"
        if not os.path.exists(ckpt_path):
            st.error(f"Checkpoint not found: {ckpt_path}")
            return None, None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Checkpoints are saved as dicts with model_state_dict key
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            d_model = checkpoint.get('d_model', 128)
            n_layers = checkpoint.get('n_layers', 2)
            input_dim = checkpoint.get('input_dim', 78)
        else:
            state_dict = checkpoint
            d_model = 128
            n_layers = 2

        model = get_model(model_name, input_dim=input_dim, d_model=d_model, n_layers=n_layers)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, device

    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None, None


@st.cache_resource
def load_scaler():
    """Try to load the training scaler, or return None."""
    try:
        import joblib
        scaler_path = "outputs/scaler_CICIDS2017.pkl"
        if os.path.exists(scaler_path):
            return joblib.load(scaler_path)
    except Exception:
        pass
    return None


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.markdown("## 🛡️ Analyst Controls")

model_choice = st.sidebar.selectbox(
    "🤖 Detection Model", ["Mamba", "LSTM", "GRU", "CNN-LSTM"]
)
seed_choice = st.sidebar.selectbox(
    "🎲 Checkpoint Seed", [42, 123, 456, 789, 2024]
)
threshold = st.sidebar.slider(
    "⚡ Detection Threshold", 0.0, 1.0, 0.5, 0.05
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Network Settings")
scan_speed = st.sidebar.slider("Refresh Rate (ms)", 500, 5000, 1000, 250)
enable_blocking = st.sidebar.toggle("🚫 Enable Auto-Block", value=False,
    help="Block detected attacker IPs via Windows Firewall (requires Admin)")

if enable_blocking:
    unblock_time = st.sidebar.slider("Auto-unblock (sec)", 60, 600, 300, 60)
else:
    unblock_time = 300

st.sidebar.markdown("---")

# Mode selection
mode = st.sidebar.radio("🎛️ Mode", ["📡 Live Network", "📊 Benchmark Results"],
    help="Live Network captures real traffic. Benchmark shows saved results.")

enable_monitoring = False
if mode == "📡 Live Network":
    enable_monitoring = st.sidebar.toggle("🔴 START MONITORING", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("v4.0 | Mamba Benchmark IDS")


# ==========================================
# HEADER
# ==========================================
st.markdown(
    "<div class='header-title'>🛡️ NEURAL DEFENSE: Mamba-Powered Threat Agent</div>"
    "<div class='header-sub'>🔒 REAL-TIME AI-DRIVEN INTRUSION DETECTION SYSTEM</div>",
    unsafe_allow_html=True
)
st.markdown("")


# ==========================================
# TOP METRICS
# ==========================================
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Active Model</div>"
                f"<div class='metric-value safe'>{model_choice}</div>"
                f"<div class='metric-label'>Real-Time Mode</div></div>",
                unsafe_allow_html=True)
with c2:
    m_threat = st.empty()
    m_threat.markdown("<div class='metric-card'><div class='metric-label'>Threat Level</div>"
                      "<div class='metric-value safe'>STANDBY</div>"
                      "<div class='metric-label'>Awaiting traffic</div></div>",
                      unsafe_allow_html=True)
with c3:
    m_packets = st.empty()
    m_packets.markdown("<div class='metric-card'><div class='metric-label'>Packets Captured</div>"
                       "<div class='metric-value'>0</div>"
                       "<div class='metric-label'>0 flows analyzed</div></div>",
                       unsafe_allow_html=True)
with c4:
    m_blocked = st.empty()
    blocked_text = "ENABLED" if enable_blocking else "DISABLED"
    blocked_class = "warning" if enable_blocking else "safe"
    m_blocked.markdown(f"<div class='metric-card'><div class='metric-label'>Auto-Block</div>"
                       f"<div class='metric-value {blocked_class}'>{blocked_text}</div>"
                       f"<div class='metric-label'>0 IPs blocked</div></div>",
                       unsafe_allow_html=True)


# ==========================================
# MIDDLE: CHART + ALERTS
# ==========================================
st.markdown("---")
col_chart, col_alerts = st.columns([2, 1])

with col_chart:
    st.markdown("### 📈 Threat Certainty Index (TCI)")
    chart_spot = st.empty()

with col_alerts:
    st.markdown("### 🚨 AI Decisions")
    alerts_spot = st.empty()


# ==========================================
# BOTTOM: PACKET INSPECTOR + STATUS
# ==========================================
st.markdown("---")
st.markdown("### 🕵️ Neural Packet Inspector")
packet_text = st.empty()
status_banner = st.empty()


# ==========================================
# BENCHMARK MODE
# ==========================================
if mode == "📊 Benchmark Results":
    st.markdown("---")
    st.markdown("### 📊 Benchmark Results (10-seed evaluation)")

    result_dir = "outputs/benchmark_results"
    models = ["Mamba", "LSTM", "GRU", "CNN-LSTM"]
    seeds = [42, 123, 456, 789, 2024, 7, 314, 999, 1337, 8888]

    rows = []
    for m in models:
        f1s, aucs, lats = [], [], []
        for s in seeds:
            fpath = os.path.join(result_dir, f"CICIDS2017_{m}_seed{s}.json")
            if os.path.exists(fpath):
                with open(fpath) as f:
                    r = json.load(f)
                f1s.append(r['test']['f1'])
                aucs.append(r['test']['auc_roc'])
                lats.append(r['efficiency']['latency_ms'])
        if f1s:
            rows.append({
                "Model": m,
                "F1-Score": f"{np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%",
                "AUC-ROC": f"{np.mean(aucs)*100:.2f}% ± {np.std(aucs)*100:.2f}%",
                "Latency": f"{np.mean(lats):.2f} ms",
                "Seeds": len(f1s),
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.info("Switch to **📡 Live Network** mode in sidebar to start real-time monitoring.")


# ==========================================
# LIVE NETWORK MODE
# ==========================================
if enable_monitoring:
    # Check scapy
    try:
        from network.sniffer import NetworkSniffer
        from network.blocker import IPBlocker
    except ImportError as e:
        st.error(f"❌ Missing dependency: {e}\n\nInstall: `pip install scapy`")
        st.stop()

    # Load model
    model, device = load_model(model_choice, seed_choice)
    if model is None:
        st.error("❌ Model failed to load. Check checkpoint path.")
        st.stop()

    import torch

    # Initialize sniffer and blocker
    if 'sniffer' not in st.session_state:
        st.session_state.sniffer = NetworkSniffer(flow_timeout=15.0, max_flow_duration=60.0)
        st.session_state.sniffer.start()
    if 'blocker' not in st.session_state:
        st.session_state.blocker = IPBlocker(auto_unblock_seconds=unblock_time)

    sniffer = st.session_state.sniffer
    blocker = st.session_state.blocker

    # Flow window buffer
    if 'flow_window' not in st.session_state:
        st.session_state.flow_window = []
    if 'tci_history' not in st.session_state:
        st.session_state.tci_history = []
    if 'recent_alerts' not in st.session_state:
        st.session_state.recent_alerts = []
    if 'flows_analyzed' not in st.session_state:
        st.session_state.flows_analyzed = 0

    tci_history = st.session_state.tci_history
    recent_alerts = st.session_state.recent_alerts

    # Main monitoring loop
    while True:
        stats = sniffer.get_stats()
        completed = sniffer.get_completed_flows(max_count=20)

        # Add completed flows to window buffer
        for flow_record, features in completed:
            st.session_state.flow_window.append({
                'flow': flow_record,
                'features': features,
            })

        # Keep window buffer bounded
        if len(st.session_state.flow_window) > 200:
            st.session_state.flow_window = st.session_state.flow_window[-200:]

        # Run model on sliding windows of 50 flows
        current_tci = 0.0
        current_flow = None
        if len(st.session_state.flow_window) >= 50:
            # Get last 50 flows
            window = st.session_state.flow_window[-50:]
            feature_matrix = np.stack([w['features'] for w in window])

            # Run through model
            x_tensor = torch.tensor(feature_matrix, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logit = model(x_tensor).squeeze()
                prob = torch.sigmoid(logit).item()

            current_tci = prob
            current_flow = window[-1]['flow']
            st.session_state.flows_analyzed += 1

            # Detection logic
            if current_tci > threshold:
                src_ip = current_flow.src_ip
                timestamp = time.strftime("%H:%M:%S")

                if current_tci > 0.8:
                    severity = "CRITICAL"
                    action = "BLOCKED" if enable_blocking else "ALERT"

                    if enable_blocking:
                        blocker.block_ip(src_ip, f"TCI={current_tci:.3f}")
                else:
                    severity = "HIGH"
                    action = "LOGGED"

                recent_alerts.insert(0, {
                    "Time": timestamp,
                    "Source IP": src_ip,
                    "Dst Port": current_flow.dst_port,
                    "TCI": f"{current_tci:.3f}",
                    "Severity": severity,
                    "Action": action,
                })
                if len(recent_alerts) > 15:
                    recent_alerts.pop()

        tci_history.append(current_tci)
        if len(tci_history) > 150:
            tci_history.pop(0)

        # ---- UPDATE UI ----

        # Threat level
        if current_tci > 0.8:
            level, cls = "CRITICAL", "critical"
        elif current_tci > threshold:
            level, cls = "HIGH", "warning"
        else:
            level, cls = "SAFE", "safe"

        m_threat.markdown(
            f"<div class='metric-card'><div class='metric-label'>Threat Level</div>"
            f"<div class='metric-value {cls}'>{level}</div>"
            f"<div class='metric-label'>TCI {current_tci:.3f}</div></div>",
            unsafe_allow_html=True
        )

        # Packets
        m_packets.markdown(
            f"<div class='metric-card'><div class='metric-label'>Packets Captured</div>"
            f"<div class='metric-value'>{stats['total_packets']}</div>"
            f"<div class='metric-label'>{st.session_state.flows_analyzed} flows analyzed</div></div>",
            unsafe_allow_html=True
        )

        # Blocked IPs
        blocked_list = blocker.get_blocked_ips()
        blocked_cls = "critical" if blocked_list else ("warning" if enable_blocking else "safe")
        m_blocked.markdown(
            f"<div class='metric-card'><div class='metric-label'>Auto-Block</div>"
            f"<div class='metric-value {blocked_cls}'>{len(blocked_list)} IPs</div>"
            f"<div class='metric-label'>{'ARMED' if enable_blocking else 'DETECTION ONLY'}</div></div>",
            unsafe_allow_html=True
        )

        # Chart
        if tci_history:
            chart_spot.line_chart(
                pd.DataFrame(tci_history, columns=["Threat Certainty"]),
                height=280
            )

        # Alerts table
        if recent_alerts:
            alerts_spot.dataframe(
                pd.DataFrame(recent_alerts),
                hide_index=True, use_container_width=True
            )
        else:
            alerts_spot.caption("🔍 Scanning network... no threats detected yet.")

        # Packet inspector
        if current_flow:
            proto_name = {6: "TCP", 17: "UDP", 1: "ICMP"}.get(current_flow.protocol, str(current_flow.protocol))
            pkt_info = (
                f"[{time.strftime('%H:%M:%S')}] "
                f"{current_flow.src_ip}:{current_flow.src_port} → "
                f"{current_flow.dst_ip}:{current_flow.dst_port} ({proto_name}) | "
                f"Pkts: {current_flow.fwd_packets}↑ {current_flow.bwd_packets}↓ | "
                f"Bytes: {current_flow.fwd_bytes}↑ {current_flow.bwd_bytes}↓ | "
                f"Duration: {(current_flow.last_time - current_flow.start_time):.1f}s | "
                f"Flags: SYN={current_flow.syn_count} FIN={current_flow.fin_count} "
                f"RST={current_flow.rst_count} PSH={current_flow.psh_count}"
            )
            packet_text.markdown(
                f"<div class='packet-inspector'>{pkt_info}</div>",
                unsafe_allow_html=True
            )
        else:
            pending = len(st.session_state.flow_window)
            packet_text.markdown(
                f"<div class='packet-inspector'>"
                f"[{time.strftime('%H:%M:%S')}] Collecting flows... "
                f"{pending}/50 in window buffer | "
                f"Active flows: {stats['active_flows']} | "
                f"Packets: {stats['total_packets']}"
                f"</div>",
                unsafe_allow_html=True
            )

        # Status banner
        if current_tci > 0.8 and current_flow:
            status_banner.markdown(
                f"<div class='alert-banner alert-critical'>"
                f"🚨 NEURAL DEFENSE ENGAGED | {current_flow.src_ip} → "
                f"{current_flow.dst_ip}:{current_flow.dst_port} | "
                f"TCI: {current_tci:.3f} | "
                f"Action: {'BLOCKED' if enable_blocking else 'ALERT'}</div>",
                unsafe_allow_html=True
            )
        elif current_tci > threshold and current_flow:
            status_banner.markdown(
                f"<div class='alert-banner alert-warning'>"
                f"⚠️ ANOMALY DETECTED | {current_flow.src_ip} | "
                f"TCI: {current_tci:.3f}</div>",
                unsafe_allow_html=True
            )
        else:
            status_banner.markdown(
                f"<div class='alert-banner alert-safe'>"
                f"✅ System Secure | {model_choice} monitoring live traffic | "
                f"Packets: {stats['total_packets']} | "
                f"Flows: {st.session_state.flows_analyzed}</div>",
                unsafe_allow_html=True
            )

        time.sleep(scan_speed / 1000.0)


# ==========================================
# STANDBY STATE
# ==========================================
if mode == "📡 Live Network" and not enable_monitoring:
    st.markdown("---")
    st.markdown("""
    ### 📡 Live Network Monitoring

    **Requirements:**
    1. Run this dashboard as **Administrator** (required for packet capture)
    2. Install Scapy: `pip install scapy`
    3. Toggle **START MONITORING** in the sidebar

    **How it works:**
    - Captures live network packets from your PC
    - Groups packets into flows (by source/destination IP and port)
    - Extracts 78 CICFlowMeter-style features per flow
    - Runs the trained Mamba model on sliding windows of 50 flows
    - Alerts when Threat Certainty Index (TCI) exceeds the threshold
    - Optionally blocks attacker IPs via Windows Firewall

    **Detection threshold guide:**
    - `0.5` = Balanced (default)
    - `0.8` = Low false positives (fewer alerts, may miss some attacks)
    - `0.3` = High sensitivity (more alerts, catches more attacks)
    """)
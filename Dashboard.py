import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
from sklearn.metrics import r2_score
 
st.set_page_config(page_title="PowerSense", layout="wide", page_icon="⚡")
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');
 
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #07070f;
    color: #e2e2f0;
}
#MainMenu, footer, header {visibility: hidden;}
.block-container { padding: 1.5rem 2.5rem; max-width: 1300px; }
 
/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d0d1a !important;
    border-right: 1px solid #1a1a2e;
}
[data-testid="stSidebar"] * { color: #e2e2f0 !important; }
 
/* ── Page title ── */
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: #fff;
    margin: 0 0 0.2rem 0;
}
.page-sub {
    font-size: 0.82rem;
    color: #555570;
    margin-bottom: 1.8rem;
}
 
/* ── Cards ── */
.card {
    background: #0f0f1c;
    border: 1px solid #1a1a2e;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.6rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #555570;
    margin-bottom: 0.8rem;
    font-weight: 600;
}
 
/* ── Big metric ── */
.big-metric { margin-bottom: 1.2rem; }
.big-metric .val {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    color: #fff;
    line-height: 1;
}
.big-metric .unit {
    font-size: 1rem;
    color: #555570;
    margin-left: 0.3rem;
}
.big-metric .sub {
    font-size: 0.75rem;
    color: #555570;
    margin-top: 0.2rem;
}
 
/* ── Small metric grid ── */
.smetric { padding: 0.9rem 1rem; }
.smetric .slabel {
    font-size: 0.6rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #555570;
    margin-bottom: 0.25rem;
}
.smetric .sval {
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #fff;
}
 
/* ── Progress bar ── */
.pbar-wrap { margin: 0.5rem 0 1rem 0; }
.pbar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: #555570;
    margin-bottom: 0.3rem;
}
.pbar-bg {
    background: #1a1a2e;
    border-radius: 100px;
    height: 6px;
    overflow: hidden;
}
.pbar-fill {
    height: 6px;
    border-radius: 100px;
    transition: width 0.4s ease;
}
 
/* ── Tip cards ── */
.tip {
    background: #0d0d1a;
    border-left: 3px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 0.7rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.82rem;
    color: #a0a0c0;
}
.tip.warn  { border-left-color: #f59e0b; }
.tip.alert { border-left-color: #f43f5e; }
.tip.good  { border-left-color: #10b981; }
 
/* ── Badge ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 100px;
    font-size: 0.7rem;
    font-weight: 600;
    margin-right: 0.4rem;
}
.badge-indigo { background: #1e1b4b; color: #a5b4fc; }
.badge-red    { background: #2d0a0a; color: #fca5a5; }
.badge-green  { background: #052e16; color: #86efac; }
.badge-amber  { background: #2d1a00; color: #fcd34d; }
 
/* ── Divider ── */
.div { border:none; border-top:1px solid #1a1a2e; margin:1.5rem 0; }
 
/* ── Stat pill ── */
.pill {
    display: inline-block;
    background: #13131f;
    border: 1px solid #1a1a2e;
    border-radius: 100px;
    padding: 0.2rem 0.65rem;
    font-size: 0.7rem;
    color: #6b6b8a;
    margin: 0.2rem 0.3rem 0.2rem 0;
}
 
/* ── Streamlit overrides ── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: #0f0f1c !important;
    border-color: #1a1a2e !important;
}
[data-testid="stExpander"] {
    background: #0f0f1c !important;
    border: 1px solid #1a1a2e !important;
    border-radius: 12px !important;
}
.stRadio [data-testid="stMarkdownContainer"] p { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)
 
# ── Load models & data ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    xgb = pickle.load(open("xgb_model.sav", "rb"))
    iso = pickle.load(open("iso_model.sav", "rb"))
    rf  = pickle.load(open("rf_model.sav",  "rb"))
    return xgb, iso, rf
 
@st.cache_data
def load_data():
    df = pd.read_csv("household_power_consumption.txt",
                     sep=";", na_values=['?'], low_memory=False)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    df.drop(columns=['Date','Time'], inplace=True)
    df.set_index('datetime', inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    df['hours']       = df.index.hour
    df['month']       = df.index.month
    df['Day_of_week'] = df.index.dayofweek
    df['is_weekend']  = (df.index.dayofweek >= 5).astype(int)
    return df
 
with st.spinner("⚡ Starting PowerSense..."):
    df = load_data()
    xgb_model, iso_model, rf_model = load_models()
 
X = df.drop(columns=['Global_active_power','Global_reactive_power','Voltage','Global_intensity'])
y = df['Global_active_power']
split      = int(len(X) * 0.8)
X_test     = X.iloc[split:]
y_test     = y.iloc[split:]
y_pred_xgb = xgb_model.predict(X_test)
anomalies  = iso_model.predict(X_test)
 
def get_tariff(h):
    if 23 <= h or h < 7:  return 0.08
    elif 7 <= h < 17:     return 0.15
    else:                  return 0.25
 
def tariff_info(h):
    if 23 <= h or h < 7:  return ("Off-Peak", "🟢", "#10b981", "0.08")
    elif 7 <= h < 17:     return ("Standard",  "🟡", "#f59e0b", "0.15")
    else:                  return ("Peak",      "🔴", "#f43f5e", "0.25")
 
# ── Top header + navigation ───────────────────────────────────────────
st.markdown("""
<div style='padding:1.5rem 0 0.5rem 0;display:flex;align-items:center;gap:1rem;border-bottom:1px solid #1a1a2e;margin-bottom:1.5rem;'>
  <div>
    <div style='font-size:0.6rem;letter-spacing:0.2em;text-transform:uppercase;color:#6366f1;margin-bottom:0.2rem;'>⚡ Electricity Intelligence</div>
    <div style='font-family:Syne,sans-serif;font-size:1.8rem;font-weight:800;color:#fff;line-height:1;'>PowerSense</div>
  </div>
</div>
""", unsafe_allow_html=True)
 
page = st.radio("", [
    "⚡  Predict My Usage",
    "📈  Model Performance",
    "🚨  Anomaly Detection",
    "📊  Usage Patterns",
], horizontal=True, label_visibility="collapsed")
 
# ════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ════════════════════════════════════════════════════════════════════
if page == "⚡  Predict My Usage":
    st.markdown('<div class="page-title">Predict My Usage</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Estimate electricity consumption based on your household activity.</div>', unsafe_allow_html=True)
 
    col_form, col_results = st.columns([1.1, 0.9], gap="large")
 
    with col_form:
        # Period
        st.markdown('<div class="card"><div class="card-title">Forecast Period</div>', unsafe_allow_html=True)
        period = st.radio("", ["Daily", "Weekly", "Monthly"],
                          horizontal=True, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
 
        # Time
        st.markdown('<div class="card"><div class="card-title">Time & Season</div>', unsafe_allow_html=True)
        hour  = st.slider("Hour of day", 0, 23, 18, format="%d:00")
        month = st.select_slider("Month", options=list(range(1,13)),
                                 format_func=lambda x: pd.Timestamp(2010,x,1).strftime('%b'))
        st.markdown('</div>', unsafe_allow_html=True)
 
        # Appliances
        st.markdown('<div class="card"><div class="card-title">Active Appliances</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            kitchen = st.checkbox("🍳 Kitchen",  value=False)
            laundry = st.checkbox("🧺 Laundry",  value=False)
        with c2:
            ac      = st.checkbox("❄️ AC / Heater", value=False)
        st.markdown('</div>', unsafe_allow_html=True)
 
    # ── Compute ──────────────────────────────────────────────────────
    sub1 = 30.0 if kitchen else 0.0
    sub2 = 20.0 if laundry else 0.0
    sub3 = 18.0 if ac      else 0.0
    dow  = 5 if hour >= 18 else 2
    iwe  = 1 if dow >= 5   else 0
 
    inp = pd.DataFrame([{
        'Sub_metering_1': sub1, 'Sub_metering_2': sub2,
        'Sub_metering_3': sub3, 'hours': hour,
        'month': month, 'Day_of_week': dow, 'is_weekend': iwe
    }])
 
    kw            = xgb_model.predict(inp)[0]
    mult          = {"Daily":60*24,"Weekly":60*24*7,"Monthly":60*24*30}[period]
    pl            = {"Daily":"day","Weekly":"week","Monthly":"month"}[period]
    kwh           = (kw / 60) * mult
    tariff        = get_tariff(hour)
    cost          = kwh * tariff
    t_name, t_emoji, t_color, t_rate = tariff_info(hour)
    pattern       = rf_model.predict(inp)[0]
    p_emoji       = {"low":"🌙 Low","medium":"⚡ Medium","high":"🔥 High"}.get(pattern, pattern)
    p_color       = {"low":"#6366f1","medium":"#f59e0b","high":"#f43f5e"}.get(pattern,"#fff")
    gauge_pct     = min(kwh / {"Daily":50,"Weekly":350,"Monthly":1500}[period], 1.0)
 
    with col_results:
        # Main consumption card
        st.markdown(f"""
        <div class="card">
          <div class="card-title">Estimated Consumption</div>
          <div class="big-metric">
            <span class="val">{kwh:.1f}</span><span class="unit">kWh</span>
            <div class="sub">per {pl}</div>
          </div>
          <div class="pbar-wrap">
            <div class="pbar-label"><span>0</span><span>Usage level</span><span>Max</span></div>
            <div class="pbar-bg">
              <div class="pbar-fill" style="width:{gauge_pct*100:.0f}%;background:{p_color};"></div>
            </div>
          </div>
          <span class="badge badge-indigo">XGBoost model</span>
          <span class="badge" style="background:#13131f;color:{p_color};">{p_emoji}</span>
        </div>
        """, unsafe_allow_html=True)
 
        # Cost + tariff
        st.markdown(f"""
        <div class="card">
          <div class="card-title">Cost Estimate</div>
          <div style="display:flex;gap:1rem;align-items:flex-end;">
            <div>
              <div style="font-family:Syne,sans-serif;font-size:2.4rem;font-weight:800;color:#fff;line-height:1;">{cost:.2f}</div>
              <div style="font-size:0.72rem;color:#555570;margin-top:0.2rem;">estimated cost / {pl}</div>
            </div>
            <div style="margin-left:auto;text-align:right;">
              <div style="font-size:0.6rem;letter-spacing:0.14em;text-transform:uppercase;color:#555570;margin-bottom:0.2rem;">Tariff Zone</div>
              <div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:{t_color};">{t_emoji} {t_name}</div>
              <div style="font-size:0.72rem;color:#555570;">{t_rate} per kWh</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
 
        # Smart tips
        st.markdown('<div class="card-title" style="margin-top:0.5rem;">Smart Tips</div>', unsafe_allow_html=True)
 
        if pattern == "high":
            st.markdown('<div class="tip alert">🔥 High usage detected — consider shifting heavy appliances to off-peak hours (11pm–7am) to reduce cost.</div>', unsafe_allow_html=True)
        elif pattern == "medium":
            st.markdown('<div class="tip warn">⚡ Medium usage — you\'re within normal range. Turning off AC when not needed could help.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="tip good">🌙 Low usage — great! Your consumption is minimal right now.</div>', unsafe_allow_html=True)
 
        if t_name == "Peak":
            st.markdown('<div class="tip warn">🔴 You\'re in peak hours (5pm–11pm). Tariff is 3× higher than off-peak. Delay laundry if possible.</div>', unsafe_allow_html=True)
        elif t_name == "Off-Peak":
            st.markdown('<div class="tip good">🟢 Great time to run washing machine or dishwasher — cheapest tariff right now.</div>', unsafe_allow_html=True)
 
        if laundry and t_name == "Peak":
            st.markdown('<div class="tip alert">🧺 Running laundry during peak hours costs ~3× more. Try scheduling it after 11pm.</div>', unsafe_allow_html=True)
 
# ════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════
elif page == "📈  Model Performance":
    st.markdown('<div class="page-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">XGBoost regression model trained on 1.6M household readings.</div>', unsafe_allow_html=True)
 
    r2   = r2_score(y_test, y_pred_xgb)
    mae  = np.mean(np.abs(y_test.values - y_pred_xgb))
    rmse = np.sqrt(np.mean((y_test.values - y_pred_xgb)**2))
 
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, color in [
        (c1, "R² Score",     f"{r2:.3f}",   "#6366f1"),
        (c2, "MAE",          f"{mae:.3f} kW","#10b981"),
        (c3, "RMSE",         f"{rmse:.3f} kW","#f59e0b"),
        (c4, "Test Samples", "409,856",      "#a0a0c0"),
    ]:
        col.markdown(f"""
        <div class="card smetric">
          <div class="slabel">{label}</div>
          <div class="sval" style="color:{color};">{val}</div>
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    col_a, col_b = st.columns([1.3, 0.7], gap="large")
 
    with col_a:
        st.markdown('<div class="card"><div class="card-title">Actual vs Predicted (1,000 samples)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.patch.set_facecolor('#0f0f1c')
        ax.set_facecolor('#0f0f1c')
        ax.scatter(y_test[:1000], y_pred_xgb[:1000],
                   alpha=0.2, color='#6366f1', s=10, label='Predictions')
        ax.plot([0,6],[0,6], color='#f43f5e', linestyle='--',
                linewidth=1.2, label='Perfect prediction')
        ax.set_xlabel("Actual (kW)", color='#555570', fontsize=9)
        ax.set_ylabel("Predicted (kW)", color='#555570', fontsize=9)
        ax.tick_params(colors='#555570', labelsize=8)
        for s in ax.spines.values(): s.set_edgecolor('#1a1a2e')
        ax.legend(facecolor='#0f0f1c', edgecolor='#1a1a2e',
                  labelcolor='#a0a0c0', fontsize=8)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
 
    with col_b:
        st.markdown('<div class="card"><div class="card-title">What These Numbers Mean</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.82rem;color:#a0a0c0;line-height:1.7;">
          <b style="color:#6366f1;">R² = {r2:.3f}</b><br>
          Model explains <b style="color:#fff;">{r2*100:.1f}%</b> of all consumption variation.<br><br>
          <b style="color:#10b981;">MAE = {mae:.3f} kW</b><br>
          On average, predictions are off by only <b style="color:#fff;">{mae:.3f} kW</b>.<br><br>
          <b style="color:#f59e0b;">RMSE = {rmse:.3f} kW</b><br>
          Even large errors stay small — model handles spikes reasonably well.<br><br>
          <b style="color:#f43f5e;">Weak zone</b><br>
          High consumption (AC, oven) is hardest to predict — human behaviour is random.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <br>
        <span class="pill">Trained: Dec 2006 → Feb 2010</span>
        <span class="pill">Tested: Feb 2010 → Nov 2010</span>
        <span class="pill">Time-based split</span>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
 
# ════════════════════════════════════════════════════════════════════
# PAGE 3 — ANOMALY DETECTION
# ════════════════════════════════════════════════════════════════════
elif page == "🚨  Anomaly Detection":
    st.markdown('<div class="page-title">Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Isolation Forest model flags unusual consumption events in real time.</div>', unsafe_allow_html=True)
 
    n_anom = int(np.sum(anomalies == -1))
    pct    = n_anom / len(anomalies) * 100
 
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, color in [
        (c1, "Anomalies Found",  f"{n_anom:,}",          "#f43f5e"),
        (c2, "% of Readings",    f"{pct:.1f}%",           "#f59e0b"),
        (c3, "Normal Readings",  f"{len(anomalies)-n_anom:,}", "#10b981"),
        (c4, "Algorithm",        "Isolation Forest",       "#a0a0c0"),
    ]:
        col.markdown(f"""
        <div class="card smetric">
          <div class="slabel">{label}</div>
          <div class="sval" style="color:{color};font-size:1.1rem;">{val}</div>
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    X_test_plot            = X_test.copy()
    X_test_plot['anomaly'] = anomalies
    normal_idx    = X_test_plot[X_test_plot['anomaly'] ==  1].index
    anomalous_idx = X_test_plot[X_test_plot['anomaly'] == -1].index
 
    st.markdown('<div class="card"><div class="card-title">Consumption Timeline — Feb to Nov 2010</div>', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(12, 3.8))
    fig2.patch.set_facecolor('#0f0f1c')
    ax2.set_facecolor('#0f0f1c')
    ax2.fill_between(normal_idx, y_test[normal_idx],
                     color='#6366f1', alpha=0.15, linewidth=0)
    ax2.plot(normal_idx, y_test[normal_idx],
             color='#6366f1', alpha=0.4, linewidth=0.5, label='Normal')
    ax2.scatter(anomalous_idx, y_test[anomalous_idx],
                color='#f43f5e', s=5, label='Anomaly', zorder=5, alpha=0.8)
    ax2.set_xlabel("Time", color='#555570', fontsize=9)
    ax2.set_ylabel("Power (kW)", color='#555570', fontsize=9)
    ax2.tick_params(colors='#555570', labelsize=8)
    for s in ax2.spines.values(): s.set_edgecolor('#1a1a2e')
    ax2.legend(facecolor='#0f0f1c', edgecolor='#1a1a2e',
               labelcolor='#a0a0c0', fontsize=8)
    st.pyplot(fig2)
    st.markdown('</div>', unsafe_allow_html=True)
 
    st.markdown("""
    <span class="pill">🔴 Red dots = unusual consumption spikes</span>
    <span class="pill">Aug 2010 gap = household on holiday</span>
    <span class="pill">Most anomalies = sudden appliance peaks</span>
    <span class="pill">Contamination threshold = 1%</span>
    """, unsafe_allow_html=True)
 
# ════════════════════════════════════════════════════════════════════
# PAGE 4 — USAGE PATTERNS
# ════════════════════════════════════════════════════════════════════
elif page == "📊  Usage Patterns":
    st.markdown('<div class="page-title">Usage Patterns</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Random Forest classifier categorises every reading into Low / Medium / High usage.</div>', unsafe_allow_html=True)
 
    y_pred_clf = rf_model.predict(X_test)
    counts     = pd.Series(y_pred_clf).value_counts()
    total      = len(y_pred_clf)
 
    low_n = counts.get('low',0);   low_p = low_n/total*100
    med_n = counts.get('medium',0);med_p = med_n/total*100
    hig_n = counts.get('high',0);  hig_p = hig_n/total*100
 
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, color in [
        (c1, "Overall Accuracy", "91%",             "#6366f1"),
        (c2, "🌙 Low Usage",     f"{low_p:.1f}%",   "#6366f1"),
        (c3, "⚡ Medium Usage",  f"{med_p:.1f}%",   "#f59e0b"),
        (c4, "🔥 High Usage",    f"{hig_p:.1f}%",   "#f43f5e"),
    ]:
        col.markdown(f"""
        <div class="card smetric">
          <div class="slabel">{label}</div>
          <div class="sval" style="color:{color};">{val}</div>
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([0.65, 0.35], gap="large")
 
    with col_a:
        st.markdown('<div class="card"><div class="card-title">Distribution of Usage Categories</div>', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        fig3.patch.set_facecolor('#0f0f1c')
        ax3.set_facecolor('#0f0f1c')
        cats   = ['Low', 'Medium', 'High']
        vals   = [low_n, med_n, hig_n]
        colors = ['#6366f1', '#f59e0b', '#f43f5e']
        bars   = ax3.bar(cats, vals, color=colors, width=0.45,
                         edgecolor='#1a1a2e', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax3.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 2000,
                     f'{v/1000:.0f}K', ha='center',
                     color='#555570', fontsize=8)
        ax3.set_ylabel("Number of Readings", color='#555570', fontsize=9)
        ax3.tick_params(colors='#555570', labelsize=9)
        for s in ax3.spines.values(): s.set_edgecolor('#1a1a2e')
        st.pyplot(fig3)
        st.markdown('</div>', unsafe_allow_html=True)
 
    with col_b:
        st.markdown('<div class="card"><div class="card-title">Category Definitions</div>', unsafe_allow_html=True)
        for emoji, label, rng, note, color in [
            ("🌙","Low",    "< 1 kW",    "Nights, nobody home. Fridge, standby devices.",          "#6366f1"),
            ("⚡","Medium", "1 – 3 kW",  "Normal daily activity. Lights, TV, laptop.",              "#f59e0b"),
            ("🔥","High",   "> 3 kW",    "AC, oven, washing machine. Only 6% of all readings.",     "#f43f5e"),
        ]:
            st.markdown(f"""
            <div style="border-left:3px solid {color};padding:0.6rem 0.8rem;margin-bottom:0.7rem;border-radius:0 8px 8px 0;background:#0d0d1a;">
              <div style="font-weight:600;color:#fff;font-size:0.85rem;">{emoji} {label} <span style="color:{color};font-size:0.75rem;margin-left:0.4rem;">{rng}</span></div>
              <div style="font-size:0.75rem;color:#555570;margin-top:0.2rem;">{note}</div>
            </div>
            """, unsafe_allow_html=True)
 
        st.markdown("""
        <div style="margin-top:0.8rem;">
          <span class="pill">F1 Low = 0.94</span>
          <span class="pill">F1 Medium = 0.88</span>
          <span class="pill">F1 High = 0.74</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
 









import streamlit as st
st.set_page_config(page_title="RL Market Visualization", layout="wide")

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import threading
from streamlit_autorefresh import st_autorefresh
from rl_training import train_rl
from state import shared_state

st_autorefresh(interval=1000, key="refresh")

# Preserve scroll position
st.markdown("""
    <script>
    window.addEventListener('beforeunload', function () {
        localStorage.setItem('scrollTop', window.scrollY);
    });
    window.addEventListener('load', function () {
        const scrollTop = localStorage.getItem('scrollTop');
        if (scrollTop) window.scrollTo(0, parseInt(scrollTop));
    });
    </script>
""", unsafe_allow_html=True)

st.title("📈 Real-Time RL Trading Market Visualization")

if 'training_started' not in st.session_state:
    st.session_state.training_started = False

if not st.session_state.training_started:
    if st.button("🚀 Start RL Training"):
        threading.Thread(target=train_rl, daemon=True).start()
        st.session_state.training_started = True
        st.success("Training started!")

# === Load market state ===
with shared_state.lock:
    market = shared_state.market
    agents = shared_state.agents
    episode = shared_state.episode
    step_wealth = shared_state.step_wealth_history

if not market or not agents:
    st.info("Waiting for training data...")
    st.stop()

# === Utility: Unique color per agent ===
def get_agent_color_map(agent_names, colorscale='Turbo'):
    n = len(agent_names)
    positions = [i / max(n - 1, 1) for i in range(n)]
    colors = sample_colorscale(colorscale, positions)
    return {name: color for name, color in zip(agent_names, colors)}

# === Wealth Share Over Time ===
st.subheader("📉 Portfolio Share of Each Agent Over Time")

if step_wealth:
    
    # Prepare the data for the line chart of portfolio share per agent over steps
    df_steps = pd.DataFrame([
        {'Episode': e, 'Step': s, 'Agent': a, 'WealthShare': w}
        for e, s, a, w in step_wealth  # Unpack the 4 values directly
    ])

    fig = go.Figure()

    # Plot portfolio share for each agent (line chart)
    for agent in df_steps['Agent'].unique():
        agent_data = df_steps[df_steps['Agent'] == agent]
        fig.add_trace(go.Scatter(x=agent_data['Step'], y=agent_data['WealthShare'], mode='lines', name=agent))

    # Add vertical dashed lines based on Episode
    for episode in set(df_steps['Episode']):
        episode_steps = df_steps[df_steps['Episode'] == episode]['Step']
        if len(episode_steps) > 0:
            fig.add_vline(x=episode_steps.max(), line_dash="dash", line_color="gray", name=f"Episode {episode}")

    fig.update_layout(
        title="Portfolio Share of Each Agent Over Time",
        xaxis_title="Step",
        yaxis_title="Portfolio Share",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Waiting for step-level portfolio tracking...")


# === Combined Wealth + Portfolio ===
st.subheader("📊 Agent Wealth Share + Portfolio Allocation")

wealths = [ag.value() for ag in agents]
names = [ag.name for ag in agents]
total_wealth = sum(wealths)

stack_data = {asset: [] for asset in market.assets}
stack_data["Cash"] = []

for ag in agents:
    val = ag.value()
    share = val / total_wealth
    for asset in market.assets:
        asset_val = ag.get_asset_quantity(asset) * market.prices[asset]
        stack_data[asset].append((asset_val / val) * share)
    stack_data["Cash"].append((ag.money / val) * share)

fig_combined = go.Figure()
for k, v in stack_data.items():
    fig_combined.add_trace(go.Bar(name=k, x=names, y=v))

fig_combined.update_layout(
    barmode='stack',
    title="Agent Wealth Share with Portfolio Breakdown",
    yaxis_title="Fraction of Total System Wealth",
    xaxis_title="Agent",
    height=450,
    legend=dict(orientation="h"),
    margin=dict(l=40, r=40, t=60, b=40)
)

st.plotly_chart(fig_combined, use_container_width=True)

# === Order Book Plots ===
st.subheader("📘 Order Books (Buy ↑ / Sell ↓, VWAP-Centered)")

for asset_idx, asset in enumerate(market.assets):
    st.markdown(f"### {asset} Order Book")
    entries = market.get_order_book_entries(asset_idx)
    if not entries:
        st.info("No orders yet.")
        continue

    df = pd.DataFrame(entries)
    vwap = market.prices[asset]
    agent_names = df['agent'].unique().tolist()
    agent_colors = get_agent_color_map(agent_names)

    price_spread = df['price'].max() - df['price'].min()
    bar_width = max(0.01, price_spread / 50)

    fig = go.Figure()

    for agent in agent_names:
        color = agent_colors[agent]
        df_buy = df[(df['agent'] == agent) & (df['side'] == 'buy')]
        df_sell = df[(df['agent'] == agent) & (df['side'] == 'sell')]

        if not df_buy.empty:
            fig.add_trace(go.Bar(
                x=df_buy['price'], y=df_buy['qty'],
                width=[bar_width] * len(df_buy),
                name=f"{agent} (Buy)", marker_color=color,
                opacity=0.7
            ))
        if not df_sell.empty:
            fig.add_trace(go.Bar(
                x=df_sell['price'], y=-df_sell['qty'],
                width=[bar_width] * len(df_sell),
                name=f"{agent} (Sell)", marker_color=color,
                opacity=0.5
            ))

    fig.add_vline(x=vwap, line_dash="dash", line_color="black", name="VWAP")

    fig.update_layout(
        barmode='relative',
        title=f"{asset} Order Book",
        xaxis_title="Price",
        yaxis_title="Quantity",
        xaxis_range=[vwap - price_spread, vwap + price_spread],
        yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='gray'),
        height=450,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h")
    )

    st.plotly_chart(fig, use_container_width=True)


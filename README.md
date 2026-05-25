# U.S. Semiconductor Supply Chain Digital Twin

**UC Berkeley INDENG 243 Analytics Lab — Spring 2026**

An interactive simulation dashboard modeling the U.S. semiconductor import supply chain, combining stochastic throughput simulation, constrained optimization, and 15 years of historical trade data.

---

## Features

- **Live SimPy simulation** — stochastic shock & recovery timeline for supply disruptions
- **SLSQP optimizer** — constrained re-allocation of supplier shares under shock scenarios
- **Historical trade context** — 2010–2025 UN Comtrade data with HHI concentration metrics
- **Scenario library** — curated shock scenarios with historical precedents (COVID, trade wars, natural disasters)
- **Dash web app** — fully interactive, runs locally in your browser

---

## Quick Start

### 1. Install dependencies

```bash
pip install dash plotly pandas numpy simpy scipy
```

### 2. Run the app

```bash
python app_digital_twin.py
```

Then open [http://127.0.0.1:8050](http://127.0.0.1:8050) in your browser.

---

## Data Files

| File | Description |
|---|---|
| `TradeData_2010_2025.csv` | UN Comtrade monthly semiconductor import data (HS 8542), USA, 2010–2025 |
| `master_panel_v2.csv` | Annual panel: HHI, port stress indices, supplier shares, port TEU volumes |
| `suppliers.csv` | Supplier baseline shares, risk scores, and allocation caps |
| `ports.csv` | Port capacities, current shares, and risk scores |
| `transport_costs.csv` | Cost-per-TEU matrix by supplier–port pair |

---

## Methods

- **Simulation**: SimPy discrete-event simulation with configurable shock magnitude and recovery half-life
- **Optimization**: `scipy.optimize.minimize` (SLSQP) minimizing cost + risk subject to share constraints
- **Concentration**: Herfindahl–Hirschman Index (HHI) computed from annual supplier shares
- **Visualization**: Plotly Dash with real-time callback updates

---

## Data Source

Trade data from [UN Comtrade](https://comtradeplus.un.org/) — HS commodity code **8542** (Electronic Integrated Circuits).

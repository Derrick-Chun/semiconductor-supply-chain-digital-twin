"""
app_digital_twin.py
U.S. Semiconductor Supply Chain Digital Twin
UC Berkeley INDENG 243 Analytics Lab, Spring 2026

Combines:
  - Live SimPy stochastic throughput simulation (shock + recovery timeline)
  - SLSQP constrained re-allocation optimizer
  - Historical trade context (2010-2025) with concentration metrics
  - Curated shock scenario library with historical precedents
  - Trade-calibrated baselines and volatility-adjusted risk scores

Run:  python app_digital_twin.py
"""

import os
import numpy as np
import pandas as pd
import simpy
from scipy.optimize import minimize

import dash
from dash import dcc, html, Input, Output, State, ctx, ALL, no_update
import plotly.graph_objects as go

# =============================================================================
# PALETTE
# =============================================================================
BG          = "#f0ede8"
CARD        = "#ffffff"
DARK        = "#0d1b2a"
DARK2       = "#1a1a1a"
TEXT        = "#1a1a1a"
MUTED       = "#888888"
MUTED2      = "#aabbcc"
ACCENT      = "#c9a84c"
RED         = "#e05c5c"
GREEN       = "#5aab6e"
BLUE        = "#4a7ba8"
BORDER      = "#e8e4de"
GRID        = "#f2efe9"
SHADOW      = "0 2px 12px rgba(0,0,0,0.07)"
RADIUS      = "16px"
MONO        = "'Courier New', monospace"
SANS        = "'Inter','Helvetica Neue',sans-serif"

SUPPLIER_COLORS = {
    "Malaysia":  ACCENT,
    "Korea":     BLUE,
    "China":     RED,
    "Japan":     GREEN,
    "Hong Kong": "#9b6ba8",
    "Singapore": "#d49062",
}

# =============================================================================
# CONSTANTS / CONFIG
# =============================================================================
SUPPLIERS_ORDER = ["Malaysia", "Korea", "China", "Japan", "Hong Kong"]
PORTS_FULL      = ["Port of Los Angeles", "Port of New York & New Jersey"]
PORT_LABEL      = {"Port of Los Angeles": "Port of LA",
                   "Port of New York & New Jersey": "Port of NY/NJ"}

TRADE_PARTNER_MAP = {
    "China":                 "China",
    "China, Hong Kong SAR":  "Hong Kong",
    "Japan":                 "Japan",
    "Rep. of Korea":         "Korea",
    "Malaysia":              "Malaysia",
    "Singapore":             "Singapore",
}

# Preset shock scenarios — for Digital Twin and Shock Library
SHOCK_PRESETS = {
    "normal": {
        "label": "Normal Operations",
        "mal": 0, "la": 0,
        "icon": "●",
        "color": GREEN,
        "desc": "No active disruption. Baseline reference for comparison.",
        "precedent": "Pre-pandemic steady state (2016-2019).",
        "impact": "No measurable throughput loss.",
    },
    "malaysia_factory": {
        "label": "Malaysia Factory Disruption",
        "mal": 40, "la": 0,
        "icon": "▲",
        "color": ACCENT,
        "desc": "40% output loss at Malaysian packaging & assembly facilities.",
        "precedent": "June 2021 COVID wave idled Intel, Infineon, STMicro, and ASE plants in Penang and Muar for several weeks; auto-grade MCU shortages hit Ford, Toyota, GM within a month.",
        "impact": "Malaysia supplies ~54% of U.S. semiconductor imports; a 40% cut removes roughly 22% of total monthly volume.",
    },
    "la_strike": {
        "label": "Port of LA Labor Strike",
        "mal": 0, "la": 60,
        "icon": "⚠",
        "color": RED,
        "desc": "60% capacity loss at San Pedro Bay; partial diversion to NY/NJ.",
        "precedent": "ILWU contract negotiations (2022-2023) caused intermittent slowdowns for 13 months; 100+ ships queued at anchor during the worst weeks.",
        "impact": "LA handles ~53% of TEU throughput for the two ports in scope; a 60% strike compresses 32% of total capacity.",
    },
    "taiwan_strait": {
        "label": "Taiwan Strait Escalation",
        "mal": 20, "la": 15,
        "icon": "◆",
        "color": BLUE,
        "desc": "Pacific routing disruption affecting all Asia-origin flows.",
        "precedent": "August 2022 PLA exercises after the Pelosi visit rerouted 50+ container vessels and added 6-12 days to Asia-to-U.S. West Coast transit; Lloyd's List recorded a ~17% spike in Asia-U.S. spot rates.",
        "impact": "All five suppliers lose 20-25% effective throughput for 30-45 days.",
    },
    "seismic": {
        "label": "Seismic Event (Kaohsiung / Penang)",
        "mal": 65, "la": 0,
        "icon": "◉",
        "color": "#9b6ba8",
        "desc": "Major earthquake idles Malaysian and regional assembly facilities.",
        "precedent": "February 2016 Kaohsiung M6.4 and April 2024 Hualien M7.4 events caused multi-week pauses at TSMC, UMC and Malaysian packaging lines; inventory impact lasted 60-90 days.",
        "impact": "Near-total loss of Malaysia throughput for 3-6 weeks.",
    },
    "regional_shutdown": {
        "label": "Pandemic-Style Regional Shutdown",
        "mal": 50, "la": 40,
        "icon": "☣",
        "color": DARK,
        "desc": "Simultaneous supplier production cuts and U.S. port congestion.",
        "precedent": "2020-2021 COVID cascade: Asia factory closures coincided with L.A./LB backlog of 100+ ships; DRAM and MCU lead times spiked to 40+ weeks.",
        "impact": "Combined supply and demand-side stress; full recovery took ~18 months.",
    },
}

# =============================================================================
# DATA LOADING
# =============================================================================
def _find_file(filename):
    candidates = [
        filename,
        os.path.join("data", filename),
        os.path.join("..", filename),
        os.path.join("..", "data", filename),
    ]
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates += [
            os.path.join(here, filename),
            os.path.join(here, "data", filename),
            os.path.join(here, "..", filename),
            os.path.join(here, "..", "data", filename),
            os.path.join(here, "..", "Data", filename),
            os.path.join(here, "..", "Data", "latest data", filename),
        ]
    except NameError:
        pass
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


def _load_csvs():
    sup_p   = _find_file("suppliers.csv")
    prt_p   = _find_file("ports.csv")
    cost_p  = _find_file("transport_costs.csv")
    trade_p = _find_file("TradeData_2010_2025.csv")
    panel_p = _find_file("master_panel_v2.csv")

    missing = []
    if not sup_p:   missing.append("suppliers.csv")
    if not prt_p:   missing.append("ports.csv")
    if not cost_p:  missing.append("transport_costs.csv")
    if missing:
        raise FileNotFoundError(f"Required data files not found: {missing}")

    sup   = pd.read_csv(sup_p)
    prt   = pd.read_csv(prt_p)
    cost  = pd.read_csv(cost_p)
    trade = pd.read_csv(trade_p) if trade_p else None
    panel = pd.read_csv(panel_p) if panel_p else None

    return sup, prt, cost, trade, panel


SUPPLIERS_DF, PORTS_DF, TRANSPORT_DF, TRADE_DF, PANEL_DF = _load_csvs()
TRADE_AVAILABLE = TRADE_DF is not None
PANEL_AVAILABLE = PANEL_DF is not None


# =============================================================================
# TRADE DATA HELPERS
# =============================================================================
def prepare_trade_data(trade_df):
    """Clean + aggregate monthly and annual supplier shares."""
    df = trade_df.copy()
    df = df[df["partnerDesc"].isin(TRADE_PARTNER_MAP.keys())].copy()
    df["supplier"]     = df["partnerDesc"].map(TRADE_PARTNER_MAP)
    df["primaryValue"] = pd.to_numeric(df["primaryValue"], errors="coerce").fillna(0.0)
    df["netWgt"]       = pd.to_numeric(df["netWgt"], errors="coerce").fillna(0.0)
    df["month_start"]  = pd.to_datetime(dict(year=df["refYear"], month=df["refMonth"], day=1))

    monthly_sup = (
        df.groupby(["month_start", "refYear", "supplier"], as_index=False)
          .agg(primaryValue=("primaryValue", "sum"), netWgt=("netWgt", "sum"))
          .sort_values(["month_start", "supplier"])
    )
    monthly_tot = (
        monthly_sup.groupby(["month_start", "refYear"], as_index=False)
                   .agg(total_primary_value=("primaryValue", "sum"),
                        total_net_wgt=("netWgt", "sum"))
                   .sort_values("month_start")
    )
    monthly_sup = monthly_sup.merge(
        monthly_tot[["month_start", "total_primary_value"]],
        on="month_start", how="left")
    monthly_sup["share"] = (monthly_sup["primaryValue"] /
                            monthly_sup["total_primary_value"]).fillna(0.0)

    annual_sup = (monthly_sup.groupby(["refYear", "supplier"], as_index=False)
                             .agg(annual_primary_value=("primaryValue", "sum"),
                                  annual_net_wgt=("netWgt", "sum")))
    ann_tot = (annual_sup.groupby("refYear", as_index=False)
                         .agg(total_annual_primary_value=("annual_primary_value", "sum")))
    annual_sup = annual_sup.merge(ann_tot, on="refYear", how="left")
    annual_sup["annual_share"] = (annual_sup["annual_primary_value"] /
                                  annual_sup["total_annual_primary_value"]).fillna(0.0)

    return monthly_sup, monthly_tot, annual_sup


if TRADE_AVAILABLE:
    MONTHLY_SUP_DF, MONTHLY_TOT_DF, ANNUAL_SUP_DF = prepare_trade_data(TRADE_DF)
else:
    MONTHLY_SUP_DF = MONTHLY_TOT_DF = ANNUAL_SUP_DF = None


# =============================================================================
# BASELINE CONSTRUCTION (from CSVs, not hardcoded)
# =============================================================================
def _build_base():
    supplier_shares = {
        row["supplier"]: float(row["current_share"])
        for _, row in SUPPLIERS_DF.iterrows()
    }
    for s in SUPPLIERS_ORDER:
        supplier_shares.setdefault(s, 0.0)
    tot_s = sum(supplier_shares.values())
    if tot_s > 0:
        supplier_shares = {k: v / tot_s for k, v in supplier_shares.items()}

    port_shares    = {row["port"]: float(row["current_share"])
                      for _, row in PORTS_DF.iterrows()}
    port_caps      = {row["port"]: float(row["capacity_teu"])
                      for _, row in PORTS_DF.iterrows()}
    supplier_risk  = {row["supplier"]: float(row["risk_score"])
                      for _, row in SUPPLIERS_DF.iterrows()}
    port_risk      = {row["port"]: float(row["port_risk"])
                      for _, row in PORTS_DF.iterrows()}
    transport_cost = {(r["supplier"], r["port"]): float(r["cost_per_teu"])
                      for _, r in TRANSPORT_DF.iterrows()}

    return {
        "supplier_shares":  supplier_shares,
        "port_shares":      port_shares,
        "port_capacities":  port_caps,
        "supplier_risk":    supplier_risk,
        "port_risk":        port_risk,
        "transport_cost":   transport_cost,
    }


BASE = _build_base()
SUPPLIER_RISK   = BASE["supplier_risk"]
PORT_RISK       = BASE["port_risk"]
TRANSPORT_COSTS = BASE["transport_cost"]


def _cost_df():
    return TRANSPORT_DF.rename(columns={}).copy()


def _build_dfs(sup_shares=None, prt_shares=None):
    if sup_shares is None: sup_shares = BASE["supplier_shares"]
    if prt_shares is None: prt_shares = BASE["port_shares"]
    sup_df = pd.DataFrame([
        {"supplier": s,
         "current_share": sup_shares.get(s, 0.0),
         "risk_score":    SUPPLIER_RISK.get(s, 0.5),
         "max_share_default": min(max(sup_shares.get(s, 0.0) * 1.5, 0.20), 0.80)}
        for s in SUPPLIERS_ORDER
    ])
    prt_df = pd.DataFrame([
        {"port": p,
         "current_share": prt_shares.get(p, 0.0),
         "port_risk":     PORT_RISK.get(p, 0.3),
         "capacity_teu":  BASE["port_capacities"].get(p, 8_000_000)}
        for p in PORTS_FULL
    ])
    return sup_df, prt_df


def _cost_matrix(sup_df, prt_df):
    sn = sup_df["supplier"].tolist()
    pn = prt_df["port"].tolist()
    M = np.zeros((len(sn), len(pn)))
    for i, s in enumerate(sn):
        for j, p in enumerate(pn):
            M[i, j] = TRANSPORT_COSTS.get((s, p), 110.0)
    return M


_BASE_SUP_DF, _BASE_PRT_DF = _build_dfs()
_BASE_CM = _cost_matrix(_BASE_SUP_DF, _BASE_PRT_DF)


# =============================================================================
# SIMULATION (SimPy — stochastic throughput over a 180-day horizon)
# =============================================================================
def run_simulation(malaysia_pct, la_pct, sim_days=180, shock_day=30, seed=42):
    """
    Day-by-day simulation. A disruption lands on `shock_day` and compounds
    daily queues at the ports. Returns baseline + shocked time series.
    """
    np.random.seed(seed)
    sup_shares = BASE["supplier_shares"]
    port_caps  = BASE["port_capacities"]
    daily_total = 0.90 * (port_caps[PORTS_FULL[0]] + port_caps[PORTS_FULL[1]]) / 365
    la_cap      = port_caps[PORTS_FULL[0]] / 365
    ny_cap      = port_caps[PORTS_FULL[1]] / 365
    la_route    = {"Malaysia": 0.65, "Korea": 0.60, "Japan": 0.65,
                   "China": 0.55, "Hong Kong": 0.55, "Singapore": 0.60}

    laq = nyq = 0.0
    la_t, ny_t, laq_h = [], [], []
    sup_daily = {s: [] for s in sup_shares}
    env = simpy.Environment()

    def loop(env):
        nonlocal laq, nyq
        for day in range(sim_days):
            yield env.timeout(1)
            shocked = day >= shock_day
            mf = (1 - malaysia_pct / 100) if shocked else 1.0
            lf = (1 - la_pct / 100)       if shocked else 1.0
            for s, share in sup_shares.items():
                f   = mf if s == "Malaysia" else 1.0
                out = daily_total * share * f * np.random.uniform(0.97, 1.03)
                laq += out * la_route.get(s, 0.6)
                nyq += out * (1 - la_route.get(s, 0.6))
                sup_daily[s].append(out)
            pla = min(laq, la_cap * lf); laq = max(laq - pla, 0.0)
            pny = min(nyq, ny_cap);      nyq = max(nyq - pny, 0.0)
            la_t.append(pla); ny_t.append(pny); laq_h.append(laq)

    env.process(loop(env))
    env.run()

    la_a, ny_a = np.array(la_t), np.array(ny_t)
    pre  = la_a[:shock_day].mean() + ny_a[:shock_day].mean() if shock_day > 0 else daily_total
    post = la_a[shock_day:].mean() + ny_a[shock_day:].mean()
    loss = float(max(0, (pre - post) / max(pre, 1e-9) * 100))

    rec = sim_days - shock_day
    for i, q in enumerate(laq_h[shock_day:]):
        if q < la_cap * 0.05:
            rec = i
            break

    po = {s: np.mean(sup_daily[s][shock_day:]) for s in sup_shares}
    tot = sum(po.values())
    pss = {k: v / tot for k, v in po.items()} if tot > 0 else dict(sup_shares)
    pla_, pny_ = la_a[shock_day:].mean(), ny_a[shock_day:].mean()
    tp = pla_ + pny_
    pps = {PORTS_FULL[0]: (pla_ / tp) if tp > 0 else BASE["port_shares"][PORTS_FULL[0]],
           PORTS_FULL[1]: (pny_ / tp) if tp > 0 else BASE["port_shares"][PORTS_FULL[1]]}
    return {
        "post_sup": pss, "post_prt": pps,
        "loss_pct": loss,
        "max_q":    float(max(laq_h[shock_day:])) if len(laq_h) > shock_day else 0,
        "rec_days": rec,
        "shock_day": shock_day,
        "days": list(range(sim_days)),
        "la_t": la_t, "ny_t": ny_t,
        "total_t": (la_a + ny_a).tolist(),
    }


# =============================================================================
# OPTIMIZER (SLSQP)
# =============================================================================
def _norm(arr):
    a = np.array(arr, dtype=float)
    s = a.sum()
    return a / s if s > 0 else np.ones_like(a) / len(a)


def _comps(X, p):
    ss = X.sum(1); ps = X.sum(0)
    ac = float(np.sum(X * p["cm"]))
    cr = max(p["cmax"] - p["cmin"], 1e-9)
    return {
        "ss": ss, "ps": ps,
        "sr": float(np.dot(ss, p["sr"])),
        "pr": float(np.dot(ps, p["pr"])),
        "sc": float(np.sum(ss ** 2)),
        "pc": float(np.sum(ps ** 2)),
        "stab": float(np.sum((ss - p["css"]) ** 2) + np.sum((ps - p["cps"]) ** 2)),
        "cost": ac, "ann": ac * p["dem"],
        "cn": (ac - p["cmin"]) / cr,
    }


def _obj(xf, p):
    c = _comps(xf.reshape(p["ns"], p["np"]), p)
    res = (p["wsr"] * c["sr"] + p["wpr"] * c["pr"] + p["wsc"] * c["sc"]
           + p["wpc"] * c["pc"] + p["wst"] * c["stab"])
    return p["wc"] * c["cn"] + p["wr"] * res


def resilience_score(c):
    raw = (0.35 * c["sr"] + 0.20 * c["pr"] + 0.20 * c["sc"]
           + 0.10 * c["pc"] + 0.15 * c["stab"])
    return round(100 * max(0.0, 1.0 - raw), 1)


def score_state(ss, ps, sr, pr, ref_s, ref_p):
    return resilience_score({
        "sr":   float(np.dot(ss, sr)),
        "pr":   float(np.dot(ps, pr)),
        "sc":   float(np.sum(ss ** 2)),
        "pc":   float(np.sum(ps ** 2)),
        "stab": float(np.sum((ss - ref_s) ** 2) + np.sum((ps - ref_p) ** 2)),
    })


def run_optimizer(sup_df, prt_df, cm,
                  dem=8_000_000, cap=0.70, rp=65,
                  weights=None, effective_sup_risk=None,
                  cap_mult=1.0):
    sn = sup_df["supplier"].tolist(); pn = prt_df["port"].tolist()
    ns, np_ = len(sn), len(pn)
    css = _norm(sup_df["current_share"].values)
    cps = _norm(prt_df["current_share"].values)
    adj = prt_df["capacity_teu"].values.astype(float) * float(cap_mult)
    if adj.sum() < dem:
        adj = adj / adj.sum() * dem * 1.1
    wr = rp / 100

    w = weights or {"wsr": 0.35, "wpr": 0.20, "wsc": 0.20, "wpc": 0.10, "wst": 0.15}
    sr = (np.array(effective_sup_risk, dtype=float)
          if effective_sup_risk is not None
          else sup_df["risk_score"].values.astype(float))

    p = {
        "ns": ns, "np": np_,
        "sr": sr,
        "pr": prt_df["port_risk"].values.astype(float),
        "cm": cm,
        "css": css, "cps": cps,
        "dem": dem,
        "cmin": float(np.min(cm)), "cmax": float(np.max(cm)),
        "wc": 1 - wr, "wr": wr,
        "wsr": w["wsr"], "wpr": w["wpr"], "wsc": w["wsc"],
        "wpc": w["wpc"], "wst": w["wst"],
    }

    si = _norm(np.minimum(css, cap))
    pi = _norm(np.minimum(cps, adj / dem))
    x0 = np.outer(si, pi).flatten()
    cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    for i in range(ns):
        cons.append({"type": "ineq",
                     "fun": lambda x, i=i: cap - x.reshape(ns, np_)[i].sum()})
    for j in range(np_):
        cons.append({"type": "ineq",
                     "fun": lambda x, j=j: adj[j] - x.reshape(ns, np_)[:, j].sum() * dem})
    res = minimize(_obj, x0, args=(p,), method="SLSQP",
                   bounds=[(0., 1.)] * (ns * np_), constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-9})
    xo = res.x.reshape(ns, np_)
    c = _comps(xo, p)
    return {"ss": c["ss"], "ps": c["ps"], "comps": c, "x": xo,
            "sn": sn, "pn": pn, "ac_per_teu": c["cost"], "annual_cost": c["ann"]}


# =============================================================================
# STRUCTURAL RISK HELPERS
# =============================================================================
def risk_band(value, low_cut, high_cut):
    if value >= high_cut: return "High"
    if value >= low_cut:  return "Medium"
    return "Low"


def overall_structural_risk(sup_hhi, mal_share, port_hhi):
    score = 0
    if sup_hhi   >= 0.30: score += 2
    elif sup_hhi >= 0.22: score += 1
    if mal_share >= 0.50: score += 2
    elif mal_share >= 0.30: score += 1
    if port_hhi  >= 0.55: score += 2
    elif port_hhi >= 0.50: score += 1
    if score >= 5: return "High"
    if score >= 3: return "Medium"
    return "Low"


def strategy_label(pref):
    if pref <= 30: return "Cost-focused"
    if pref >= 70: return "Resilience-focused"
    return "Balanced"


# =============================================================================
# TRADE-BASED BASELINE HELPERS
# =============================================================================
def get_latest_full_year():
    return int(ANNUAL_SUP_DF["refYear"].max()) if TRADE_AVAILABLE else None


def get_latest_12m_share(supplier_order):
    if not TRADE_AVAILABLE: return None
    latest = MONTHLY_SUP_DF["month_start"].max()
    start  = latest - pd.DateOffset(months=11)
    win = MONTHLY_SUP_DF[(MONTHLY_SUP_DF["month_start"] >= start) &
                         (MONTHLY_SUP_DF["month_start"] <= latest)]
    agg = win.groupby("supplier", as_index=False).agg(pv=("primaryValue", "sum"))
    base = pd.DataFrame({"supplier": supplier_order})
    agg = base.merge(agg, on="supplier", how="left").fillna(0.0)
    return pd.Series(_norm(agg["pv"].values), index=agg["supplier"].tolist())


def get_year_share(year, supplier_order):
    if not TRADE_AVAILABLE: return None
    yr = ANNUAL_SUP_DF[ANNUAL_SUP_DF["refYear"] == year].copy()
    base = pd.DataFrame({"supplier": supplier_order})
    yr = base.merge(yr[["supplier", "annual_share"]], on="supplier", how="left").fillna(0.0)
    return pd.Series(_norm(yr["annual_share"].values), index=yr["supplier"].tolist())


def compute_trade_risk_adjustment(vol_w, trend_w, window_months=12):
    """Volatility + trend-based historical risk signal in [0,1]."""
    if not TRADE_AVAILABLE:
        return pd.Series(np.zeros(len(SUPPLIERS_ORDER)), index=SUPPLIERS_ORDER)
    pivot = (MONTHLY_SUP_DF.pivot(index="month_start", columns="supplier", values="share")
                           .reindex(columns=SUPPLIERS_ORDER).fillna(0.0).sort_index())
    if len(pivot) < 2:
        return pd.Series(np.zeros(len(SUPPLIERS_ORDER)), index=SUPPLIERS_ORDER)
    use = min(window_months, len(pivot))
    recent = pivot.tail(use)
    vol = recent.std(axis=0).fillna(0.0)
    if len(recent) >= 2:
        half = max(1, len(recent) // 2)
        first = recent.head(half).mean(axis=0)
        last  = recent.tail(half).mean(axis=0)
        trend_drop = (first - last).clip(lower=0.0)
    else:
        trend_drop = pd.Series(np.zeros(len(SUPPLIERS_ORDER)), index=SUPPLIERS_ORDER)

    def mm(s):
        s = s.copy()
        if (s.max() - s.min()) <= 1e-9:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.min()) / (s.max() - s.min())

    adj = (vol_w * mm(vol) + trend_w * mm(trend_drop)).clip(0, 1)
    return adj.reindex(SUPPLIERS_ORDER).fillna(0.0)


def compute_annual_hhi():
    if not TRADE_AVAILABLE: return None
    hhi = (ANNUAL_SUP_DF.groupby("refYear")
                         .apply(lambda g: float(np.sum(g["annual_share"].values ** 2)))
                         .reset_index(name="hhi").sort_values("refYear"))
    return hhi


def compute_monthly_hhi():
    if not TRADE_AVAILABLE: return None
    hhi = (MONTHLY_SUP_DF.groupby("month_start")
                          .apply(lambda g: float(np.sum(g["share"].values ** 2)))
                          .reset_index(name="hhi").sort_values("month_start"))
    return hhi


def compute_supplier_volatility():
    """Rolling 12-month std of supplier share, most recent window."""
    if not TRADE_AVAILABLE:
        return pd.DataFrame({"supplier": SUPPLIERS_ORDER,
                             "volatility_12m": [0.0] * len(SUPPLIERS_ORDER)})
    pivot = (MONTHLY_SUP_DF.pivot(index="month_start", columns="supplier", values="share")
                            .sort_index())
    vol = pivot.rolling(12, min_periods=6).std().iloc[-1].fillna(0.0)
    df = pd.DataFrame({"supplier": vol.index, "volatility_12m": vol.values})
    df = df[df["supplier"].isin(SUPPLIERS_ORDER + ["Singapore"])]
    return df.sort_values("volatility_12m", ascending=False).reset_index(drop=True)


# =============================================================================
# PLOTLY CHART BUILDERS
# =============================================================================
def _axes_style():
    return dict(
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(family=SANS),
        margin=dict(l=48, r=24, t=12, b=40),
        xaxis=dict(showgrid=False, zeroline=False,
                   tickfont=dict(size=11, color=MUTED, family=SANS),
                   title=dict(font=dict(size=12, color=MUTED, family=SANS))),
        yaxis=dict(gridcolor=GRID, showgrid=True, zeroline=False,
                   tickfont=dict(size=11, color=MUTED, family=SANS),
                   title=dict(font=dict(size=12, color=MUTED, family=SANS))),
    )


def plot_total_imports(metric="primaryValue"):
    y = "total_primary_value" if metric == "primaryValue" else "total_net_wgt"
    ylab = "Import Value (USD)" if metric == "primaryValue" else "Net Weight (kg)"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=MONTHLY_TOT_DF["month_start"], y=MONTHLY_TOT_DF[y],
        line=dict(color=ACCENT, width=2),
        fill="tozeroy", fillcolor="rgba(201,168,76,0.10)",
        hovertemplate="%{x|%b %Y}<br>" + ylab + ": %{y:,.0f}<extra></extra>",
        name="Total",
    ))
    st = _axes_style()
    st["xaxis"]["title"] = dict(text="Month", font=dict(size=12, color=MUTED, family=SANS))
    st["yaxis"]["title"] = dict(text=ylab, font=dict(size=12, color=MUTED, family=SANS))
    fig.update_layout(**st, showlegend=False)
    return fig


def plot_supplier_stacked_area(suppliers_to_show):
    pivot = (MONTHLY_SUP_DF.pivot(index="month_start", columns="supplier", values="share")
                           .fillna(0.0).sort_index())
    cols = [s for s in suppliers_to_show if s in pivot.columns]
    fig = go.Figure()
    for s in cols:
        fig.add_trace(go.Scatter(
            x=pivot.index, y=pivot[s] * 100, name=s, stackgroup="one",
            line=dict(width=0.5, color=SUPPLIER_COLORS.get(s, "#888")),
            fillcolor=SUPPLIER_COLORS.get(s, "#888"),
            hovertemplate="%{x|%b %Y}<br>" + s + ": %{y:.1f}%<extra></extra>",
        ))
    st = _axes_style()
    st["xaxis"]["title"] = dict(text="Month", font=dict(size=12, color=MUTED, family=SANS))
    st["yaxis"]["title"] = dict(text="Share (%)", font=dict(size=12, color=MUTED, family=SANS))
    fig.update_layout(**st, legend=dict(orientation="h", y=-0.18, x=0,
                                        font=dict(size=11, color=TEXT, family=SANS),
                                        bgcolor="rgba(0,0,0,0)"))
    return fig


def plot_annual_supplier_share(suppliers_to_show):
    fig = go.Figure()
    for s in suppliers_to_show:
        sub = ANNUAL_SUP_DF[ANNUAL_SUP_DF["supplier"] == s].sort_values("refYear")
        if sub.empty: continue
        fig.add_trace(go.Scatter(
            x=sub["refYear"], y=sub["annual_share"] * 100,
            name=s, mode="lines+markers",
            line=dict(color=SUPPLIER_COLORS.get(s, "#888"), width=2),
            marker=dict(size=6),
            hovertemplate="%{x}<br>" + s + ": %{y:.1f}%<extra></extra>",
        ))
    st = _axes_style()
    st["xaxis"]["title"] = dict(text="Year", font=dict(size=12, color=MUTED, family=SANS))
    st["yaxis"]["title"] = dict(text="Annual Share (%)", font=dict(size=12, color=MUTED, family=SANS))
    fig.update_layout(**st, legend=dict(orientation="h", y=-0.18, x=0,
                                        font=dict(size=11, color=TEXT, family=SANS),
                                        bgcolor="rgba(0,0,0,0)"))
    return fig


def plot_hhi_trend(mode="annual"):
    if mode == "annual":
        df = compute_annual_hhi()
        x = df["refYear"]; x_title = "Year"
        hover = "%{x}<br>HHI: %{y:.3f}<extra></extra>"
    else:
        df = compute_monthly_hhi()
        x = df["month_start"]; x_title = "Month"
        hover = "%{x|%b %Y}<br>HHI: %{y:.3f}<extra></extra>"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=df["hhi"], mode="lines+markers" if mode == "annual" else "lines",
        line=dict(color=DARK, width=2),
        marker=dict(size=6, color=DARK),
        fill="tozeroy", fillcolor="rgba(13,27,42,0.06)",
        hovertemplate=hover, name="HHI",
    ))
    # Concentration band guides
    fig.add_hline(y=0.25, line=dict(color=ACCENT, width=1, dash="dot"),
                  annotation_text="Moderate concentration", annotation_position="top left",
                  annotation_font=dict(color=ACCENT, size=10))
    fig.add_hline(y=0.40, line=dict(color=RED, width=1, dash="dot"),
                  annotation_text="High concentration", annotation_position="top left",
                  annotation_font=dict(color=RED, size=10))
    st = _axes_style()
    st["xaxis"]["title"] = dict(text=x_title, font=dict(size=12, color=MUTED, family=SANS))
    st["yaxis"]["title"] = dict(text="HHI", font=dict(size=12, color=MUTED, family=SANS))
    fig.update_layout(**st, showlegend=False)
    return fig


def plot_yoy_growth():
    if not TRADE_AVAILABLE: return go.Figure()
    tot = ANNUAL_SUP_DF.groupby("refYear", as_index=False)["annual_primary_value"].sum()
    tot["yoy"] = tot["annual_primary_value"].pct_change() * 100
    colors = [GREEN if v >= 0 else RED for v in tot["yoy"].fillna(0)]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tot["refYear"], y=tot["yoy"], marker_color=colors,
        hovertemplate="%{x}<br>YoY: %{y:+.1f}%<extra></extra>",
        name="YoY growth",
    ))
    fig.add_hline(y=0, line=dict(color="#333", width=1))
    st = _axes_style()
    st["xaxis"]["title"] = dict(text="Year", font=dict(size=12, color=MUTED, family=SANS))
    st["yaxis"]["title"] = dict(text="YoY Change (%)", font=dict(size=12, color=MUTED, family=SANS))
    fig.update_layout(**st, showlegend=False)
    return fig


def plot_volatility_bars():
    df = compute_supplier_volatility()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["volatility_12m"] * 100, y=df["supplier"], orientation="h",
        marker_color=[SUPPLIER_COLORS.get(s, "#888") for s in df["supplier"]],
        hovertemplate="%{y}<br>12-month σ: %{x:.2f}%<extra></extra>",
        name="Volatility",
    ))
    st = _axes_style()
    st["xaxis"]["title"] = dict(text="12-month Share σ (percentage points)",
                                font=dict(size=12, color=MUTED, family=SANS))
    st["yaxis"]["title"] = None
    st["yaxis"]["autorange"] = "reversed"
    st["margin"] = dict(l=100, r=24, t=12, b=40)
    fig.update_layout(**st, showlegend=False)
    return fig


def plot_baseline_vs_optimized_bars(supplier_names, baseline_vals, optimized_vals):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=supplier_names, x=[v * 100 for v in baseline_vals], name="Baseline",
        orientation="h", marker_color=DARK2, opacity=0.45, marker_line_width=0,
        hovertemplate="%{y}: %{x:.1f}%<extra>Baseline</extra>",
    ))
    fig.add_trace(go.Bar(
        y=supplier_names, x=[v * 100 for v in optimized_vals], name="Optimized",
        orientation="h", marker_color=ACCENT, marker_line_width=0,
        hovertemplate="%{y}: %{x:.1f}%<extra>Optimized</extra>",
    ))
    st = _axes_style()
    st["xaxis"]["title"] = dict(text="Share (%)", font=dict(size=12, color=MUTED, family=SANS))
    st["yaxis"]["autorange"] = "reversed"
    st["margin"] = dict(l=100, r=24, t=12, b=40)
    fig.update_layout(
        **st, barmode="group",
        legend=dict(x=0.98, y=0.02, xanchor="right", yanchor="bottom",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=TEXT, family=SANS)),
    )
    return fig


def plot_port_bars(port_names, cur_vals, opt_vals):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[PORT_LABEL.get(p, p) for p in port_names], y=[v * 100 for v in cur_vals],
        name="Current", marker_color=DARK2, opacity=0.45, marker_line_width=0,
        hovertemplate="%{x}: %{y:.1f}%<extra>Current</extra>",
    ))
    fig.add_trace(go.Bar(
        x=[PORT_LABEL.get(p, p) for p in port_names], y=[v * 100 for v in opt_vals],
        name="Optimized", marker_color=ACCENT, marker_line_width=0,
        hovertemplate="%{x}: %{y:.1f}%<extra>Optimized</extra>",
    ))
    st = _axes_style()
    st["xaxis"]["title"] = None
    st["yaxis"]["title"] = dict(text="Share (%)", font=dict(size=12, color=MUTED, family=SANS))
    fig.update_layout(
        **st, barmode="group",
        legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=TEXT, family=SANS)),
    )
    return fig


def plot_objective_breakdown(comp):
    labels = ["Supplier Risk", "Port Risk", "Supplier Conc.",
              "Port Conc.", "Stability", "Cost (norm.)"]
    values = [comp["sr"], comp["pr"], comp["sc"], comp["pc"], comp["stab"], comp["cn"]]
    colors = [RED, BLUE, ACCENT, GREEN, "#9b6ba8", DARK]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=values, marker_color=colors, marker_line_width=0,
        hovertemplate="%{x}: %{y:.3f}<extra></extra>",
    ))
    st = _axes_style()
    st["xaxis"]["title"] = None
    st["yaxis"]["title"] = dict(text="Component value", font=dict(size=12, color=MUTED, family=SANS))
    fig.update_layout(**st, showlegend=False)
    return fig


# =============================================================================
# DASH APP
# =============================================================================
app = dash.Dash(
    __name__,
    title="U.S. Semi Supply Chain Digital Twin",
    suppress_callback_exceptions=True,
    update_title=None,
)
server = app.server


# =============================================================================
# LAYOUT PRIMITIVES
# =============================================================================
def card(extra=None):
    s = {"background": CARD, "borderRadius": RADIUS, "boxShadow": SHADOW,
         "padding": "18px", "boxSizing": "border-box", "overflow": "hidden"}
    if extra: s.update(extra)
    return s


def dark_card(extra=None):
    s = {"background": DARK, "borderRadius": RADIUS, "boxShadow": SHADOW,
         "padding": "18px", "boxSizing": "border-box", "overflow": "hidden",
         "color": "#ffffff"}
    if extra: s.update(extra)
    return s


def accent_card(extra=None):
    s = {"background": ACCENT, "borderRadius": RADIUS, "boxShadow": SHADOW,
         "padding": "18px", "boxSizing": "border-box", "overflow": "hidden"}
    if extra: s.update(extra)
    return s


def label(text, color=MUTED, size="11px", weight="500", mb="6px", upper=True):
    return html.Div(text, style={
        "fontSize": size, "fontWeight": weight, "color": color,
        "marginBottom": mb, "letterSpacing": "0.06em",
        "textTransform": "uppercase" if upper else "none", "fontFamily": SANS,
    })


def card_title(text, color=TEXT):
    return html.Div(text, style={
        "fontSize": "16px", "fontWeight": "700", "color": color,
        "marginBottom": "4px", "fontFamily": SANS,
    })


def card_subtitle(text, color=MUTED):
    return html.Div(text, style={
        "fontSize": "12px", "color": color, "marginBottom": "10px",
        "fontFamily": SANS, "lineHeight": "1.4",
    })


def kpi_card(label_text, value_text, detail_text=None, color=TEXT, accent_bar=True):
    return html.Div([
        label(label_text, mb="6px"),
        html.Div(value_text, style={
            "fontFamily": MONO, "fontSize": "26px", "fontWeight": "800",
            "color": color, "letterSpacing": "-0.01em", "lineHeight": "1.1",
        }),
        html.Div(detail_text or "", style={
            "fontSize": "11px", "color": MUTED, "marginTop": "4px",
            "fontFamily": SANS,
        }),
    ], style=card({
        "borderLeft": f"4px solid {ACCENT}" if accent_bar else "none",
        "minHeight": "96px",
    }))


GRAPH_CFG = {"displayModeBar": False, "responsive": True}


# =============================================================================
# NAVIGATION / HEADER
# =============================================================================
def make_header():
    return html.Div([
        html.Div([
            html.Div("INDENG 243 · ANALYTICS LAB · SPRING 2026", style={
                "fontSize": "9px", "color": ACCENT, "letterSpacing": "0.18em",
                "fontWeight": "700", "textTransform": "uppercase", "fontFamily": SANS,
            }),
            html.Div("U.S. Semiconductor Supply Chain Digital Twin", style={
                "fontSize": "22px", "color": "#ffffff", "fontWeight": "800",
                "fontFamily": SANS, "marginTop": "4px", "letterSpacing": "-0.01em",
            }),
            html.Div("Stochastic simulation · constrained optimization · historical trade context",
                     style={"fontSize": "12px", "color": MUTED2, "fontFamily": SANS,
                            "marginTop": "4px"}),
        ], style={"flex": "1", "minWidth": "0"}),
        html.Div([
            html.Div("SUPPLIERS IN SCOPE", style={
                "fontSize": "9px", "color": MUTED2, "letterSpacing": "0.12em",
                "fontWeight": "600", "fontFamily": SANS,
            }),
            html.Div(" · ".join(SUPPLIERS_ORDER), style={
                "fontSize": "12px", "color": "#ffffff",
                "fontFamily": MONO, "marginTop": "4px",
            }),
        ], style={"textAlign": "right"}),
    ], style={
        "background": DARK, "padding": "22px 28px",
        "display": "flex", "alignItems": "center", "gap": "18px",
    })


def make_tabs():
    tab_style = {
        "padding": "14px 22px", "fontFamily": SANS, "fontSize": "13px",
        "fontWeight": "600", "color": MUTED, "letterSpacing": "0.04em",
        "border": "none", "backgroundColor": CARD,
        "borderBottom": f"3px solid transparent",
    }
    tab_selected = {
        **tab_style, "color": TEXT,
        "borderBottom": f"3px solid {ACCENT}",
        "backgroundColor": CARD,
    }
    return dcc.Tabs(
        id="tabs",
        value="overview",
        children=[
            dcc.Tab(label="Overview",           value="overview",   style=tab_style, selected_style=tab_selected),
            dcc.Tab(label="Historical Context", value="historical", style=tab_style, selected_style=tab_selected),
            dcc.Tab(label="Live Digital Twin",  value="twin",       style=tab_style, selected_style=tab_selected),
            dcc.Tab(label="Strategy Optimizer", value="optimizer",  style=tab_style, selected_style=tab_selected),
            dcc.Tab(label="Shock Library",      value="shocks",     style=tab_style, selected_style=tab_selected),
        ],
        style={"backgroundColor": CARD, "borderBottom": f"1px solid {BORDER}"},
    )


def make_footer():
    return html.Div(
        "UC Berkeley INDENG 243 · Analytics Lab · Spring 2026 · "
        "Data: UN Comtrade · Port of LA · Port of NY/NJ",
        style={
            "fontSize": "10px", "color": "#bbbbbb", "textAlign": "center",
            "padding": "14px", "fontFamily": SANS,
        },
    )


# =============================================================================
# TAB LAYOUTS
# =============================================================================
def layout_overview():
    total_latest = None
    top_supplier, top_share = "Malaysia", BASE["supplier_shares"].get("Malaysia", 0)
    latest_hhi = None
    la_share = BASE["port_shares"].get(PORTS_FULL[0], 0)
    ny_share = BASE["port_shares"].get(PORTS_FULL[1], 0)

    if TRADE_AVAILABLE:
        latest_year = get_latest_full_year()
        total_latest = float(
            ANNUAL_SUP_DF[ANNUAL_SUP_DF["refYear"] == latest_year]["annual_primary_value"].sum())
        latest_shares = get_year_share(latest_year, SUPPLIERS_ORDER + ["Singapore"])
        if latest_shares is not None:
            top_row = latest_shares.sort_values(ascending=False).head(1)
            top_supplier, top_share = top_row.index[0], float(top_row.iloc[0])
            latest_hhi = float(np.sum(latest_shares.values ** 2))

    # KPI row
    kpis = html.Div([
        kpi_card(
            "Latest Annual Imports",
            f"${total_latest/1e9:.1f}B" if total_latest else "—",
            f"UN Comtrade, {get_latest_full_year()}" if TRADE_AVAILABLE else "Trade data unavailable",
        ),
        kpi_card(
            "Top Supplier",
            f"{top_supplier}  {top_share:.0%}",
            "Share of U.S. semi imports, latest full year",
            color=ACCENT,
        ),
        kpi_card(
            "Supplier HHI",
            f"{latest_hhi:.2f}" if latest_hhi else "—",
            f"{'High' if latest_hhi and latest_hhi >= 0.30 else 'Moderate' if latest_hhi and latest_hhi >= 0.22 else 'Low'} concentration"
            if latest_hhi else "",
            color=RED if latest_hhi and latest_hhi >= 0.30 else ACCENT,
        ),
        kpi_card(
            "Port Split",
            f"LA {la_share:.0%}  ·  NY/NJ {ny_share:.0%}",
            "TEU throughput, 2022-2024 average",
            color=BLUE,
        ),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)",
              "gap": "12px", "marginBottom": "12px"})

    # Hero
    hero = html.Div([
        html.Div([
            html.Div("ABOUT THIS TOOL", style={
                "fontSize": "9px", "fontWeight": "600", "color": ACCENT,
                "letterSpacing": "2px", "textTransform": "uppercase",
                "fontFamily": SANS, "marginBottom": "6px",
            }),
            html.Div("A working digital twin of U.S. semiconductor imports", style={
                "fontSize": "22px", "fontWeight": "800", "color": "#ffffff",
                "fontFamily": SANS, "marginBottom": "14px", "lineHeight": "1.25",
            }),
            html.Div([
                "Built for decision-makers protecting critical chip imports from global disruptions. ",
                "Uses SimPy stochastic simulations and a constrained optimization engine to model ",
                "how supply flows from major partners like Malaysia and China through domestic ports. ",
            ], style={"fontSize": "13px", "color": MUTED2, "fontFamily": SANS,
                      "lineHeight": "1.7", "marginBottom": "10px"}),
            html.Div([
                "The system calculates a ",
                html.Span("Resilience Score", style={"color": ACCENT, "fontWeight": "700"}),
                " that balances transportation costs against concentration risk. ",
                "Adjust disruption sliders on the Live Digital Twin tab to simulate ",
                "real-world shocks such as port labor strikes or geopolitical events.",
            ], style={"fontSize": "13px", "color": MUTED2, "fontFamily": SANS,
                      "lineHeight": "1.7"}),
        ], style=dark_card({"padding": "28px", "borderLeft": f"5px solid {ACCENT}",
                             "flex": "2"})),
        html.Div([
            html.Div("START HERE", style={
                "fontSize": "9px", "fontWeight": "600", "color": DARK,
                "letterSpacing": "2px", "fontFamily": SANS, "marginBottom": "10px",
            }),
            html.Div("Four ways to explore", style={
                "fontSize": "16px", "fontWeight": "800", "color": DARK,
                "fontFamily": SANS, "marginBottom": "14px",
            }),
            html.Ol([
                html.Li([html.Strong("Historical Context"),
                         " — how U.S. supplier concentration evolved since 2010."],
                        style={"fontSize": "12px", "marginBottom": "8px", "color": DARK}),
                html.Li([html.Strong("Live Digital Twin"),
                         " — simulate a shock and watch the optimizer respond."],
                        style={"fontSize": "12px", "marginBottom": "8px", "color": DARK}),
                html.Li([html.Strong("Strategy Optimizer"),
                         " — trade-calibrated baselines with cost / resilience tradeoff."],
                        style={"fontSize": "12px", "marginBottom": "8px", "color": DARK}),
                html.Li([html.Strong("Shock Library"),
                         " — curated scenarios with historical precedents."],
                        style={"fontSize": "12px", "marginBottom": "0", "color": DARK}),
            ], style={"paddingLeft": "18px", "margin": "0", "fontFamily": SANS}),
        ], style=accent_card({"flex": "1"})),
    ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"})

    # Methodology
    method = html.Div([
        html.Div([
            card_title("SimPy Stochastic Simulation"),
            card_subtitle(
                "A discrete-event engine runs a 180-day horizon with ±3% daily noise. "
                "A shock lands on day 30; queues build at each port and partially drain "
                "based on remaining capacity. Reports daily throughput, peak queue, "
                "and recovery time."
            ),
            html.Div([
                html.Span("Inputs: ", style={"color": MUTED, "fontSize": "11px",
                                              "fontFamily": SANS, "fontWeight": "600",
                                              "textTransform": "uppercase",
                                              "letterSpacing": "0.08em"}),
                html.Span("supplier shares · port capacities · shock magnitude",
                          style={"fontFamily": MONO, "fontSize": "11px", "color": TEXT}),
            ], style={"marginTop": "8px"}),
        ], style=card({"flex": "1"})),
        html.Div([
            card_title("SLSQP Constrained Optimizer"),
            card_subtitle(
                "Solves for a supplier×port flow matrix minimizing a weighted blend of "
                "transport cost, supplier/port risk, Herfindahl concentration, and deviation "
                "from the current network. Constraints: flows sum to 1, supplier cap (default 70%), "
                "port capacity."
            ),
            html.Div([
                html.Span("Preference knob: ", style={"color": MUTED, "fontSize": "11px",
                                                       "fontFamily": SANS, "fontWeight": "600",
                                                       "textTransform": "uppercase",
                                                       "letterSpacing": "0.08em"}),
                html.Span("0 = cost-first  ↔  100 = resilience-first",
                          style={"fontFamily": MONO, "fontSize": "11px", "color": TEXT}),
            ], style={"marginTop": "8px"}),
        ], style=card({"flex": "1"})),
        html.Div([
            card_title("Trade-Calibrated Baseline"),
            card_subtitle(
                "Baselines and supplier risk scores can be calibrated from UN Comtrade "
                "monthly import data (2010-2025). A historical volatility + trend-decline "
                "signal boosts the risk score of suppliers that have been unstable or "
                "steadily losing share."
            ),
            html.Div([
                html.Span("Data: ", style={"color": MUTED, "fontSize": "11px",
                                            "fontFamily": SANS, "fontWeight": "600",
                                            "textTransform": "uppercase",
                                            "letterSpacing": "0.08em"}),
                html.Span(f"{len(MONTHLY_SUP_DF['month_start'].unique())} months · "
                          f"6 partners" if TRADE_AVAILABLE else "Trade data unavailable",
                          style={"fontFamily": MONO, "fontSize": "11px", "color": TEXT}),
            ], style={"marginTop": "8px"}),
        ], style=card({"flex": "1"})),
    ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"})

    return html.Div([kpis, hero, method],
                    style={"padding": "18px 14px", "background": BG})


def layout_historical():
    if not TRADE_AVAILABLE:
        return html.Div(
            card_title("Historical trade data not loaded"),
            style=card({"margin": "18px 14px"})
        )

    years = sorted(ANNUAL_SUP_DF["refYear"].unique().tolist())
    all_sup = sorted(MONTHLY_SUP_DF["supplier"].unique().tolist())

    # Top row: context card + imports chart
    row1 = html.Div([
        html.Div([
            html.Div("HISTORICAL CONTEXT", style={
                "fontSize": "9px", "fontWeight": "600", "color": ACCENT,
                "letterSpacing": "2px", "fontFamily": SANS, "marginBottom": "6px",
            }),
            html.Div("15 years of U.S. semiconductor imports", style={
                "fontSize": "18px", "fontWeight": "800", "color": "#ffffff",
                "fontFamily": SANS, "marginBottom": "12px", "lineHeight": "1.3",
            }),
            html.Div(
                "Monthly trade flows from the six major Asian partners. "
                "Concentration has steadily risen since 2010; Malaysia's share has "
                "doubled from about 26% to over 50%.",
                style={"fontSize": "12px", "color": MUTED2, "fontFamily": SANS,
                       "lineHeight": "1.6"},
            ),
        ], style=dark_card({"flex": "1", "borderLeft": f"4px solid {ACCENT}"})),

        html.Div([
            html.Div([
                card_title("Monthly U.S. Semiconductor Imports"),
                card_subtitle("Total value of integrated-circuit imports (HS 8542) from all partners."),
                dcc.RadioItems(
                    id="hist-metric",
                    options=[{"label": " Import Value", "value": "primaryValue"},
                             {"label": " Net Weight", "value": "netWgt"}],
                    value="primaryValue", inline=True,
                    style={"fontSize": "11px", "fontFamily": SANS,
                           "color": MUTED, "marginBottom": "8px"},
                ),
            ]),
            dcc.Graph(id="g-hist-total", config=GRAPH_CFG, style={"height": "280px"}),
        ], style=card({"flex": "2.2"})),
    ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"})

    # Row 2: supplier share trend + HHI
    row2 = html.Div([
        html.Div([
            card_title("Supplier Share Evolution"),
            card_subtitle("Stacked monthly share by country of origin."),
            html.Div([
                dcc.Checklist(
                    id="hist-suppliers",
                    options=[{"label": " " + s, "value": s} for s in all_sup],
                    value=all_sup, inline=True,
                    style={"fontSize": "11px", "fontFamily": SANS, "color": MUTED,
                           "marginBottom": "4px"},
                ),
            ]),
            dcc.RadioItems(
                id="hist-trend-view",
                options=[{"label": " Monthly (stacked)", "value": "monthly"},
                         {"label": " Annual (lines)", "value": "annual"}],
                value="monthly", inline=True,
                style={"fontSize": "11px", "fontFamily": SANS, "color": MUTED,
                       "marginBottom": "4px"},
            ),
            dcc.Graph(id="g-hist-trend", config=GRAPH_CFG, style={"height": "310px"}),
        ], style=card({"flex": "1.4"})),

        html.Div([
            card_title("Concentration (HHI) over Time"),
            card_subtitle("Herfindahl-Hirschman Index — higher means more concentrated sourcing."),
            dcc.RadioItems(
                id="hist-hhi-mode",
                options=[{"label": " Annual", "value": "annual"},
                         {"label": " Monthly", "value": "monthly"}],
                value="annual", inline=True,
                style={"fontSize": "11px", "fontFamily": SANS, "color": MUTED,
                       "marginBottom": "4px"},
            ),
            dcc.Graph(id="g-hist-hhi", config=GRAPH_CFG, style={"height": "330px"}),
        ], style=card({"flex": "1"})),
    ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"})

    # Row 3: YoY growth + volatility
    row3 = html.Div([
        html.Div([
            card_title("Year-over-Year Import Growth"),
            card_subtitle("Total semiconductor import value relative to the prior year."),
            dcc.Graph(figure=plot_yoy_growth(), config=GRAPH_CFG, style={"height": "260px"}),
        ], style=card({"flex": "1"})),
        html.Div([
            card_title("Supplier Volatility (12-month rolling σ)"),
            card_subtitle("Standard deviation of the monthly import share. "
                          "Higher values indicate a more erratic supplier."),
            dcc.Graph(figure=plot_volatility_bars(), config=GRAPH_CFG, style={"height": "260px"}),
        ], style=card({"flex": "1"})),
    ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"})

    return html.Div([row1, row2, row3], style={"padding": "18px 14px", "background": BG})


def layout_twin():
    """Teammate's v3 dashboard layout (with bug fixes). Fully preserved."""
    struct = _static_structural_risk()

    control_bar = html.Div([
        html.Div("Shock Controls", style={
            "fontSize": "12px", "fontWeight": "800", "color": TEXT,
            "letterSpacing": "0.12em", "textTransform": "uppercase",
            "fontFamily": SANS, "whiteSpace": "nowrap", "flexShrink": "0",
            "paddingRight": "20px",
        }),
        html.Div([
            label("Malaysia Disruption"),
            dcc.Slider(id="sl-mal", min=0, max=100, step=5, value=30,
                       marks={0: "0%", 25: "25%", 50: "50%", 75: "75%", 100: "100%"},
                       tooltip={"always_visible": True, "placement": "bottom"},
                       className="dash-slider"),
        ], style={"flex": "35", "padding": "0 16px", "minWidth": "160px"}),
        html.Div([
            label("Port of LA Capacity Loss"),
            dcc.Slider(id="sl-la", min=0, max=100, step=5, value=0,
                       marks={0: "0%", 30: "30%", 60: "60%", 100: "100%"},
                       tooltip={"always_visible": True, "placement": "bottom"},
                       className="dash-slider"),
        ], style={"flex": "35", "padding": "0 16px", "minWidth": "160px"}),
        html.Div([
            label("Risk Level", color=MUTED, mb="4px"),
            html.Div(id="risk-label", style={
                "fontFamily": MONO, "fontSize": "22px", "fontWeight": "800",
                "letterSpacing": "0.12em", "textAlign": "center", "padding": "4px 0",
            }),
        ], style={"flex": "30", "paddingLeft": "16px", "minWidth": "110px",
                  "borderLeft": f"1px solid {BORDER}"}),
    ], style=card({
        "display": "flex", "flexDirection": "row", "alignItems": "center",
        "gap": "0", "margin": "0 0 10px 0", "flexShrink": "0",
        "padding": "14px 18px",
    }))

    # Preset quick buttons
    preset_bar = html.Div([
        html.Div("Or load a preset shock:", style={
            "fontSize": "11px", "color": MUTED, "fontFamily": SANS,
            "textTransform": "uppercase", "letterSpacing": "0.08em",
            "fontWeight": "600", "marginRight": "10px",
        }),
        *[html.Button(
            SHOCK_PRESETS[k]["label"],
            id={"type": "preset-btn", "idx": k},
            n_clicks=0,
            style={
                "fontSize": "11px", "fontFamily": SANS, "fontWeight": "600",
                "padding": "6px 12px", "borderRadius": "20px",
                "border": f"1px solid {BORDER}", "backgroundColor": CARD,
                "cursor": "pointer", "color": TEXT, "marginRight": "6px",
            },
        ) for k in SHOCK_PRESETS.keys()],
    ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap",
              "margin": "0 0 10px 0", "padding": "8px 14px",
              "background": CARD, "borderRadius": RADIUS, "boxShadow": SHADOW})

    # Structural risk strip
    strip = html.Div([
        html.Div([
            label("Concentration Risk", color=MUTED, mb="4px"),
            html.Div(id="struct-overall", style={
                "fontFamily": MONO, "fontSize": "22px", "fontWeight": "800",
                "letterSpacing": "0.10em", "marginBottom": "4px",
            }),
            html.Div(
                "Inherent concentration and dependency in the network under the current "
                "disruption, before optimization responds.",
                style={"fontSize": "11px", "color": MUTED, "fontFamily": SANS,
                       "lineHeight": "1.4"},
            ),
        ], style={"flex": "1.2", "minWidth": "0"}),
        html.Div([
            label("Supplier Concentration", mb="4px"),
            html.Div(id="struct-supplier-label", style={
                "fontSize": "16px", "fontWeight": "700", "fontFamily": SANS, "color": TEXT}),
            html.Div(id="struct-supplier-value", style={
                "fontSize": "11px", "color": MUTED, "fontFamily": MONO}),
        ], style={"flex": "1", "minWidth": "0"}),
        html.Div([
            label("Malaysia Dependence", mb="4px"),
            html.Div(id="struct-malaysia-label", style={
                "fontSize": "16px", "fontWeight": "700", "fontFamily": SANS, "color": TEXT}),
            html.Div(id="struct-malaysia-value", style={
                "fontSize": "11px", "color": MUTED, "fontFamily": MONO}),
        ], style={"flex": "1", "minWidth": "0"}),
        html.Div([
            label("Port Concentration", mb="4px"),
            html.Div(id="struct-port-label", style={
                "fontSize": "16px", "fontWeight": "700", "fontFamily": SANS, "color": TEXT}),
            html.Div(id="struct-port-value", style={
                "fontSize": "11px", "color": MUTED, "fontFamily": MONO}),
        ], style={"flex": "1", "minWidth": "0"}),
    ], style=card({
        "display": "flex", "flexDirection": "row", "alignItems": "stretch",
        "gap": "18px", "margin": "0 0 10px 0",
        "padding": "14px 18px", "flexShrink": "0",
    }))

    # Row 1: About + Timeline + Resilience
    row1 = html.Div([
        html.Div([
            html.Div([
                html.Div("ABOUT THIS TOOL", style={
                    "fontSize": "9px", "fontWeight": "600", "color": ACCENT,
                    "letterSpacing": "2px", "textTransform": "uppercase",
                    "fontFamily": SANS, "marginBottom": "6px",
                }),
                html.Div("U.S. Semiconductor Import Supply Chain", style={
                    "fontSize": "16px", "fontWeight": "700", "color": "#ffffff",
                    "fontFamily": SANS, "marginBottom": "12px", "lineHeight": "1.3",
                }),
                html.Div("Built on 20 years of U.S. import data and real port throughput "
                         "from the Port of LA and Port of NY/NJ.",
                         style={"fontSize": "13px", "color": MUTED2, "marginBottom": "5px",
                                "fontFamily": SANS, "lineHeight": "1.7"}),
                html.Div("The U.S. sources roughly 65% of its semiconductors from a single "
                         "country, Malaysia.",
                         style={"fontSize": "13px", "color": MUTED2, "marginBottom": "5px",
                                "fontFamily": SANS, "lineHeight": "1.7"}),
                html.Div("Use the sliders to simulate a disruption. The optimizer responds "
                         "in real time.",
                         style={"fontSize": "13px", "color": MUTED2,
                                "fontFamily": SANS, "lineHeight": "1.7"}),
            ], style=dark_card({"flex": "62", "borderLeft": f"4px solid {ACCENT}"})),

            html.Div([
                html.Div("LIVE SCENARIO", style={
                    "fontSize": "9px", "fontWeight": "600", "color": "#7a6020",
                    "letterSpacing": "2px", "textTransform": "uppercase",
                    "fontFamily": SANS, "marginBottom": "8px",
                }),
                dcc.Markdown(id="narrative-text", style={
                    "fontSize": "13px", "color": TEXT, "lineHeight": "1.6",
                    "fontFamily": SANS,
                }),
            ], style=accent_card({"flex": "38"})),
        ], style={"flex": "30", "display": "flex", "flexDirection": "column",
                  "gap": "12px", "minWidth": "0", "height": "100%", "flexShrink": "0"}),

        html.Div([
            card_title("Supply Flow Timeline"),
            card_subtitle("Each point is one day of semiconductor imports measured in TEU. "
                          "The gap between the grey baseline and the red disrupted line "
                          "shows the daily cost of the shock."),
            dcc.Graph(id="g-timeline", config=GRAPH_CFG,
                      style={"height": "320px", "width": "100%"}),
        ], style=card({"flex": "48", "display": "flex", "flexDirection": "column",
                       "minWidth": "0", "height": "400px"})),

        html.Div([
            label("System Resilience", color="#aaaaaa", size="11px"),
            html.Div(
                "A score from 0 to 100 that measures how well the supply chain can absorb "
                "disruption. Watch it drop under shock, then see the optimizer bring it back.",
                style={"fontSize": "12px", "color": "#666", "marginBottom": "6px",
                       "fontFamily": SANS, "lineHeight": "1.4"},
            ),
            html.Div("PRE-SHOCK → DISRUPTED → BEST ACHIEVABLE",
                     style={"fontSize": "10px", "color": "#666666",
                            "marginBottom": "10px", "fontFamily": SANS,
                            "lineHeight": "1.4", "letterSpacing": "0.04em"}),
            html.Div(id="g-resilience", style={
                "display": "flex", "flex": "1", "alignItems": "center",
                "justifyContent": "center", "minHeight": "0", "overflow": "hidden",
            }),
        ], style=dark_card({
            "flex": "22", "display": "flex", "flexDirection": "column",
            "minWidth": "0", "height": "100%", "overflow": "hidden",
        })),
    ], style={"display": "flex", "gap": "12px", "minHeight": "0"})

    # Row 2: supplier + port + recommendation
    row2 = html.Div([
        html.Div([
            card_title("Supplier Allocation"),
            card_subtitle("The gold bars show the optimizer's recommended sourcing mix. "
                          "Grey shows current baseline allocations."),
            dcc.Graph(id="g-supplier", config=GRAPH_CFG,
                      style={"height": "260px", "width": "100%"}),
        ], style=card({"flex": "1", "display": "flex", "flexDirection": "column",
                       "minWidth": "0", "height": "340px"})),

        html.Div([
            card_title("Port Split"),
            html.Div(f"LA handles about {BASE['port_shares'].get(PORTS_FULL[0], 0.53):.0%} "
                     f"of dual-hub TEU throughput and NY/NJ about "
                     f"{BASE['port_shares'].get(PORTS_FULL[1], 0.47):.0%}. "
                     "The optimizer rebalances the mix to reduce single-hub exposure.",
                     style={"fontSize": "12px", "color": MUTED, "marginBottom": "4px",
                            "fontFamily": SANS, "lineHeight": "1.4", "fontWeight": "400"}),
            html.Div([
                dcc.Graph(id="g-port",
                          config={"displayModeBar": False, "responsive": False},
                          style={"height": "180px", "width": "200px"}),
            ], style={"height": "180px", "overflow": "hidden", "display": "flex",
                      "alignItems": "center", "justifyContent": "center"}),
        ], style=card({"flex": "1", "display": "flex", "flexDirection": "column",
                       "minWidth": "0", "height": "340px"})),

        html.Div([
            card_title("Recommendation", color="#ffffff"),
            html.Div("Here is what the optimizer recommends to reduce supply chain risk "
                     "under the current scenario.",
                     style={"fontSize": "12px", "color": MUTED2, "marginBottom": "8px",
                            "fontFamily": SANS, "lineHeight": "1.4"}),
            html.Div(id="g-rec", style={"flex": "1", "overflowY": "hidden"}),
        ], style=dark_card({
            "flex": "1", "display": "flex", "flexDirection": "column",
            "minWidth": "0", "height": "340px",
        })),
    ], style={"display": "flex", "gap": "8px", "minHeight": "0", "marginBottom": "12px"})

    return html.Div([control_bar, preset_bar, strip, row1, row2],
                    style={"padding": "18px 14px", "background": BG})


def layout_optimizer():
    years = sorted(ANNUAL_SUP_DF["refYear"].unique().tolist()) if TRADE_AVAILABLE else []
    default_year = max(years) if years else 2024

    # Sidebar-like control card on left, results on right
    controls = html.Div([
        card_title("Decision Inputs"),
        card_subtitle("Set strategy preferences and constraints, then click Run."),

        label("Objective Preference", mb="4px"),
        dcc.Dropdown(id="opt-preset",
                     options=[{"label": "Cost-focused", "value": "cost"},
                              {"label": "Balanced",     "value": "balanced"},
                              {"label": "Resilience-focused", "value": "resilience"}],
                     value="balanced", clearable=False,
                     style={"fontSize": "12px", "marginBottom": "10px"}),

        label("Resilience Preference (0 = cost ↔ 100 = resilience)", mb="4px"),
        dcc.Slider(id="opt-pref", min=0, max=100, step=5, value=50,
                   marks={0: "0", 25: "25", 50: "50", 75: "75", 100: "100"},
                   tooltip={"always_visible": False, "placement": "bottom"}),
        html.Div(style={"height": "10px"}),

        label("Maximum Supplier Share Cap", mb="4px"),
        dcc.Slider(id="opt-cap", min=0.20, max=0.80, step=0.05, value=0.70,
                   marks={0.2: "20%", 0.4: "40%", 0.6: "60%", 0.8: "80%"},
                   tooltip={"always_visible": False, "placement": "bottom"}),
        html.Div(style={"height": "10px"}),

        label("Port Capacity Assumption", mb="4px"),
        dcc.RadioItems(id="opt-cap-mult",
                       options=[{"label": " Reduced (−15%)",  "value": 0.85},
                                {"label": " Current",          "value": 1.00},
                                {"label": " Expanded (+15%)",  "value": 1.15}],
                       value=1.00, inline=False,
                       style={"fontSize": "12px", "fontFamily": SANS}),
        html.Div(style={"height": "10px"}),

        label("Target Annual TEU Demand", mb="4px"),
        dcc.Input(id="opt-demand", type="number", value=8_000_000,
                  min=100_000, max=20_000_000, step=100_000, debounce=True,
                  style={"width": "100%", "fontFamily": MONO, "padding": "6px",
                         "border": f"1px solid {BORDER}", "borderRadius": "6px"}),
        html.Div(style={"height": "14px"}),

        html.Hr(style={"border": f"none", "borderTop": f"1px solid {BORDER}"}),
        label("Trade-Based Baseline", mb="4px"),
        dcc.Dropdown(id="opt-baseline-mode",
                     options=([{"label": "suppliers.csv baseline",    "value": "csv"}]
                              + ([{"label": "Latest full year",       "value": "latest_year"},
                                  {"label": "Latest 12 months",       "value": "l12m"},
                                  {"label": "Custom year",            "value": "custom"}]
                                 if TRADE_AVAILABLE else [])),
                     value="csv", clearable=False,
                     style={"fontSize": "12px", "marginBottom": "8px"}),
        html.Div([
            label("Custom year", mb="4px"),
            dcc.Dropdown(id="opt-custom-year",
                         options=[{"label": str(y), "value": y} for y in years],
                         value=default_year, clearable=False,
                         style={"fontSize": "12px"}),
        ], id="opt-custom-year-wrap", style={"display": "none"}),

        html.Hr(style={"border": f"none", "borderTop": f"1px solid {BORDER}"}),
        label("Historical Risk Calibration", mb="4px"),
        dcc.Checklist(id="opt-use-hist",
                      options=[{"label": " Enable historical risk adjustment",
                                "value": "on"}],
                      value=["on"] if TRADE_AVAILABLE else [],
                      style={"fontSize": "11px", "color": MUTED}),
        html.Div([
            label("Lookback window (months)", mb="4px"),
            dcc.Slider(id="opt-window", min=6, max=36, step=3, value=12,
                       marks={6: "6", 12: "12", 24: "24", 36: "36"},
                       tooltip={"always_visible": False}),
            html.Div(style={"height": "6px"}),
            label("Adjustment strength", mb="4px"),
            dcc.Slider(id="opt-strength", min=0.0, max=1.0, step=0.05, value=0.35,
                       marks={0: "0", 0.5: "0.5", 1: "1"}),
            html.Div(style={"height": "6px"}),
            label("Volatility weight", mb="4px"),
            dcc.Slider(id="opt-volw", min=0.0, max=1.0, step=0.05, value=0.60,
                       marks={0: "0", 0.5: "0.5", 1: "1"}),
            html.Div(style={"height": "6px"}),
            label("Trend-decline weight", mb="4px"),
            dcc.Slider(id="opt-trendw", min=0.0, max=1.0, step=0.05, value=0.40,
                       marks={0: "0", 0.5: "0.5", 1: "1"}),
        ], id="opt-hist-wrap",
           style={"display": "block" if TRADE_AVAILABLE else "none"}),

        html.Hr(style={"border": f"none", "borderTop": f"1px solid {BORDER}"}),
        html.Details([
            html.Summary("Advanced Weights", style={"fontSize": "11px",
                                                     "color": MUTED, "cursor": "pointer",
                                                     "fontFamily": SANS, "fontWeight": "600",
                                                     "textTransform": "uppercase",
                                                     "letterSpacing": "0.08em"}),
            html.Div(style={"height": "8px"}),
            label("Supplier Risk", mb="2px"),
            dcc.Slider(id="w-sr", min=0, max=1, step=0.05, value=0.35,
                       marks={0: "0", 0.5: "0.5", 1: "1"}),
            label("Port Risk", mb="2px"),
            dcc.Slider(id="w-pr", min=0, max=1, step=0.05, value=0.20,
                       marks={0: "0", 0.5: "0.5", 1: "1"}),
            label("Supplier Concentration", mb="2px"),
            dcc.Slider(id="w-sc", min=0, max=1, step=0.05, value=0.20,
                       marks={0: "0", 0.5: "0.5", 1: "1"}),
            label("Port Concentration", mb="2px"),
            dcc.Slider(id="w-pc", min=0, max=1, step=0.05, value=0.10,
                       marks={0: "0", 0.5: "0.5", 1: "1"}),
            label("Stability Penalty", mb="2px"),
            dcc.Slider(id="w-st", min=0, max=1, step=0.05, value=0.15,
                       marks={0: "0", 0.5: "0.5", 1: "1"}),
        ]),

        html.Div(style={"height": "14px"}),
        html.Button("Run Optimizer", id="opt-run", n_clicks=0,
                    style={
                        "width": "100%", "padding": "10px 14px",
                        "backgroundColor": ACCENT, "color": "#ffffff",
                        "border": "none", "borderRadius": "8px",
                        "fontSize": "13px", "fontWeight": "700",
                        "fontFamily": SANS, "letterSpacing": "0.08em",
                        "textTransform": "uppercase", "cursor": "pointer",
                    }),
    ], style=card({"flex": "0 0 320px", "display": "flex", "flexDirection": "column"}))

    results = html.Div(id="opt-results", style={"flex": "1", "minWidth": "0"})

    return html.Div([
        html.Div([controls, results],
                 style={"display": "flex", "gap": "12px"}),
    ], style={"padding": "18px 14px", "background": BG})


def layout_shock_library():
    intro = html.Div([
        html.Div("SHOCK LIBRARY", style={
            "fontSize": "9px", "fontWeight": "600", "color": ACCENT,
            "letterSpacing": "2px", "fontFamily": SANS, "marginBottom": "6px",
        }),
        html.Div("Six curated disruption scenarios with historical precedents", style={
            "fontSize": "20px", "fontWeight": "800", "color": "#ffffff",
            "fontFamily": SANS, "marginBottom": "10px", "lineHeight": "1.25",
        }),
        html.Div("Each scenario encodes a specific real-world failure mode. Click a card "
                 "to load that preset in the Live Digital Twin.",
                 style={"fontSize": "12px", "color": MUTED2, "fontFamily": SANS,
                        "lineHeight": "1.6"}),
    ], style=dark_card({"margin": "0 0 12px 0", "borderLeft": f"4px solid {ACCENT}"}))

    cards = []
    for key, p in SHOCK_PRESETS.items():
        cards.append(html.Div([
            html.Div([
                html.Div(p["icon"], style={
                    "fontSize": "28px", "color": p["color"], "marginRight": "12px",
                    "fontFamily": MONO, "fontWeight": "800", "lineHeight": "1",
                }),
                html.Div([
                    html.Div(p["label"], style={
                        "fontSize": "15px", "fontWeight": "800", "color": TEXT,
                        "fontFamily": SANS, "marginBottom": "2px",
                    }),
                    html.Div(f"Malaysia −{p['mal']}%  ·  Port of LA −{p['la']}%",
                             style={"fontSize": "10px", "color": MUTED,
                                    "fontFamily": MONO, "letterSpacing": "0.05em"}),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}),

            html.Div(p["desc"], style={
                "fontSize": "12px", "color": TEXT, "fontFamily": SANS,
                "lineHeight": "1.5", "marginBottom": "10px",
            }),

            html.Div("HISTORICAL PRECEDENT", style={
                "fontSize": "9px", "color": MUTED, "fontFamily": SANS,
                "fontWeight": "700", "letterSpacing": "0.12em", "marginBottom": "4px",
            }),
            html.Div(p["precedent"], style={
                "fontSize": "11px", "color": MUTED, "fontFamily": SANS,
                "lineHeight": "1.5", "marginBottom": "10px", "fontStyle": "italic",
            }),

            html.Div("ESTIMATED IMPACT", style={
                "fontSize": "9px", "color": MUTED, "fontFamily": SANS,
                "fontWeight": "700", "letterSpacing": "0.12em", "marginBottom": "4px",
            }),
            html.Div(p["impact"], style={
                "fontSize": "11px", "color": TEXT, "fontFamily": SANS,
                "lineHeight": "1.5", "marginBottom": "14px",
            }),

            html.Button("Open in Digital Twin →",
                id={"type": "shock-btn", "idx": key}, n_clicks=0,
                style={
                    "width": "100%", "padding": "8px 14px",
                    "backgroundColor": DARK, "color": "#ffffff", "border": "none",
                    "borderRadius": "6px", "fontSize": "11px", "fontWeight": "700",
                    "fontFamily": SANS, "letterSpacing": "0.08em",
                    "textTransform": "uppercase", "cursor": "pointer",
                }),
        ], style=card({"borderTop": f"4px solid {p['color']}", "minHeight": "260px"})))

    grid = html.Div(cards, style={
        "display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "12px",
    })

    return html.Div([intro, grid], style={"padding": "18px 14px", "background": BG})


# =============================================================================
# STATIC STRUCTURAL RISK (baseline, for initial state)
# =============================================================================
def _static_structural_risk():
    base_s = np.array([BASE["supplier_shares"].get(s, 0) for s in SUPPLIERS_ORDER], dtype=float)
    base_p = np.array([BASE["port_shares"].get(p, 0) for p in PORTS_FULL], dtype=float)
    sup_hhi = float(np.sum(base_s ** 2))
    port_hhi = float(np.sum(base_p ** 2))
    mal = float(BASE["supplier_shares"].get("Malaysia", 0))
    return {
        "overall": overall_structural_risk(sup_hhi, mal, port_hhi),
        "supplier_hhi": sup_hhi, "port_hhi": port_hhi, "malaysia_share": mal,
        "supplier_label": risk_band(sup_hhi, 0.22, 0.30),
        "malaysia_label": risk_band(mal, 0.30, 0.50),
        "port_label":     risk_band(port_hhi, 0.50, 0.55),
    }


# =============================================================================
# MAIN APP LAYOUT
# =============================================================================
app.layout = html.Div([
    dcc.Store(id="preset-store", data={"mal": 0, "la": 0, "source": "default"}),
    make_header(),
    make_tabs(),
    html.Div(id="tab-content"),
    make_footer(),
], style={"minHeight": "100vh", "background": BG, "fontFamily": SANS})


# Custom CSS
app.index_string = """
<!DOCTYPE html>
<html>
<head>
  {%metas%}<title>{%title%}</title>{%favicon%}{%css%}
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { overflow-y: auto; background: """ + BG + """; }
    .dash-slider .rc-slider-rail { background: """ + BORDER + """; }
    .dash-slider .rc-slider-track { background: """ + ACCENT + """; }
    .dash-slider .rc-slider-handle { border-color: """ + ACCENT + """; background: #fff; }
    .dash-slider .rc-slider-tooltip-content {
      background: """ + ACCENT + """; border-color: """ + ACCENT + """;
      color: #fff; font-family: """ + MONO + """; font-size: 11px;
    }
    .rc-slider-mark-text { font-family: """ + SANS + """; font-size: 10px; color: """ + MUTED + """; }
    .js-plotly-plot .plotly .cursor-default { cursor: default !important; }
    button:hover { opacity: 0.9; }
    .Select-control, .Select-menu-outer { font-family: """ + SANS + """; }
  </style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>
"""


# =============================================================================
# TAB ROUTER
# =============================================================================
@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "historical": return layout_historical()
    if tab == "twin":       return layout_twin()
    if tab == "optimizer":  return layout_optimizer()
    if tab == "shocks":     return layout_shock_library()
    return layout_overview()


# =============================================================================
# HISTORICAL CONTEXT CALLBACKS
# =============================================================================
@app.callback(Output("g-hist-total", "figure"), Input("hist-metric", "value"))
def _hist_total(metric):
    return plot_total_imports(metric)


@app.callback(Output("g-hist-trend", "figure"),
              Input("hist-trend-view", "value"),
              Input("hist-suppliers", "value"))
def _hist_trend(view, suppliers):
    if not suppliers:
        return go.Figure()
    if view == "annual":
        return plot_annual_supplier_share(suppliers)
    return plot_supplier_stacked_area(suppliers)


@app.callback(Output("g-hist-hhi", "figure"), Input("hist-hhi-mode", "value"))
def _hist_hhi(mode):
    return plot_hhi_trend(mode)


# =============================================================================
# PRESET BUTTONS — apply on Twin tab or from Shock Library
# =============================================================================
@app.callback(
    Output("preset-store", "data"),
    Output("tabs", "value", allow_duplicate=True),
    Input({"type": "shock-btn", "idx": ALL}, "n_clicks"),
    Input({"type": "preset-btn", "idx": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def _apply_preset(shock_clicks, preset_clicks):
    trig = ctx.triggered_id
    if not trig or not isinstance(trig, dict):
        return no_update, no_update
    key = trig["idx"]
    if key not in SHOCK_PRESETS:
        return no_update, no_update
    p = SHOCK_PRESETS[key]
    data = {"mal": p["mal"], "la": p["la"], "source": key}
    if trig["type"] == "shock-btn":
        return data, "twin"
    return data, no_update


@app.callback(Output("sl-mal", "value"),
              Output("sl-la", "value"),
              Input("preset-store", "data"),
              prevent_initial_call=True)
def _sync_sliders(data):
    if not data: return no_update, no_update
    return data.get("mal", 0), data.get("la", 0)


# =============================================================================
# LIVE DIGITAL TWIN CALLBACK (ported from v3, bugs fixed)
# =============================================================================
@app.callback(
    Output("g-timeline",       "figure"),
    Output("g-supplier",       "figure"),
    Output("g-port",           "figure"),
    Output("g-resilience",     "children"),
    Output("g-rec",            "children"),
    Output("risk-label",       "children"),
    Output("risk-label",       "style"),
    Output("narrative-text",   "children"),
    Output("struct-overall",   "children"),
    Output("struct-overall",   "style"),
    Output("struct-supplier-label", "children"),
    Output("struct-supplier-value", "children"),
    Output("struct-malaysia-label", "children"),
    Output("struct-malaysia-value", "children"),
    Output("struct-port-label",     "children"),
    Output("struct-port-value",     "children"),
    Input("sl-mal", "value"),
    Input("sl-la",  "value"),
)
def update_twin(mal_pct, la_pct):
    mal_pct = mal_pct or 0
    la_pct  = la_pct  or 0

    base_sim  = run_simulation(0, 0, seed=42)
    shock_sim = run_simulation(mal_pct, la_pct, seed=42)

    post_sup = shock_sim["post_sup"]
    post_prt = shock_sim["post_prt"]

    sh_sup_df, sh_prt_df = _build_dfs(post_sup, post_prt)
    sh_cm = _cost_matrix(sh_sup_df, sh_prt_df)
    opt = run_optimizer(sh_sup_df, sh_prt_df, sh_cm)

    # Structural risk under shock
    shock_sup_arr = _norm(np.array([post_sup.get(s, 0) for s in SUPPLIERS_ORDER], dtype=float))
    shock_prt_arr = _norm(np.array([post_prt.get(p, 0) for p in PORTS_FULL],      dtype=float))
    shock_sup_hhi = float(np.sum(shock_sup_arr ** 2))
    shock_prt_hhi = float(np.sum(shock_prt_arr ** 2))
    mal_idx = SUPPLIERS_ORDER.index("Malaysia")
    shock_mal_share = float(shock_sup_arr[mal_idx])

    struct_sup_lbl = risk_band(shock_sup_hhi, 0.22, 0.30)
    struct_mal_lbl = risk_band(shock_mal_share, 0.30, 0.50)
    struct_prt_lbl = risk_band(shock_prt_hhi, 0.50, 0.55)
    struct_overall = overall_structural_risk(shock_sup_hhi, shock_mal_share, shock_prt_hhi)
    struct_color = RED if struct_overall == "High" else (
        ACCENT if struct_overall == "Medium" else GREEN)
    struct_style = {
        "fontFamily": MONO, "fontSize": "22px", "fontWeight": "800",
        "letterSpacing": "0.10em", "marginBottom": "4px", "color": struct_color,
    }

    # Resilience scores
    base_s = _norm([BASE["supplier_shares"].get(s, 0) for s in SUPPLIERS_ORDER])
    base_p = _norm([BASE["port_shares"].get(p, 0)     for p in PORTS_FULL])
    sr = np.array([SUPPLIER_RISK.get(s, 0.5) for s in SUPPLIERS_ORDER])
    pr = np.array([PORT_RISK.get(p, 0.3)     for p in PORTS_FULL])
    post_s = _norm([post_sup.get(s, 0) for s in SUPPLIERS_ORDER])
    post_p = _norm([post_prt.get(p, 0) for p in PORTS_FULL])

    base_sc = score_state(base_s, base_p, sr, pr, base_s, base_p)
    loss_penalty = shock_sim["loss_pct"] * 0.8
    shock_sc = max(0, round(
        score_state(post_s, post_p, sr, pr, base_s, base_p) - loss_penalty, 1))
    opt_sc = resilience_score(opt["comps"])
    delta_pts = opt_sc - shock_sc

    # FIGURE 1 — Timeline
    days = shock_sim["days"]
    b_total = base_sim["total_t"]
    s_total = shock_sim["total_t"]
    b_smooth = pd.Series(b_total).rolling(7, min_periods=1).mean().tolist()
    s_smooth = pd.Series(s_total).rolling(7, min_periods=1).mean().tolist()

    fig_tl = go.Figure()
    fig_tl.add_trace(go.Scatter(
        x=days, y=b_smooth, name="Baseline",
        line=dict(color=DARK2, width=1.5, dash="dot"), opacity=0.45,
        hovertemplate="Day %{x}: %{y:,.0f} TEU<extra>Baseline</extra>",
    ))
    fig_tl.add_trace(go.Scatter(
        x=days, y=s_smooth, name="Shocked",
        line=dict(color=RED, width=2.5),
        hovertemplate="Day %{x}: %{y:,.0f} TEU<extra>Shocked</extra>",
    ))
    fig_tl.add_vline(x=30, line_width=1.5, line_dash="dash", line_color="#bbb")
    fig_tl.add_annotation(
        x=28, y=s_smooth[28], text="Shock lands",
        showarrow=True, arrowhead=2, ax=-30, ay=30,
        arrowcolor=RED, font=dict(color=RED, size=11, family=SANS))
    y_gap = (b_smooth[90] + s_smooth[90]) / 2
    fig_tl.add_annotation(
        x=90, y=y_gap, text="Gap = daily import loss",
        showarrow=False, font=dict(color=MUTED, size=11, family=SANS))
    fig_tl.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=CARD, margin=dict(l=46, r=16, t=6, b=36),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99,
                    bgcolor="rgba(255,255,255,0.8)", bordercolor="#eeeeee",
                    borderwidth=1,
                    font=dict(size=12, family=SANS, color=TEXT)),
        xaxis=dict(title=dict(text="Day", font=dict(size=12, color=MUTED, family=SANS)),
                   showgrid=False, zeroline=False,
                   tickfont=dict(size=11, color=MUTED, family=SANS)),
        yaxis=dict(title=dict(text="TEU / day",
                              font=dict(size=12, color=MUTED, family=SANS)),
                   gridcolor=GRID, showgrid=True, zeroline=False,
                   tickfont=dict(size=11, color=MUTED, family=SANS)),
        autosize=True, font=dict(family=SANS),
    )

    # FIGURE 2 — Supplier Allocation (FIX: barmode=group, not overlay)
    base_vals = [BASE["supplier_shares"].get(s, 0) * 100 for s in SUPPLIERS_ORDER]
    opt_vals  = [float(opt["ss"][i]) * 100 for i in range(len(SUPPLIERS_ORDER))]
    fig_sup = go.Figure()
    fig_sup.add_trace(go.Bar(
        y=SUPPLIERS_ORDER, x=base_vals, name="Baseline",
        orientation="h", marker_color=DARK2, opacity=0.45, marker_line_width=0,
        hovertemplate="%{y}: %{x:.1f}%<extra>Baseline</extra>"))
    fig_sup.add_trace(go.Bar(
        y=SUPPLIERS_ORDER, x=opt_vals, name="Optimized",
        orientation="h", marker_color=ACCENT, marker_line_width=0,
        hovertemplate="%{y}: %{x:.1f}%<extra>Optimized</extra>"))
    fig_sup.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        barmode="group",
        margin=dict(l=6, r=16, t=6, b=36),
        xaxis=dict(title=dict(text="Share (%)",
                              font=dict(size=12, color=MUTED, family=SANS)),
                   gridcolor=GRID, showgrid=True, zeroline=False,
                   tickfont=dict(size=11, color=MUTED, family=SANS)),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", showgrid=False, autorange="reversed",
                   tickfont=dict(size=12, color=TEXT, family=SANS)),
        legend=dict(x=0.98, y=0.02, xanchor="right", yanchor="bottom",
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11, family=SANS, color=TEXT)),
        autosize=True, font=dict(family=SANS),
    )

    # FIGURE 3 — Port donut
    opt_ps = [float(opt["ps"][0]), float(opt["ps"][1])]
    la_pct_v = opt_ps[0] * 100
    fig_port = go.Figure()
    fig_port.add_trace(go.Pie(
        labels=[PORT_LABEL[PORTS_FULL[0]], PORT_LABEL[PORTS_FULL[1]]],
        values=opt_ps, hole=0.62,
        marker_colors=[DARK2, ACCENT],
        textinfo="label+percent", textposition="outside",
        textfont=dict(size=11, family=SANS),
        hovertemplate="%{label}: %{percent:.1%}<extra></extra>",
        showlegend=False, pull=[0.04, 0]))
    fig_port.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        margin=dict(l=10, r=10, t=5, b=5),
        annotations=[dict(
            text=f"<b>{la_pct_v:.0f}%</b><br><span style='font-size:10px'>LA opt.</span>",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
            font=dict(size=16, family=MONO, color=TEXT))],
        height=200, width=200, autosize=False,
    )

    # Resilience card
    def score_block(val, lbl, color):
        return html.Div([
            html.Div(str(val), style={
                "fontFamily": MONO, "fontSize": "2rem", "fontWeight": "700",
                "color": color, "lineHeight": "1", "letterSpacing": "-0.02em"}),
            html.Div(lbl, style={
                "fontSize": "11px", "color": "#666", "marginTop": "4px",
                "letterSpacing": "0.08em", "textTransform": "uppercase",
                "fontFamily": SANS}),
        ], style={"marginBottom": "10px"})

    _hr = html.Hr(style={"border": "none", "borderTop": "1px solid #333333",
                          "margin": "8px 0"})

    if opt_sc >= base_sc:
        recovery_msg = "Optimizer fully restores resilience. Reallocation eliminates the shock impact."
    elif opt_sc >= shock_sc + 10:
        recovery_msg = "Optimizer partially recovers resilience, the best possible given current constraints."
    else:
        recovery_msg = "Severe disruption. Even optimal reallocation cannot fully compensate."

    resilience_children = [
        html.Div([
            score_block(base_sc,  "PRE-SHOCK",        "#ffffff"),
            _hr,
            score_block(shock_sc, "UNDER DISRUPTION", RED),
            _hr,
            score_block(opt_sc,   "BEST ACHIEVABLE",  ACCENT),
            html.Div(recovery_msg, style={
                "fontSize": "11px", "color": "#888888", "marginTop": "6px",
                "fontFamily": SANS, "lineHeight": "1.4", "fontStyle": "italic"}),
        ], style={"textAlign": "center", "width": "100%"}),
    ]

    # Recommendation bullets
    base_sv = {s: BASE["supplier_shares"].get(s, 0) for s in SUPPLIERS_ORDER}
    opt_sv  = {SUPPLIERS_ORDER[i]: float(opt["ss"][i]) for i in range(len(SUPPLIERS_ORDER))}
    deltas = sorted([(s, opt_sv[s] - base_sv[s]) for s in SUPPLIERS_ORDER],
                     key=lambda x: abs(x[1]), reverse=True)

    bullets = []
    for s, d in deltas[:3]:
        if abs(d) < 0.004: continue
        arrow  = "↑" if d > 0 else "↓"
        action = "Increase" if d > 0 else "Reduce"
        bullets.append(html.Div([
            html.Span(arrow, style={
                "color": ACCENT, "marginRight": "8px", "fontWeight": "800",
                "fontSize": "18px", "lineHeight": "1", "fontFamily": MONO}),
            html.Div([
                html.Span(f"{action} ", style={"fontWeight": "600", "color": "#ffffff"}),
                html.Span(s, style={"color": ACCENT}),
                html.Span(f"  {base_sv[s]:.0%} → {opt_sv[s]:.0%}",
                           style={"color": MUTED2, "marginLeft": "4px"}),
            ], style={"fontSize": "14px", "color": "#ffffff", "fontFamily": SANS,
                      "lineHeight": "1.3"}),
        ], style={"display": "flex", "alignItems": "flex-start", "marginBottom": "12px"}))

    base_la = BASE["port_shares"].get(PORTS_FULL[0], 0.53)
    opt_la  = float(opt["ps"][0])
    d_la    = opt_la - base_la
    if abs(d_la) > 0.01:
        arr = "↑" if d_la > 0 else "↓"
        bullets.append(html.Div([
            html.Span(arr, style={
                "color": ACCENT, "marginRight": "8px", "fontWeight": "800",
                "fontSize": "18px", "lineHeight": "1", "fontFamily": MONO}),
            html.Div([
                html.Span("Port of LA load", style={"color": "#ffffff"}),
                html.Span(f"  {base_la:.0%} → {opt_la:.0%}",
                           style={"color": MUTED2, "marginLeft": "4px"}),
            ], style={"fontSize": "14px", "fontFamily": SANS, "lineHeight": "1.3"}),
        ], style={"display": "flex", "alignItems": "flex-start", "marginBottom": "12px"}))

    bullets.append(html.Div(
        f"Resilience improves by {delta_pts:+.1f} points.",
        style={"marginTop": "14px", "fontSize": "14px", "color": ACCENT,
               "fontWeight": "700", "fontFamily": MONO}))

    # Risk label
    if mal_pct >= 35 and la_pct >= 35:
        risk_txt, risk_col = "CRITICAL", "#c0392b"
    elif mal_pct >= 35 or la_pct >= 35 or (mal_pct >= 15 and la_pct >= 15):
        risk_txt, risk_col = "HIGH", RED
    elif mal_pct >= 20 or la_pct >= 20:
        risk_txt, risk_col = "MODERATE", ACCENT
    else:
        risk_txt, risk_col = "LOW", GREEN

    risk_style = {
        "fontFamily": MONO, "fontSize": "22px", "fontWeight": "800",
        "letterSpacing": "0.12em", "textAlign": "center", "padding": "8px 0",
        "color": risk_col,
    }

    if mal_pct == 0 and la_pct == 0:
        narrative = (
            f"No active disruptions right now. "
            f"System resilience is **{base_sc} / 100**. "
            f"Use the sliders above or pick a preset shock to try a scenario."
        )
    else:
        if opt_sc >= base_sc:
            tail = f"The optimizer recovers it to **{opt_sc}** by reallocating supplier shares."
        else:
            tail = (f"The optimizer achieves a best-case recovery to **{opt_sc}**. "
                    f"Full recovery is not possible under these conditions.")
        narrative = (
            f"**{mal_pct}%** Malaysia disruption + **{la_pct}%** LA capacity loss causes "
            f"**{shock_sim['loss_pct']:.1f}%** throughput loss, dropping resilience from "
            f"**{base_sc}** to **{shock_sc}**. {tail}"
        )

    return (
        fig_tl, fig_sup, fig_port,
        resilience_children, bullets,
        risk_txt, risk_style, narrative,
        struct_overall, struct_style,
        struct_sup_lbl, f"HHI = {shock_sup_hhi:.3f}",
        struct_mal_lbl, f"{shock_mal_share:.1%}",
        struct_prt_lbl, f"HHI = {shock_prt_hhi:.3f}",
    )


# =============================================================================
# OPTIMIZER TAB CALLBACKS
# =============================================================================
@app.callback(
    Output("opt-pref", "value"),
    Input("opt-preset", "value"),
)
def _opt_preset_to_pref(preset):
    return {"cost": 20, "balanced": 50, "resilience": 80}.get(preset, 50)


@app.callback(
    Output("opt-custom-year-wrap", "style"),
    Input("opt-baseline-mode", "value"),
)
def _toggle_custom_year(mode):
    if mode == "custom":
        return {"display": "block", "marginTop": "8px"}
    return {"display": "none"}


@app.callback(
    Output("opt-hist-wrap", "style"),
    Input("opt-use-hist", "value"),
)
def _toggle_hist(on):
    if on and "on" in on and TRADE_AVAILABLE:
        return {"display": "block"}
    return {"display": "none"}


@app.callback(
    Output("opt-results", "children"),
    Input("opt-run", "n_clicks"),
    State("opt-pref", "value"),
    State("opt-cap", "value"),
    State("opt-cap-mult", "value"),
    State("opt-demand", "value"),
    State("opt-baseline-mode", "value"),
    State("opt-custom-year", "value"),
    State("opt-use-hist", "value"),
    State("opt-window", "value"),
    State("opt-strength", "value"),
    State("opt-volw", "value"),
    State("opt-trendw", "value"),
    State("w-sr", "value"),
    State("w-pr", "value"),
    State("w-sc", "value"),
    State("w-pc", "value"),
    State("w-st", "value"),
)
def _run_optimizer_tab(n, pref, cap, cap_mult, demand, mode, custom_year,
                       use_hist, window, strength, volw, trendw,
                       wsr, wpr, wsc, wpc, wst):
    if not n:
        return html.Div([
            html.Div("Configure inputs on the left, then click Run Optimizer.",
                     style={"color": MUTED, "fontSize": "13px", "fontFamily": SANS,
                            "textAlign": "center", "padding": "60px 20px"}),
        ], style=card({"minHeight": "200px"}))

    # Baseline shares
    if mode == "csv" or not TRADE_AVAILABLE:
        baseline_shares = pd.Series(
            _norm(SUPPLIERS_DF["current_share"].values),
            index=SUPPLIERS_DF["supplier"].tolist(),
        )
        baseline_label = "suppliers.csv baseline"
    elif mode == "latest_year":
        y = get_latest_full_year()
        baseline_shares = get_year_share(y, SUPPLIERS_ORDER)
        baseline_label = f"{y} full-year trade shares"
    elif mode == "l12m":
        baseline_shares = get_latest_12m_share(SUPPLIERS_ORDER)
        baseline_label = "latest 12-month trade shares"
    else:
        baseline_shares = get_year_share(custom_year, SUPPLIERS_ORDER)
        baseline_label = f"{custom_year} full-year trade shares"

    baseline_shares = baseline_shares.reindex(SUPPLIERS_ORDER).fillna(0)

    base_risk = pd.Series(
        [SUPPLIER_RISK.get(s, 0.5) for s in SUPPLIERS_ORDER],
        index=SUPPLIERS_ORDER,
    )
    if use_hist and "on" in use_hist and TRADE_AVAILABLE:
        adj = compute_trade_risk_adjustment(volw, trendw, window)
        eff_risk = (base_risk + strength * adj).clip(0, 1)
    else:
        eff_risk = base_risk.copy()

    sup_df = pd.DataFrame({
        "supplier": SUPPLIERS_ORDER,
        "current_share": baseline_shares.values,
        "risk_score":    eff_risk.values,
        "max_share_default": [min(v * 1.5, 0.80) for v in baseline_shares.values],
    })
    prt_df = pd.DataFrame([
        {"port": p, "current_share": BASE["port_shares"].get(p, 0),
         "port_risk": PORT_RISK.get(p, 0.3),
         "capacity_teu": BASE["port_capacities"].get(p, 8_000_000)}
        for p in PORTS_FULL
    ])
    cm = _cost_matrix(sup_df, prt_df)

    weights = {"wsr": wsr, "wpr": wpr, "wsc": wsc, "wpc": wpc, "wst": wst}

    try:
        result = run_optimizer(
            sup_df, prt_df, cm,
            dem=float(demand), cap=float(cap), rp=int(pref),
            weights=weights, effective_sup_risk=eff_risk.values,
            cap_mult=float(cap_mult),
        )
    except Exception as e:
        return html.Div(f"Optimizer error: {e}",
                        style=card({"color": RED, "fontFamily": SANS, "padding": "30px"}))

    comps = result["comps"]
    opt_sup = result["ss"]
    opt_prt = result["ps"]

    # Current (pre-opt) baseline metrics
    cur_s = _norm([BASE["supplier_shares"].get(s, 0) for s in SUPPLIERS_ORDER])
    cur_p = np.array([BASE["port_shares"].get(p, 0) for p in PORTS_FULL])
    cur_sup_hhi = float(np.sum(cur_s ** 2))
    cur_prt_hhi = float(np.sum(cur_p ** 2))
    cur_mal = float(BASE["supplier_shares"].get("Malaysia", 0))
    cur_x = np.outer(cur_s, cur_p)
    cur_ac = float(np.sum(cur_x * cm))
    cur_ann = cur_ac * float(demand)

    opt_sup_hhi = comps["sc"]
    opt_prt_hhi = comps["pc"]
    opt_mal = float(opt_sup[SUPPLIERS_ORDER.index("Malaysia")])
    opt_ac = comps["cost"]
    opt_ann = comps["ann"]

    d_sup_hhi = opt_sup_hhi - cur_sup_hhi
    d_prt_hhi = opt_prt_hhi - cur_prt_hhi
    d_mal     = opt_mal - cur_mal
    d_ann     = opt_ann - cur_ann

    # Risk panel
    sup_lbl  = risk_band(cur_sup_hhi, 0.22, 0.30)
    mal_lbl  = risk_band(cur_mal, 0.30, 0.50)
    prt_lbl  = risk_band(cur_prt_hhi, 0.50, 0.55)
    overall  = overall_structural_risk(cur_sup_hhi, cur_mal, cur_prt_hhi)
    score    = resilience_score(comps)

    def delta_span(val, good_is_negative=True, fmt="{:+.3f}"):
        is_good = (val < 0) if good_is_negative else (val > 0)
        color = GREEN if is_good else (RED if abs(val) > 1e-9 else MUTED)
        return html.Span(fmt.format(val), style={
            "fontFamily": MONO, "fontSize": "11px", "color": color,
            "fontWeight": "700", "marginLeft": "6px",
        })

    risk_row = html.Div([
        kpi_card("Overall Structural Risk", overall,
                 "Composite of supplier, Malaysia, port concentration",
                 color=RED if overall == "High" else (ACCENT if overall == "Medium" else GREEN)),
        kpi_card("Supplier Concentration", sup_lbl, f"HHI = {cur_sup_hhi:.3f}"),
        kpi_card("Malaysia Dependence", mal_lbl, f"{cur_mal:.1%}"),
        kpi_card("Port Concentration", prt_lbl, f"HHI = {cur_prt_hhi:.3f}"),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)",
              "gap": "12px", "marginBottom": "12px"})

    delta_row = html.Div([
        html.Div([
            label("Supplier HHI"),
            html.Div([
                html.Span(f"{opt_sup_hhi:.3f}", style={
                    "fontFamily": MONO, "fontSize": "22px", "fontWeight": "800", "color": TEXT}),
                delta_span(d_sup_hhi, good_is_negative=True),
            ]),
            html.Div(f"Current: {cur_sup_hhi:.3f}", style={
                "fontSize": "11px", "color": MUTED, "marginTop": "4px", "fontFamily": SANS}),
        ], style=card({"borderLeft": f"4px solid {ACCENT}"})),
        html.Div([
            label("Port HHI"),
            html.Div([
                html.Span(f"{opt_prt_hhi:.3f}", style={
                    "fontFamily": MONO, "fontSize": "22px", "fontWeight": "800", "color": TEXT}),
                delta_span(d_prt_hhi, good_is_negative=True),
            ]),
            html.Div(f"Current: {cur_prt_hhi:.3f}", style={
                "fontSize": "11px", "color": MUTED, "marginTop": "4px", "fontFamily": SANS}),
        ], style=card({"borderLeft": f"4px solid {BLUE}"})),
        html.Div([
            label("Malaysia Share"),
            html.Div([
                html.Span(f"{opt_mal:.1%}", style={
                    "fontFamily": MONO, "fontSize": "22px", "fontWeight": "800", "color": TEXT}),
                delta_span(d_mal, good_is_negative=True, fmt="{:+.1%}"),
            ]),
            html.Div(f"Current: {cur_mal:.1%}", style={
                "fontSize": "11px", "color": MUTED, "marginTop": "4px", "fontFamily": SANS}),
        ], style=card({"borderLeft": f"4px solid {RED}"})),
        html.Div([
            label("Annual Transport Cost"),
            html.Div([
                html.Span(f"${opt_ann/1e6:.1f}M", style={
                    "fontFamily": MONO, "fontSize": "22px", "fontWeight": "800", "color": TEXT}),
                delta_span(d_ann / 1e6, good_is_negative=True, fmt="{:+.1f}M"),
            ]),
            html.Div(f"Current: ${cur_ann/1e6:.1f}M", style={
                "fontSize": "11px", "color": MUTED, "marginTop": "4px", "fontFamily": SANS}),
        ], style=card({"borderLeft": f"4px solid {GREEN}"})),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)",
              "gap": "12px", "marginBottom": "12px"})

    # Summary card
    sup_dir  = "reduced" if d_sup_hhi < 0 else "increased"
    prt_dir  = "reduced" if d_prt_hhi < 0 else "increased"
    mal_dir  = "reduced" if d_mal < 0 else "increased"
    cost_dir = "increased" if d_ann > 0 else "reduced"
    summary = html.Div([
        card_title("What Changed?"),
        html.Div([
            "The optimized strategy ",
            html.Strong(f"{sup_dir} supplier concentration"), ", ",
            html.Strong(f"{prt_dir} port concentration"), ", and ",
            html.Strong(f"{mal_dir} dependence on Malaysia"), ". ",
            "Compared with the current network, annual transport cost ",
            html.Strong(f"{cost_dir} by ${abs(d_ann):,.0f}"), ".",
        ], style={"fontSize": "13px", "color": TEXT, "fontFamily": SANS,
                  "lineHeight": "1.6"}),
    ], style=card({"marginBottom": "12px"}))

    # KPIs
    kpis = html.Div([
        kpi_card("Strategy", strategy_label(pref),
                 f"Preference = {pref}", color=DARK),
        kpi_card("Cost / TEU", f"${opt_ac:.2f}",
                 "Optimized weighted average"),
        kpi_card("Annual Cost", f"${opt_ann/1e6:.2f}M",
                 f"Demand = {demand:,.0f} TEU"),
        kpi_card("Resilience Score", f"{score}/100",
                 "Higher is better", color=ACCENT),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)",
              "gap": "12px", "marginBottom": "12px"})

    # Charts
    chart_row = html.Div([
        html.Div([
            card_title("Baseline vs Optimized — Supplier"),
            card_subtitle(f"Baseline: {baseline_label}"),
            dcc.Graph(figure=plot_baseline_vs_optimized_bars(
                SUPPLIERS_ORDER, baseline_shares.values, opt_sup),
                config=GRAPH_CFG, style={"height": "280px"}),
        ], style=card({"flex": "1"})),
        html.Div([
            card_title("Current vs Optimized — Port"),
            card_subtitle("Share of total TEU throughput."),
            dcc.Graph(figure=plot_port_bars(PORTS_FULL, cur_p, opt_prt),
                      config=GRAPH_CFG, style={"height": "280px"}),
        ], style=card({"flex": "1"})),
    ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"})

    obj_chart = html.Div([
        card_title("Objective Component Breakdown"),
        card_subtitle("Component values contributing to the solver's objective."),
        dcc.Graph(figure=plot_objective_breakdown(comps),
                  config=GRAPH_CFG, style={"height": "260px"}),
    ], style=card({"marginBottom": "12px"}))

    # Tables
    sup_table = pd.DataFrame({
        "supplier": SUPPLIERS_ORDER,
        "baseline": [f"{v:.2%}" for v in baseline_shares.values],
        "optimized": [f"{v:.2%}" for v in opt_sup],
        "change":   [f"{(opt_sup[i] - baseline_shares.values[i]):+.2%}"
                     for i in range(len(SUPPLIERS_ORDER))],
        "base_risk": [f"{v:.2f}" for v in base_risk.values],
        "effective_risk": [f"{v:.2f}" for v in eff_risk.values],
    })
    prt_table = pd.DataFrame({
        "port":      [PORT_LABEL[p] for p in PORTS_FULL],
        "current":   [f"{v:.2%}" for v in cur_p],
        "optimized": [f"{v:.2%}" for v in opt_prt],
        "opt TEU":   [f"{v * float(demand):,.0f}" for v in opt_prt],
    })
    flow_share = pd.DataFrame(result["x"], index=SUPPLIERS_ORDER,
                               columns=[PORT_LABEL[p] for p in PORTS_FULL])
    flow_teu = flow_share * float(demand)
    flow_share_disp = (flow_share * 100).round(2).astype(str) + "%"
    flow_teu_disp   = flow_teu.round(0).astype(int).astype(str)

    def _table(df, index=False):
        cols = (["index"] if index else []) + list(df.columns)
        df_show = df.reset_index().rename(columns={"index": ""}) if index else df
        return html.Table(
            [html.Thead(html.Tr([html.Th(c, style={
                "fontSize": "11px", "fontFamily": SANS, "color": MUTED,
                "textTransform": "uppercase", "letterSpacing": "0.06em",
                "textAlign": "left", "padding": "6px 8px",
                "borderBottom": f"1px solid {BORDER}",
            }) for c in df_show.columns]))] +
            [html.Tbody([
                html.Tr([html.Td(df_show.iloc[i][c], style={
                    "fontSize": "12px", "fontFamily": MONO, "color": TEXT,
                    "padding": "6px 8px",
                    "borderBottom": f"1px solid {GRID}"})
                         for c in df_show.columns])
                for i in range(len(df_show))
            ])],
            style={"width": "100%", "borderCollapse": "collapse",
                   "fontFamily": SANS})

    tables = html.Div([
        html.Div([
            card_title("Supplier Allocation Table"),
            _table(sup_table),
        ], style=card({"flex": "1"})),
        html.Div([
            card_title("Port Allocation Table"),
            _table(prt_table),
        ], style=card({"flex": "1"})),
    ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"})

    flow_tables = html.Div([
        html.Div([
            card_title("Supplier → Port Flow Shares"),
            card_subtitle("Each cell is the share of total demand routed on that lane."),
            _table(flow_share_disp.reset_index().rename(columns={"index": "Supplier"})),
        ], style=card({"flex": "1"})),
        html.Div([
            card_title("Supplier → Port Annual TEU"),
            card_subtitle("Physical TEU volume implied by the flow-share allocation."),
            _table(flow_teu_disp.reset_index().rename(columns={"index": "Supplier"})),
        ], style=card({"flex": "1"})),
    ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"})

    exec_summary = html.Div([
        html.Div("EXECUTIVE SUMMARY", style={
            "fontSize": "9px", "fontWeight": "600", "color": ACCENT,
            "letterSpacing": "2px", "fontFamily": SANS, "marginBottom": "10px",
        }),
        html.Div([
            html.Div([html.Strong("Strategy: "), strategy_label(pref)]),
            html.Div([html.Strong("Baseline source: "), baseline_label]),
            html.Div([html.Strong("Historical risk adjustment: "),
                      "On" if (use_hist and "on" in use_hist and TRADE_AVAILABLE) else "Off"]),
            html.Div([html.Strong("Supplier cap: "), f"{cap:.0%}"]),
            html.Div([html.Strong("Port capacity assumption: "), f"×{cap_mult:.2f}"]),
            html.Div([html.Strong("Target demand: "), f"{demand:,.0f} TEU"]),
            html.Div(style={"height": "8px"}),
            html.Div([html.Strong("Avg cost / TEU: "), f"${opt_ac:.2f}"]),
            html.Div([html.Strong("Annual cost: "), f"${opt_ann:,.0f}"]),
            html.Div([html.Strong("Resilience score: "), f"{score}/100"]),
        ], style={"fontSize": "13px", "color": "#ffffff",
                  "fontFamily": SANS, "lineHeight": "1.9"}),
    ], style=dark_card({"borderLeft": f"4px solid {ACCENT}"}))

    return html.Div([
        risk_row,
        delta_row,
        summary,
        kpis,
        chart_row,
        obj_chart,
        tables,
        flow_tables,
        exec_summary,
    ])


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    app.run(debug=False, port=port)

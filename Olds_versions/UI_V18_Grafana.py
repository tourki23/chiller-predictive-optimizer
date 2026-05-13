import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import xgboost as xgb
from sqlalchemy import create_engine, text
from datetime import date, timedelta
import warnings
import joblib
import tensorflow as tf
import os
from flask_caching import Cache
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import base64

# --- NOUVEAU : Imports pour le monitoring Prometheus ---
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from flask import Response

warnings.filterwarnings('ignore')

# ===========================================================
# CONFIGURATION DYNAMIQUE DES MODÈLES
# ===========================================================
MODELS_CONFIG = {
    "physiciens": {
        "LSTM 128u Attention": "lstm_128u_Attention",
        "LSTM 128u NoAttention": "lstm_128u_NoAttention",
        "LSTM 64u Attention": "lstm_64u_Attention",
        "LSTM 64u NoAttention": "lstm_64u_NoAttention"
    },
    "energetiques": {
        "XGBoost Depth 3": "modele_financier_depth_3.json",
        "XGBoost Depth 5": "modele_financier_depth_5.json",
        "XGBoost Depth 7": "modele_financier_depth_7.json",
        "XGBoost Depth 9": "modele_financier_depth_9.json"
    }
}

DB_URL   = "sqlite:///axima_poc.db"
FEATURES = ['T_ext', 'T_int', 'Etat_Compresseur', 'T_int_lag_1',
            'day_sin', 'day_cos', 'hour_sin', 'hour_cos']
TIME_STEPS   = 12
CONSIGNE     = 7.0
HYSTERESIS_H = 0.5
HYSTERESIS_L = 0.3
INTERVAL_MS  = 600

# ===========================================================
# COULEURS PRINCIPALES & HARMONISÉES
# ===========================================================
couleur_ia  = "#5BD4F2"  # Cyan doux
couleur_opt = "#2ECC71"  # Vert frais, émeraude, non agressif

# ===========================================================
# STYLE UI — DASHBOARD PROFESSIONNEL
# ===========================================================
APP_BG = "#050505"         # Noir beaucoup plus dense et profond
CARD_BG = "#0D1117"        # Noir ardoise très sombre pour les cartes
CARD_BG_SOFT = "#161B22"
BORDER_SOFT = "rgba(255,255,255,0.08)"
TEXT_MUTED = "#A8B3C7"
TEXT_LIGHT = "#F4F7FB"
ACCENT_ORANGE = "#F8A706"

STYLE_PAGE = {
    "background": f"linear-gradient(135deg, {APP_BG} 0%, #0a0a0a 45%, #000000 100%)",
    "minHeight": "100vh",
    "padding": "18px 22px 28px 22px",
    "fontFamily": "Inter, Segoe UI, Roboto, Arial, sans-serif"
}

STYLE_CARD = {
    "backgroundColor": CARD_BG,
    "border": f"1px solid {BORDER_SOFT}",
    "borderRadius": "18px",
    "boxShadow": "0 12px 35px rgba(0,0,0,0.40)"
}

STYLE_CARD_HEADER = {
    "backgroundColor": "rgba(255,255,255,0.03)",
    "borderBottom": f"1px solid {BORDER_SOFT}",
    "fontWeight": "700",
    "letterSpacing": "0.2px",
    "color": TEXT_LIGHT
}

STYLE_GRAPH_CARD = {
    "backgroundColor": CARD_BG,
    "border": f"1px solid {BORDER_SOFT}",
    "borderRadius": "18px",
    "padding": "10px",
    "boxShadow": "0 12px 35px rgba(0,0,0,0.30)",
    "marginBottom": "14px"
}

STYLE_TABLE_HEADER = {
    'backgroundColor': CARD_BG_SOFT,
    'color': TEXT_LIGHT,
    'fontWeight': 'bold',
    'textAlign': 'center',
    'border': '1px solid rgba(255,255,255,0.08)'
}

STYLE_TABLE_DATA = {
    'backgroundColor': CARD_BG,
    'color': '#EAF0F8',
    'border': '1px solid rgba(255,255,255,0.05)'
}

STYLE_TABLE_CELL = {
    'textAlign': 'center',
    'padding': '10px',
    'fontFamily': 'Inter, Segoe UI, Arial',
    'fontSize': '13px'
}

STYLE_TABLE = {
    'overflowX': 'auto',
    'borderRadius': '16px',
    'border': '1px solid rgba(255,255,255,0.08)',
    'boxShadow': '0 12px 35px rgba(0,0,0,0.22)'
}

# Variables globales pour le moteur de calcul lourd
scaler_X = None
modele_physicien = None
modele_financier = None

print("=" * 50)
print("🌐 JUMEAU NUMÉRIQUE V7 — MLOps Dashboard & Data")
print("=" * 50)

# ===========================================================
# CHARGEMENT INITIAL & BDD
# ===========================================================
engine   = create_engine(DB_URL)
executor = ThreadPoolExecutor(max_workers=2)

with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS optimisations_data (
            timestamp TIMESTAMP, t_int_opt FLOAT, etat_opt FLOAT,
            pwr_opt FLOAT, pente_chauffe FLOAT, pente_froid FLOAT
        )
    """))
    conn.execute(text("DELETE FROM optimisations_data"))

print("📥 Chargement en mémoire des données de simulation 2026...")
df_sim = (pd.read_sql_table('simulation_data_2026_table', engine)
            .sort_values('timestamp')
            .reset_index(drop=True))
df_sim['timestamp'] = pd.to_datetime(df_sim['timestamp'])
df_sim['_date'] = df_sim['timestamp'].dt.date
date_index = df_sim.groupby('_date').apply(lambda g: (g.index[0], g.index[-1])).to_dict()
print(f"   ✅ {len(df_sim):,} lignes chargées")


# ===========================================================
# INIT DASH APP & PROMETHEUS METRICS
# ===========================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)

# Définition des sondes Prometheus
PREDICTION_COUNT = Counter('app_lstm_predictions_total', 'Nombre total de predictions LSTM')
PREDICTION_LATENCY = Histogram('app_lstm_prediction_latency_seconds', 'Temps mis pour une prediction LSTM')

# Endpoint pour que Prometheus vienne lire les données
@app.server.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


# ===========================================================
# MOTEUR DE CALCUL
# ===========================================================
def _build_sequences_vectorized(idx_start: int, idx_end: int) -> np.ndarray:
    raw_block = df_sim.iloc[idx_start - TIME_STEPS: idx_end][FEATURES].values
    n_seq = idx_end - idx_start
    scaled_block = scaler_X.transform(raw_block)
    seqs = np.lib.stride_tricks.sliding_window_view(
        scaled_block, window_shape=(TIME_STEPS, len(FEATURES))
    )[:n_seq, 0]
    return seqs.astype(np.float32)

def _compute_optimisation(t_preds, etat_reel_jour, etat_init, t_init):
    t_opts, etat_opts, pente_chauffes, pente_froids = [], [], [], []
    t_opt_current = t_init
    etat_opt_current = etat_init
    pente_chauffe_list, pente_froid_list = [], []
    t_pred_prec = t_init

    for k in range(len(t_preds)):
        t_pred_current = t_preds[k]
        pente_obs = t_pred_current - t_pred_prec
        t_pred_prec = t_pred_current

        if etat_reel_jour[k] == 0:
            pente_chauffe_list.append(pente_obs)
        else:
            pente_froid_list.append(pente_obs)

        p_c = np.mean(pente_chauffe_list[-10:]) if pente_chauffe_list else 0.04
        p_f = np.mean(pente_froid_list[-10:]) if pente_froid_list else -0.12

        etat_opt_current = (
            1.0 if t_opt_current > CONSIGNE + HYSTERESIS_H
            else (0.0 if t_opt_current < CONSIGNE - HYSTERESIS_L else etat_opt_current)
        )

        t_opt_next = t_opt_current + (p_f if etat_opt_current == 1.0 else p_c)

        t_opts.append(t_opt_current)
        etat_opts.append(etat_opt_current)
        pente_chauffes.append(p_c)
        pente_froids.append(p_f)
        t_opt_current = t_opt_next

    return (
        np.array(t_opts),
        np.array(etat_opts),
        np.array(pente_chauffes),
        np.array(pente_froids)
    )

def _write_optimisation_async(df_opt):
    try:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM optimisations_data"))
        df_opt.to_sql('optimisations_data', engine, if_exists='append', index=False)
    except Exception:
        pass


# 👇 On ajoute le décorateur Prometheus pour mesurer le temps de cette fonction
@PREDICTION_LATENCY.time()
def compute_day_data(target_date_str):
    if scaler_X is None:
        return None

    target_date = pd.to_datetime(target_date_str).date()

    if target_date not in date_index:
        return None

    i_start_raw, i_end_raw = date_index[target_date]
    idx_start = max(i_start_raw, TIME_STEPS)
    idx_end = min(idx_start + 144, i_end_raw + 1)
    df_day = df_sim.iloc[idx_start:idx_end].copy()

    seqs = _build_sequences_vectorized(idx_start, idx_end)

    # 👇 On incrémente le compteur Prometheus de prédictions
    PREDICTION_COUNT.inc()

    t_preds = modele_physicien.predict(
        seqs,
        batch_size=64,
        verbose=0
    ).flatten()

    df_day['t_int_pred'] = t_preds

    df_day['pwr_pred_ia'] = modele_financier.predict(
        df_day[['t_int_pred', 'Etat_Compresseur']].rename(columns={'t_int_pred': 'T_int'})
    )

    etat_init = float(df_sim.iloc[idx_start - 1]['Etat_Compresseur'])
    t_init = float(df_sim.iloc[idx_start - 1]['T_int'])

    t_opts, etat_opts, pente_c, pente_f = _compute_optimisation(
        t_preds,
        df_day['Etat_Compresseur'].values,
        etat_init,
        t_init
    )

    df_day['t_int_opt'] = t_opts
    df_day['etat_opt'] = etat_opts
    df_day['pente_chauffe'] = pente_c
    df_day['pente_froid'] = pente_f

    df_day['pwr_opt'] = modele_financier.predict(
        pd.DataFrame({
            'T_int': t_opts,
            'Etat_Compresseur': etat_opts
        })
    )

    df_opt_db = df_day[
        ['timestamp', 't_int_opt', 'etat_opt', 'pwr_opt', 'pente_chauffe', 'pente_froid']
    ].copy()

    df_opt_db['timestamp'] = df_opt_db['timestamp'].astype(str)

    executor.submit(_write_optimisation_async, df_opt_db)

    df_day['timestamp'] = df_day['timestamp'].astype(str)

    return df_day.to_dict('records')

_day_cache = {}
_cache_lock = threading.Lock()

def get_day_data_cached(date_str):
    with _cache_lock:
        if date_str not in _day_cache:
            _day_cache[date_str] = compute_day_data(date_str)
        return _day_cache[date_str]

# ===========================================================
# VISUALISATION
# ===========================================================
def _make_common_layout(x_range, ui_rev, extra=None):
    layout = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            family="Inter, Segoe UI, Arial",
            color="#EAF0F8",
            size=12
        ),

        xaxis=dict(
            range=x_range,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            tickformat="%H:%M",
            autorange=False,
            zeroline=False
        ),

        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.38,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11)
        ),

        margin=dict(l=45, r=25, t=45, b=45),
        hovermode="x unified",
        
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_color="black",
            bordercolor="rgba(0,0,0,0.5)"
        ),
        
        uirevision=ui_rev
    )

    if extra:
        layout.update(extra)

    return layout

def build_figures(df_plot, is_opti_active, target_date, x_range, ref_m1, ref_m2):
    ui_rev = f"{target_date}_{ref_m1}_{ref_m2}"

    if df_plot.empty:
        fig_meteo = go.Figure(layout=_make_common_layout(x_range, ui_rev, {'title': dict(text=f"Météo : {target_date.strftime('%d/%m/%Y')}", x=0.5)}))
        fig_temp = go.Figure(layout=_make_common_layout(x_range, ui_rev, {'title': dict(text=f"Simulation Thermique LSTM — Modèle : {ref_m1}", x=0.01)}))
        fig_pwr = go.Figure(layout=_make_common_layout(x_range, ui_rev, {'title': dict(text=f"Consommation Énergétique XGBoost — Modèle : {ref_m2}", x=0.01)}))
        return fig_meteo, fig_temp, fig_pwr

    diff_r = df_plot['Etat_Compresseur'].diff().fillna(0)
    on_reel = df_plot[diff_r == 1]
    off_reel = df_plot[diff_r == -1]

    fig_meteo = go.Figure(
        data=[
            go.Scatter(x=df_plot['timestamp'], y=df_plot['T_ext'], mode='lines', name="T° Extérieure", line=dict(color="#FF4444"))
        ],
        layout=_make_common_layout(x_range, ui_rev, {'title': dict(text=f"Météo : {target_date.strftime('%d/%m/%Y')}", x=0.5)})
    )

    traces_temp = [
        go.Scatter(x=df_plot['timestamp'], y=[CONSIGNE] * len(df_plot), mode='lines', name="T° Consigne", line=dict(color='gray', dash='dot')),
        go.Scatter(x=df_plot['timestamp'], y=df_plot['T_int'], mode='lines', name="T° Réelle", line=dict(color="#4A84D4", width=2)),
        go.Scatter(x=on_reel['timestamp'], y=on_reel['T_int'], mode='markers', marker=dict(color='#A4E063', size=10), name='ON (Réel)'),
        go.Scatter(x=off_reel['timestamp'], y=off_reel['T_int'], mode='markers', marker=dict(color='#D65F5F', size=10), name='OFF (Réel)'),
        go.Scatter(x=df_plot['timestamp'], y=df_plot['t_int_pred'], mode='lines', name="T° IA", line=dict(color=couleur_ia, dash='dash')),
    ]

    if is_opti_active:
        on_opt = df_plot[df_plot['etat_opt'].diff() == 1]
        off_opt = df_plot[df_plot['etat_opt'].diff() == -1]

        traces_temp += [
            go.Scatter(x=df_plot['timestamp'], y=df_plot['t_int_opt'], mode='lines', name="T° Optimisée", line=dict(color=couleur_opt, width=2)),
            go.Scatter(x=on_opt['timestamp'], y=on_opt['t_int_opt'], mode='markers', marker=dict(color='#A4E063', size=14, symbol='star', line=dict(color='white', width=1)), name='ON (Opti)'),
            go.Scatter(x=off_opt['timestamp'], y=off_opt['t_int_opt'], mode='markers', marker=dict(color='#D65F5F', size=14, symbol='star', line=dict(color='white', width=1)), name='OFF (Opti)'),
        ]

    fig_temp = go.Figure(
        data=traces_temp,
        layout=_make_common_layout(x_range, ui_rev, {'title': dict(text=f"Simulation Thermique LSTM — Modèle : {ref_m1}", x=0.01)})
    )

    fig_pwr = go.Figure(
        data=[
            go.Scatter(x=df_plot['timestamp'], y=df_plot['Puissance_Elec_kW'], mode='lines', name="Réel", line=dict(color="#f89406")),
            go.Scatter(x=df_plot['timestamp'], y=df_plot['pwr_pred_ia'], mode='lines', name=f"IA: {ref_m2}", line=dict(color=couleur_ia, dash='dash')),
        ],
        layout=_make_common_layout(x_range, ui_rev, {'title': dict(text=f"Consommation Énergétique XGBoost — Modèle : {ref_m2}", x=0.01)})
    )

    if is_opti_active:
        fig_pwr.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['pwr_opt'], mode='lines', name="Optimisé", line=dict(color=couleur_opt)))

    return fig_meteo, fig_temp, fig_pwr

def kpi_row_item(label, energy_value, time_value, energy_color, time_color, subtitle=None):
    return html.Div([
        html.Div(label, style={
            "color": TEXT_MUTED,
            "fontSize": "12px",
            "textTransform": "uppercase",
            "letterSpacing": "0.8px"
        }),
        html.Div([
            html.Div(energy_value, style={
                "color": energy_color,
                "fontSize": "25px",
                "fontWeight": "800",
                "lineHeight": "1.15",
            }),
            html.Div(time_value, style={
                "color": time_color,
                "fontSize": "25px",
                "fontWeight": "800",
                "lineHeight": "1.15",
                "marginLeft": "auto" 
            }),
        ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '3px'}),
        html.Div(subtitle or "", style={
            "color": TEXT_MUTED,
            "fontSize": "12px",
            "marginTop": "2px"
        })
    ], style={
        "padding": "13px 14px",
        "backgroundColor": "rgba(255,255,255,0.035)",
        "border": "1px solid rgba(255,255,255,0.07)",
        "borderRadius": "14px",
        "marginBottom": "10px"
    })

def build_kpi(df_plot, is_opti_active):
    if df_plot.empty:
        total_reel_kwh, total_ia_kwh, total_opt_kwh = 0.0, 0.0, 0.0
        total_reel_mn, total_ia_mn, total_opt_mn = 0, 0, 0
    else:
        total_reel_kwh = df_plot['Puissance_Elec_kW'].sum() * (10 / 60)
        total_ia_kwh = df_plot['pwr_pred_ia'].sum() * (10 / 60)
        total_opt_kwh = df_plot['pwr_opt'].sum() * (10 / 60) if is_opti_active else 0
        
        total_reel_mn = (df_plot['Etat_Compresseur'] == 1).sum() * 10
        total_ia_mn = (df_plot['Etat_Compresseur'] == 1).sum() * 10 
        total_opt_mn = (df_plot['etat_opt'] == 1).sum() * 10 if is_opti_active else 0
    
    delta = total_opt_kwh - total_reel_kwh
    delta_text = f"{delta:.1f} kWh" if delta < 0 else f"+{delta:.1f} kWh"
    
    return html.Div([
        kpi_row_item(
            "USINE — 24H",
            f"{total_reel_kwh:.1f} kWh",
            f"{total_reel_mn} mn",
            ACCENT_ORANGE,
            ACCENT_ORANGE,
            "Consommation réelle"
        ),

        kpi_row_item(
            "PRÉDICTION IA",
            f"{total_ia_kwh:.1f} kWh",
            f"{total_ia_mn} mn",
            couleur_ia,
            couleur_ia,
            "Estimation modèle énergétique"
        ),

        html.Div([
            kpi_row_item(
                "OPTIMISATION IA",
                f"{total_opt_kwh:.1f} kWh",
                f"{total_opt_mn} mn",
                couleur_opt,
                couleur_opt,
                f"Écart vs réel : {delta_text}"
            )
        ], style={'display': 'block' if is_opti_active else 'none'})
    ])

# ===========================================================
# FONCTION UTILITAIRE : CHARGEMENT D'IMAGES EN BASE64
# ===========================================================
def get_b64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
            return f"data:image/png;base64,{encoded}"
    return None

# ===========================================================
# LAYOUT PRINCIPAL
# ===========================================================
tab_supervision_content = dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardHeader(
                "🎛️ Control Center",
                style=STYLE_CARD_HEADER
            ),
            dbc.CardBody([
                html.Div("Model Registry", style={
                    "color": TEXT_MUTED,
                    "fontSize": "12px",
                    "textTransform": "uppercase",
                    "letterSpacing": "1px",
                    "marginBottom": "10px"
                }),

                html.Label(
                    "Modèle Physicien — LSTM",
                    className="fw-bold",
                    style={'color': couleur_ia}
                ),

                dcc.Dropdown(
                    id='dropdown-physicien',
                    options=[
                        {'label': k, 'value': k}
                        for k in MODELS_CONFIG['physiciens'].keys()
                    ],
                    value="LSTM 128u Attention",
                    className="mb-3 text-dark"
                ),

                html.Label(
                    "Modèle Énergétique — XGBoost",
                    className="fw-bold",
                    style={'color': couleur_opt}
                ),

                dcc.Dropdown(
                    id='dropdown-energetique',
                    options=[
                        {'label': k, 'value': k}
                        for k in MODELS_CONFIG['energetiques'].keys()
                    ],
                    value="XGBoost Depth 5",
                    className="mb-3 text-dark"
                ),

                html.Div(id="registry-notification"),

                html.Hr(style={"borderColor": "rgba(255,255,255,0.12)"}),

                html.Div("Simulation", style={
                    "color": TEXT_MUTED,
                    "fontSize": "12px",
                    "textTransform": "uppercase",
                    "letterSpacing": "1px",
                    "marginBottom": "10px"
                }),

                html.Label("Jour de simulation — 2026", className="fw-bold"),

                dcc.DatePickerSingle(
                    id='date-picker',
                    date=date(2026, 5, 15),
                    display_format='DD/MM/YYYY',
                    className="mb-3 w-100"
                ),

                dbc.Button(
                    "▶️ Play / Pause",
                    id="btn-play",
                    color="primary", 
                    className="w-100 mb-2 fw-bold"
                ),

                dbc.Button(
                    "🔄 Reset Simulation",
                    id="btn-reset",
                    color="secondary",
                    className="w-100 mb-2 fw-bold"
                ),

                dbc.Button(
                    "⚡ Optimisation énergétique",
                    id="btn-opti",
                    style={"color": couleur_opt, "borderColor": couleur_opt, "backgroundColor": "transparent"},
                    className="w-100 mb-3 fw-bold"
                ),

                html.Div(id="sim-status-notification")

            ])
        ], style=STYLE_CARD),

        html.Br(),

        dbc.Card([
            dbc.CardHeader(
                html.Div([
                    html.Span("⚡ Energy KPIs"),
                    html.Span("⏱️ Time", style={'marginLeft': 'auto'})
                ], style={'display': 'flex', 'alignItems': 'center'}),
                style=STYLE_CARD_HEADER
            ),
            dbc.CardBody(id='kpi-energie')
        ], style=STYLE_CARD)
    ], width=2),

    dbc.Col([
        html.Div(
            dcc.Graph(id='graph-meteo', style={'height': '20vh'}),
            style=STYLE_GRAPH_CARD
        ),
        html.Div(
            dcc.Graph(id='graph-temperature', style={'height': '31vh'}),
            style=STYLE_GRAPH_CARD
        ),
        html.Div(
            dcc.Graph(id='graph-puissance', style={'height': '31vh'}),
            style=STYLE_GRAPH_CARD
        )
    ], width=10)
])

app.layout = dbc.Container([
    dcc.Store(id='store-daily-data'),
    dcc.Store(id='store-idx', data=0),
    dcc.Store(id='store-play-state', data=False),
    dcc.Store(id='store-opti-state', data=False),
    dcc.Store(id='store-models-loaded', data=0),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.Span("❄️", style={
                        "fontSize": "34px",
                        "marginRight": "14px"
                    }),

                    html.Div([
                        html.H2(
                            "Chiller Digital Twin",
                            className="mb-0",
                            style={
                                "color": TEXT_LIGHT,
                                "fontWeight": "800",
                                "letterSpacing": "0.3px",
                                "marginRight": "15px"
                            }
                        ),

                        html.Div(
                            "Predictive Energy Optimization · LSTM/XGBoost · MLOps Registry",
                            style={
                                "color": TEXT_MUTED,
                                "fontSize": "17px",
                                "marginBottom": "0px"
                            }
                        )
                    ], style={"display": "flex", "alignItems": "center"}),
                ], style={
                    "display": "flex",
                    "alignItems": "center"
                }),

            ], style={
                "padding": "22px 26px",
                "background": "linear-gradient(135deg, rgba(91,212,242,0.12), rgba(46,204,113,0.06))",
                "border": f"1px solid {BORDER_SOFT}",
                "borderRadius": "22px",
                "boxShadow": "0 18px 45px rgba(0,0,0,0.30)"
            })
        ], width=9),

        dbc.Col([
            html.Div([
                html.Div("Developed by", style={
                    "color": TEXT_MUTED,
                    "fontSize": "12px",
                    "textTransform": "uppercase",
                    "letterSpacing": "1px"
                }),

                html.Div("TOURKI Mahmoud", style={
                    "color": TEXT_LIGHT,
                    "fontWeight": "700",
                    "fontSize": "17px"
                }),

                html.Div("Industrial AI Dashboard", style={
                    "color": couleur_ia,
                    "fontSize": "13px",
                    "marginTop": "4px"
                })
            ], style={
                "height": "100%",
                "padding": "22px",
                "backgroundColor": CARD_BG,
                "border": f"1px solid {BORDER_SOFT}",
                "borderRadius": "22px",
                "boxShadow": "0 18px 45px rgba(0,0,0,0.25)",
                "display": "flex",
                "flexDirection": "column",
                "justifyContent": "center",
                "textAlign": "right"
            })
        ], width=3)
    ], className="mb-4"),

    dbc.Tabs(
        id="main-tabs",
        active_tab="tab-sup",
        className="custom-tabs",
        children=[
            dbc.Tab(
                tab_supervision_content,
                label="📈 Supervision",
                tab_id="tab-sup",
                className="pt-3"
            ),

            dbc.Tab(
                html.Div(id="tab-opti-content", className="pt-4"),
                label="💾 Table Optimisation",
                tab_id="tab-opti"
            ),

            dbc.Tab(
                html.Div(id="tab-perf-content", className="pt-4"),
                label="📊 Performances Modèles",
                tab_id="tab-perf"
            ),

            dbc.Tab(
                html.Div(id="tab-sim-content", className="pt-4"),
                label="📋 Table Simulation 24h",
                tab_id="tab-sim"
            )
        ]
    ),

    dcc.Interval(
        id='interval',
        interval=INTERVAL_MS,
        n_intervals=0,
        disabled=True
    )

], fluid=True, style=STYLE_PAGE)

# ===========================================================
# CALLBACKS : ONGLET 1 — SUPERVISION
# ===========================================================
@app.callback(
    [
        Output('registry-notification', 'children'),
        Output('store-models-loaded', 'data')
    ],
    [
        Input('dropdown-physicien', 'value'),
        Input('dropdown-energetique', 'value')
    ]
)
def update_models_registry_ui(phys_name, ener_name):
    global scaler_X, modele_physicien, modele_financier, _day_cache

    try:
        folder_phys = MODELS_CONFIG['physiciens'][phys_name].strip()
        file_ener = MODELS_CONFIG['energetiques'][ener_name].strip()

        path_phys_dir = os.path.join(
            "trained_models",
            "modeles_physiciens",
            folder_phys
        )

        path_ener_file = os.path.join(
            "trained_models",
            "modeles_energetiques",
            file_ener
        )

        scaler_X = joblib.load(
            os.path.join(path_phys_dir, "scaler.pkl")
        )

        modele_physicien = tf.keras.models.load_model(
            os.path.join(
                path_phys_dir,
                f"modele_physicien_{folder_phys}.keras"
            ),
            compile=False
        )

        modele_financier = xgb.XGBRegressor()
        modele_financier.load_model(path_ener_file)

        with _cache_lock:
            _day_cache = {}

        return html.Div([
            html.Span("✔️ Modèles chargés", style={
                "color": couleur_opt,
                "fontWeight": "700"
            }),
            html.Div(
                f"{phys_name} · {ener_name}",
                style={
                    "color": TEXT_MUTED,
                    "fontSize": "12px",
                    "marginTop": "3px"
                }
            )
        ], className="text-center mb-3"), time.time()

    except Exception as e:
        return html.Div(
            f"❌ Erreur: {str(e)}",
            className="text-danger fw-bold text-center mb-3"
        ), dash.no_update

@app.callback(
    Output('store-daily-data', 'data'),
    [
        Input('date-picker', 'date'),
        Input('store-models-loaded', 'data')
    ]
)
def manage_daily_batch(picked_date, _):
    if not picked_date or scaler_X is None:
        return dash.no_update

    dict_data = get_day_data_cached(picked_date)

    return dict_data if dict_data else None

@app.callback(
    [
        Output('graph-meteo', 'figure'),
        Output('graph-temperature', 'figure'),
        Output('graph-puissance', 'figure'),
        Output('kpi-energie', 'children'),
        Output('store-idx', 'data'),
        Output('store-play-state', 'data'),
        Output('sim-status-notification', 'children'),
        Output('interval', 'disabled')
    ],
    [
        Input('interval', 'n_intervals'),
        Input('btn-play', 'n_clicks'),
        Input('btn-reset', 'n_clicks'),
        Input('store-daily-data', 'data'),
        Input('store-opti-state', 'data')
    ],
    [
        State('store-idx', 'data'),
        State('store-play-state', 'data'),
        State('dropdown-physicien', 'value'),
        State('dropdown-energetique', 'value')
    ]
)
def update_everything(
    n_int,
    n_play,
    n_reset,
    data,
    is_opti,
    current_idx,
    is_playing,
    phys_name,
    ener_name
):
    if not data:
        return dash.no_update

    ctx = dash.callback_context

    if not ctx.triggered:
        trigger_ids = ['store-daily-data']
    else:
        trigger_ids = [
            t['prop_id'].split('.')[0]
            for t in ctx.triggered
        ]

    if 'store-daily-data' in trigger_ids or 'btn-reset' in trigger_ids:
        current_idx, is_playing = 0, False

    elif 'btn-play' in trigger_ids:
        is_playing = not is_playing

    elif 'interval' in trigger_ids and is_playing:
        current_idx = min(current_idx + 1, len(data))

    if current_idx == 0 and not is_playing:
        sim_status = dbc.Alert(
            "⏹️ Prêt à démarrer",
            color="secondary",
            className="p-2 text-center mb-0 fw-bold"
        )
        interval_disabled = True

    elif is_playing and current_idx < len(data):
        sim_status = dbc.Alert(
            "▶️ En cours",
            color="success",
            className="p-2 text-center mb-0 fw-bold"
        )
        interval_disabled = False

    elif current_idx >= len(data):
        sim_status = dbc.Alert(
            "⏹️ Terminée",
            color="secondary", 
            className="p-2 text-center mb-0 fw-bold text-white"
        )
        interval_disabled = True
        is_playing = False

    else:
        sim_status = dbc.Alert(
            "⏸️ En pause",
            color="warning",
            className="p-2 text-center mb-0 fw-bold text-dark"
        )
        interval_disabled = True

    target_date = pd.to_datetime(data[0]['timestamp']).date()

    x_range = [
        target_date.strftime('%Y-%m-%d 00:00:00'),
        (target_date + timedelta(days=1)).strftime('%Y-%m-%d 00:00:00')
    ]

    current_idx = max(2, current_idx) if current_idx > 0 else 0

    df_plot = pd.DataFrame(data[:current_idx])

    if not df_plot.empty:
        df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'])

    f_m, f_t, f_p = build_figures(
        df_plot,
        is_opti,
        target_date,
        x_range,
        phys_name,
        ener_name
    )

    return (
        f_m,
        f_t,
        f_p,
        build_kpi(df_plot, is_opti),
        current_idx,
        is_playing,
        sim_status,
        interval_disabled
    )

@app.callback(
    [
        Output('store-opti-state', 'data'),
        Output('btn-opti', 'outline')
    ],
    Input('btn-opti', 'n_clicks'),
    State('store-opti-state', 'data'),
    prevent_initial_call=True
)
def toggle_optimisation(n_clicks, is_opti_active):
    new_state = not is_opti_active
    return new_state, not new_state

# ===========================================================
# CALLBACKS : ONGLET 2 — DONNÉES OPTIMISATION SQLITE
# ===========================================================
@app.callback(
    Output('tab-opti-content', 'children'),
    [
        Input('main-tabs', 'active_tab'),
        Input('dropdown-physicien', 'value'),
        Input('interval', 'n_intervals')
    ]
)
def update_opti_table(active_tab, phys_name, n_int):
    if active_tab != "tab-opti":
        return dash.no_update

    try:
        df_opti = pd.read_sql(
            'SELECT * FROM optimisations_data ORDER BY timestamp DESC',
            engine
        )

        for col in df_opti.select_dtypes(include=['float', 'float64', 'float32']).columns:
            if col in ['pente_chauffe', 'pente_froid']:
                df_opti[col] = df_opti[col].round(4)
            else:
                df_opti[col] = df_opti[col].round(2)

        if df_opti.empty:
            return html.Div(
                "Aucune donnée d'optimisation générée. Veuillez cliquer sur Play dans l'onglet Supervision.",
                className="text-warning mt-4"
            )

        target_ochre_headers = ['timestamp', 't_int_opt', 'etat_opt', 'pwr_opt', 'pente_chauffe', 'pente_froid']

        table = dash_table.DataTable(
            data=df_opti.to_dict('records'),
            columns=[
                {'name': i, 'id': i}
                for i in df_opti.columns
            ],
            page_size=20,
            style_table=STYLE_TABLE,
            style_header={**STYLE_TABLE_HEADER, 'fontSize': '26px'},
            style_header_conditional=[
                {'if': {'column_id': c}, 'color': ACCENT_ORANGE} for c in target_ochre_headers
            ],
            style_data=STYLE_TABLE_DATA,
            style_cell=STYLE_TABLE_CELL,
            style_as_list_view=True
        )

        return html.Div([
            html.H3(
                f"📊 Registre d’Optimisation — Modèle actif : {phys_name}",
                style={"color": TEXT_LIGHT, "fontWeight": "800"},
                className="mb-2"
            ),
            html.Div(
                "Historique des consignes optimisées générées par le jumeau numérique.",
                style={"color": TEXT_MUTED},
                className="mb-4"
            ),
            table
        ])

    except Exception as e:
        return html.Div(
            f"Erreur de lecture de la base de données : {e}",
            className="text-danger"
        )

# ===========================================================
# CALLBACKS : ONGLET 3 — PERFORMANCES MODÈLES
# ===========================================================
@app.callback(
    Output('tab-perf-content', 'children'),
    [
        Input('main-tabs', 'active_tab'),
        Input('dropdown-physicien', 'value')
    ]
)
def update_perf_tab(active_tab, phys_name):
    if active_tab != "tab-perf":
        return dash.no_update

    folder_name = MODELS_CONFIG['physiciens'][phys_name].strip()
    suffix = folder_name.replace('lstm_', '')
    base_path = f"trained_models/modeles_physiciens/{folder_name}"

    img_val = get_b64_image(f"{base_path}/validation_vs_test_{suffix}.png")
    img_feat = get_b64_image(f"{base_path}/feature_importance_lstm_{suffix}.png")
    img_learn = get_b64_image(f"{base_path}/learning_curve_{suffix}.png")

    base_style = {
        'cursor': 'zoom-in', 'transition': 'transform 0.3s ease',
        'position': 'relative', 'zIndex': 1, 'borderRadius': '16px',
        'boxShadow': '0 12px 30px rgba(0,0,0,0.25)'
    }

    images_row = dbc.Row([
        dbc.Col(html.Img(id='img-perf-val', src=img_val, n_clicks=0, style=base_style.copy(), className="img-fluid border border-secondary") if img_val else html.P("Graphique Validation non trouvé.", className="text-muted"), width=4),
        dbc.Col(html.Img(id='img-perf-feat', src=img_feat, n_clicks=0, style=base_style.copy(), className="img-fluid border border-secondary") if img_feat else html.P("Graphique Feature Importance non trouvé.", className="text-muted"), width=4),
        dbc.Col(html.Img(id='img-perf-learn', src=img_learn, n_clicks=0, style=base_style.copy(), className="img-fluid border border-secondary") if img_learn else html.P("Graphique Learning Curve non trouvé.", className="text-muted"), width=4),
    ], className="mb-5")

    try:
        df_metrics = pd.read_csv(
            'trained_models/metrics_summary.csv',
            header=None,
            names=['Nom du Modèle', 'Cible', 'MAE', 'RMSE']
        )
        df_metrics = df_metrics[~df_metrics['Nom du Modèle'].str.contains('Modèle', case=False, na=False)]

        target_ochre_headers_metrics = ['Nom du Modèle', 'Cible', 'MAE', 'RMSE']

        metrics_table = dash_table.DataTable(
            data=df_metrics.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df_metrics.columns],
            style_header={**STYLE_TABLE_HEADER, 'fontSize': '26px'},
            style_header_conditional=[
                {'if': {'column_id': c}, 'color': ACCENT_ORANGE} for c in target_ochre_headers_metrics
            ],
            style_data=STYLE_TABLE_DATA,
            style_cell=STYLE_TABLE_CELL,
            style_table=STYLE_TABLE,
            style_data_conditional=[
                {'if': {'filter_query': '{Nom du Modèle} contains "LSTM"'}, 'backgroundColor': 'rgba(91, 212, 242, 0.15)'},
                {'if': {'filter_query': '{Nom du Modèle} contains "XGB"'}, 'backgroundColor': 'rgba(46, 204, 113, 0.15)'},
            ]
        )

    except Exception as e:
        metrics_table = html.Div(f"Fichier metrics_summary.csv introuvable ou illisible ({e}).", className="text-danger")

    return html.Div([
        html.H3(f"📈 Performances des Modèles — Variante : {phys_name}", style={"color": TEXT_LIGHT, "fontWeight": "800"}, className="mb-2"),
        html.Div("Visualisation des métriques de validation, importance des variables et courbes d’apprentissage.", style={"color": TEXT_MUTED}, className="mb-4"),
        images_row,
        html.Hr(style={"borderColor": "rgba(255,255,255,0.12)", "marginBottom": "30px"}),
        html.H3("📋 Tableau Récapitulatif des Métriques MLOps", style={"color": TEXT_LIGHT, "fontWeight": "800"}, className="mb-2"),
        html.Div("Comparaison synthétique des performances des variantes LSTM et XGBoost.", style={"color": TEXT_MUTED}, className="mb-4"),
        dbc.Row(dbc.Col(metrics_table, width=10))
    ])

# ===========================================================
# CALLBACKS : ONGLET 4 — TABLE SIMULATION DU JOUR
# ===========================================================
@app.callback(
    Output('tab-sim-content', 'children'),
    [
        Input('main-tabs', 'active_tab'),
        Input('store-daily-data', 'data'),
        Input('date-picker', 'date')
    ]
)
def update_sim_table(active_tab, daily_data, picked_date):
    if active_tab != "tab-sim":
        return dash.no_update

    if not daily_data:
        return html.Div(
            "Veuillez sélectionner une date valide.",
            className="text-warning"
        )

    df_day = pd.DataFrame(daily_data)

    colonnes_brutes = ['timestamp', 'T_ext', 'T_int', 'Etat_Compresseur', 'Puissance_Elec_kW', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos']

    df_clean = df_day[colonnes_brutes].copy()

    for col in df_clean.select_dtypes(
        include=['float', 'float64', 'float32']
    ).columns:
        df_clean[col] = df_clean[col].round(2)

    target_ochre_headers_brutes = ['timestamp', 'T_ext', 'T_int', 'Etat_Compresseur', 'Puissance_Elec_kW', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos']

    table = dash_table.DataTable(
        data=df_clean.to_dict('records'),
        columns=[
            {'name': i, 'id': i}
            for i in df_clean.columns
        ],
        page_size=20,
        style_table=STYLE_TABLE,
        style_header={**STYLE_TABLE_HEADER, 'fontSize': '26px'},
        style_header_conditional=[
            {'if': {'column_id': c}, 'color': ACCENT_ORANGE} for c in target_ochre_headers_brutes
        ],
        style_data=STYLE_TABLE_DATA,
        style_cell=STYLE_TABLE_CELL,
        style_as_list_view=True
    )

    formatted_date = pd.to_datetime(picked_date).strftime('%d/%m/%Y')

    return html.Div([
        html.H3(
            f"📋 Données de Simulation Brutes — {formatted_date}",
            style={"color": TEXT_LIGHT, "fontWeight": "800"},
            className="mb-2"
        ),

        html.Div(
            "Données usine utilisées pour alimenter le jumeau numérique sur une fenêtre de 24h.",
            style={"color": TEXT_MUTED},
            className="mb-4"
        ),

        table
    ])

# ===========================================================
# CALLBACKS : ONGLET 3 — ZOOM DES IMAGES SUR CLIC
# ===========================================================
def create_zoom_callback(img_id):
    @app.callback(
        Output(img_id, 'style'),
        Input(img_id, 'n_clicks'),
        State(img_id, 'style'),
        prevent_initial_call=True
    )
    def toggle_zoom(n_clicks, current_style):
        if not current_style:
            current_style = {
                'transition': 'transform 0.3s ease',
                'position': 'relative'
            }

        if n_clicks and n_clicks % 2 == 1:
            current_style.update({
                'transform': 'scale(2.0)',
                'zIndex': 1050,
                'cursor': 'zoom-out',
                'boxShadow': '0px 10px 30px rgba(0,0,0,0.8)'
            })

            if img_id == 'img-perf-val':
                current_style['transformOrigin'] = 'top left'
            else:
                current_style['transformOrigin'] = 'center center'

        else:
            current_style.update({
                'transform': 'scale(1.0)',
                'zIndex': 1,
                'cursor': 'zoom-in',
                'boxShadow': 'none',
                'transformOrigin': 'center center'
            })

        return current_style

create_zoom_callback('img-perf-val')
create_zoom_callback('img-perf-feat')
create_zoom_callback('img-perf-learn')

# ===========================================================
# LANCEMENT
# ===========================================================
update_models_registry_ui("LSTM 128u Attention", "XGBoost Depth 5")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9050, debug=False)
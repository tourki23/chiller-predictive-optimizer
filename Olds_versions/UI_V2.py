import dash
from dash import dcc, html, Input, Output, State
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
WINDOW_PENTE = 10

couleur_ia  = '#B833FF'
couleur_opt = '#55AF0F'

# Variables globales pour les modèles (chargées dynamiquement)
scaler_X = None
modele_physicien = None
modele_financier = None
m1_ref = ""
m2_ref = ""

print("=" * 50)
print("🌐 JUMEAU NUMÉRIQUE V14 — Multi-Modèles & Registry")
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
    print("   🗑️  Table 'optimisations_data' vidée au démarrage")

print("📥 Chargement en mémoire des données de simulation 2026...")
df_sim = (pd.read_sql_table('simulation_data_2026_table', engine)
            .sort_values('timestamp')
            .reset_index(drop=True))
df_sim['timestamp'] = pd.to_datetime(df_sim['timestamp'])
df_sim['_date'] = df_sim['timestamp'].dt.date
date_index = df_sim.groupby('_date').apply(lambda g: (g.index[0], g.index[-1])).to_dict()
print(f"   ✅ {len(df_sim):,} lignes chargées")

# ===========================================================
# MOTEUR DE CALCUL (Vectorisé)
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

        if etat_reel_jour[k] == 0: pente_chauffe_list.append(pente_obs)
        else: pente_froid_list.append(pente_obs)

        p_c = np.mean(pente_chauffe_list[-10:]) if pente_chauffe_list else 0.04
        p_f = np.mean(pente_froid_list[-10:]) if pente_froid_list else -0.12

        etat_opt_current = (1.0 if t_opt_current > CONSIGNE + HYSTERESIS_H
                            else (0.0 if t_opt_current < CONSIGNE - HYSTERESIS_L
                                  else etat_opt_current))
        t_opt_next = t_opt_current + (p_f if etat_opt_current == 1.0 else p_c)

        t_opts.append(t_opt_current)
        etat_opts.append(etat_opt_current)
        pente_chauffes.append(p_c)
        pente_froids.append(p_f)
        t_opt_current = t_opt_next

    return (np.array(t_opts), np.array(etat_opts), np.array(pente_chauffes), np.array(pente_froids))

def _write_optimisation_async(df_opt):
    try:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM optimisations_data"))
        df_opt.to_sql('optimisations_data', engine, if_exists='append', index=False)
    except Exception as e: print(f"⚠️ Erreur BDD: {e}")

def compute_day_data(target_date_str):
    target_date = pd.to_datetime(target_date_str).date()
    if target_date not in date_index: return None
    
    i_start_raw, i_end_raw = date_index[target_date]
    idx_start = max(i_start_raw, TIME_STEPS)
    idx_end = min(idx_start + 144, i_end_raw + 1)
    df_day = df_sim.iloc[idx_start:idx_end].copy()

    seqs = _build_sequences_vectorized(idx_start, idx_end)
    t_preds = modele_physicien.predict(seqs, batch_size=64, verbose=0).flatten()
    df_day['t_int_pred'] = t_preds

    X_ia = df_day[['t_int_pred', 'Etat_Compresseur']].rename(columns={'t_int_pred': 'T_int'})
    df_day['pwr_pred_ia'] = modele_financier.predict(X_ia)

    etat_init = float(df_sim.iloc[idx_start - 1]['Etat_Compresseur'])
    t_init = float(df_sim.iloc[idx_start - 1]['T_int'])
    t_opts, etat_opts, pente_c, pente_f = _compute_optimisation(t_preds, df_day['Etat_Compresseur'].values, etat_init, t_init)

    df_day['t_int_opt'], df_day['etat_opt'] = t_opts, etat_opts
    df_day['pente_chauffe'], df_day['pente_froid'] = pente_c, pente_f

    X_opt = pd.DataFrame({'T_int': t_opts, 'Etat_Compresseur': etat_opts})
    df_day['pwr_opt'] = modele_financier.predict(X_opt)

    df_opt_db = df_day[['timestamp', 't_int_opt', 'etat_opt', 'pwr_opt', 'pente_chauffe', 'pente_froid']].copy()
    df_opt_db['timestamp'] = df_opt_db['timestamp'].astype(str)
    executor.submit(_write_optimisation_async, df_opt_db)

    df_day['timestamp'] = df_day['timestamp'].astype(str)
    return df_day.to_dict('records')

_day_cache = {}
_cache_lock = threading.Lock()

def get_day_data_cached(date_str):
    with _cache_lock:
        if date_str not in _day_cache: _day_cache[date_str] = compute_day_data(date_str)
        return _day_cache[date_str]

# ===========================================================
# VISUALISATION
# ===========================================================

def _make_common_layout(x_range, extra=None):
    layout = dict(template="plotly_dark", xaxis=dict(range=x_range, showgrid=True, tickformat="%H:%M"),
                  legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5),
                  margin=dict(l=40, r=20, t=40, b=40), uirevision="fixed")
    if extra: layout.update(extra)
    return layout

def build_figures(df_plot, is_opti_active, target_date, x_range):
    diff_r = df_plot['Etat_Compresseur'].diff().fillna(0)
    on_reel, off_reel = df_plot[diff_r == 1], df_plot[diff_r == -1]
    
    t_int_opt_visuel = df_plot['t_int_opt'].copy()
    idx_first_on = df_plot.index[df_plot['etat_opt'] == 1.0].min()
    if pd.notna(idx_first_on) and idx_first_on > 0: t_int_opt_visuel.loc[:idx_first_on - 1] = np.nan
    elif pd.isna(idx_first_on): t_int_opt_visuel[:] = np.nan

    fig_meteo = go.Figure(data=[go.Scatter(x=df_plot['timestamp'], y=df_plot['T_ext'], mode='lines', name="T° Extérieure", line=dict(color="#CA3FC5"))],
                          layout=_make_common_layout(x_range, {'title': dict(text=f"Météo : {target_date.strftime('%d/%m/%Y')}", x=0.5)}))

    traces_temp = [
        go.Scatter(x=df_plot['timestamp'], y=df_plot['T_int'], mode='lines', name="🔵 Réel", line=dict(color="#2485DB", width=2)),
        go.Scatter(x=on_reel['timestamp'], y=on_reel['T_int'], mode='markers', marker=dict(color='#00FF00', size=11), name='🟢 ON Réel'),
        go.Scatter(x=off_reel['timestamp'], y=off_reel['T_int'], mode='markers', marker=dict(color='#FF4444', size=11), name='🔴 OFF Réel'),
        go.Scatter(x=df_plot['timestamp'], y=df_plot['t_int_pred'], mode='lines', name=f"🟣 IA: {m1_ref}", line=dict(color=couleur_ia, dash='dash')),
        go.Scatter(x=df_plot['timestamp'], y=[CONSIGNE]*len(df_plot), mode='lines', name="Consigne", line=dict(color='white', dash='dot'), opacity=0.4),
    ]
    if is_opti_active:
        on_opt = df_plot[df_plot['etat_opt'].diff() == 1]
        off_opt = df_plot[df_plot['etat_opt'].diff() == -1]
        traces_temp += [
            go.Scatter(x=df_plot['timestamp'], y=t_int_opt_visuel, mode='lines', name="🟢 Optimisé", line=dict(color=couleur_opt, width=2)),
            go.Scatter(x=on_opt['timestamp'], y=on_opt['t_int_opt'], mode='markers', marker=dict(color='#00FF00', size=14, symbol='star'), name='★ ON Opti'),
            go.Scatter(x=off_opt['timestamp'], y=off_opt['t_int_opt'], mode='markers', marker=dict(color='#FF4444', size=14, symbol='star'), name='★ OFF Opti'),
        ]

    fig_temp = go.Figure(data=traces_temp, layout=_make_common_layout(x_range, {'title': dict(text="Simulation Thermique LSTM", x=0.01)}))
    
    fig_pwr = go.Figure(data=[
        go.Scatter(x=df_plot['timestamp'], y=df_plot['Puissance_Elec_kW'], mode='lines', name="Réel", line=dict(color="white")),
        go.Scatter(x=df_plot['timestamp'], y=df_plot['pwr_pred_ia'], mode='lines', name=f"IA: {m2_ref}", line=dict(color=couleur_ia, dash='dash')),
    ], layout=_make_common_layout(x_range, {'title': dict(text="Consommation Énergétique XGBoost", x=0.01)}))
    if is_opti_active: fig_pwr.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['pwr_opt'], mode='lines', name="Optimisé", line=dict(color=couleur_opt)))

    return fig_meteo, fig_temp, fig_pwr

def build_kpi(df_plot, is_opti_active):
    total_reel = df_plot['Puissance_Elec_kW'].sum() * (10/60)
    total_ia = df_plot['pwr_pred_ia'].sum() * (10/60)
    total_opt = df_plot['pwr_opt'].sum() * (10/60) if is_opti_active else 0
    return html.Div([
        html.H5("Usine (24h) :", className="text-white mt-1"), html.H3(f"{total_reel:.1f} kWh", className="text-warning mb-3"),
        html.H5("Prédiction IA :", className="text-white mt-1"), html.H3(f"{total_ia:.1f} kWh", style={'color': couleur_ia}, className="mb-3"),
        html.Div([
            html.H5("Optimisation IA :", className="text-white mt-1"),
            html.H3(f"{total_opt:.1f} kWh", style={'color': couleur_opt}),
            html.Span(f"(-{total_reel-total_opt:.1f} kWh)", className="text-success fw-bold")
        ], style={'display': 'block' if is_opti_active else 'none'})
    ])

def build_tab_stats(df_day):
    # (Logique de l'onglet stats inchangée)
    time_reel_min = int((df_day['Etat_Compresseur'] == 1).sum() * 10)
    time_opt_min  = int((df_day['etat_opt'] == 1).sum() * 10)
    fig_stats = go.Figure(data=[
        go.Bar(name='Réel', x=['Min', 'Moy', 'Max'], y=[df_day['T_int'].min(), df_day['T_int'].mean(), df_day['T_int'].max()]),
        go.Bar(name='IA', x=['Min', 'Moy', 'Max'], y=[df_day['t_int_pred'].min(), df_day['t_int_pred'].mean(), df_day['t_int_pred'].max()]),
        go.Bar(name='Opti', x=['Min', 'Moy', 'Max'], y=[df_day['t_int_opt'].min(), df_day['t_int_opt'].mean(), df_day['t_int_opt'].max()])
    ], layout=dict(template="plotly_dark", barmode='group'))
    return dbc.Row([dbc.Col(dcc.Graph(figure=fig_stats), width=8), dbc.Col(html.Div(f"Gain Temps: {time_reel_min - time_opt_min} min"), width=4)])

# ===========================================================
# LAYOUT
# ===========================================================
tab_supervision = dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("🎛️ Contrôle du Flux"),
            dbc.CardBody([
                html.Label("1. Modèle Physicien (LSTM) :", className="fw-bold text-info"),
                dcc.Dropdown(
                    id='dropdown-physicien',
                    options=[{'label': k, 'value': k} for k in MODELS_CONFIG['physiciens'].keys()],
                    value="LSTM 128u Attention", className="mb-3 text-dark"
                ),
                html.Label("2. Modèle Énergétique (XGB) :", className="fw-bold text-success"),
                dcc.Dropdown(
                    id='dropdown-energetique',
                    options=[{'label': k, 'value': k} for k in MODELS_CONFIG['energetiques'].keys()],
                    value="XGBoost Depth 5", className="mb-3 text-dark"
                ),
                html.Hr(style={"borderColor": "gray"}),
                html.Label("Sélectionnez un jour (2026) :"),
                dcc.DatePickerSingle(id='date-picker', date=date(2026, 12, 1), display_format='DD/MM/YYYY', className="mb-3 w-100"),
                dbc.Button("⏸️ Play / Pause", id="btn-play", color="warning", className="w-100 mb-2"),
                dbc.Button("🔄 Reset", id="btn-reset", color="danger", outline=True, className="w-100 mb-3"),
                dbc.Button("✨ Optimisation énergétique", id="btn-opti", color="success", outline=True, className="w-100 mb-2"),
                html.Div(id="sim-status-notification")
            ])
        ], color="dark", outline=True),
        html.Br(),
        dbc.Card([dbc.CardHeader("⚡ Énergie Cumulée"), dbc.CardBody(id='kpi-energie')], color="dark", outline=True)
    ], width=2),
    dbc.Col([
        dcc.Graph(id='graph-meteo', style={'height': '20vh', 'marginBottom': '10px'}),
        dcc.Graph(id='graph-temperature', style={'height': '30vh', 'marginBottom': '10px'}),
        dcc.Graph(id='graph-puissance', style={'height': '30vh'})
    ], width=10)
])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

app.layout = dbc.Container([
    dcc.Store(id='store-daily-data'), dcc.Store(id='store-idx', data=1),
    dcc.Store(id='store-play-state', data=False), dcc.Store(id='store-opti-state', data=False),
    
    dbc.Row([
        dbc.Col(html.H2("🏭 Jumeau Numérique Axima - MLOps Registry", className="text-primary mt-4"), width=9),
        dbc.Col(html.Div("Developed by TOURKI Mahmoud", className="text-end mt-5 small text-muted"), width=3)
    ]),
    
    dbc.Tabs([
        dbc.Tab(tab_supervision, label="📈 Supervision", className="pt-3"),
        dbc.Tab(dcc.Loading(html.Div(id="tab-kpi-content", className="p-4")), label="📊 Analyses (24h)")
    ]),
    dcc.Interval(id='interval', interval=INTERVAL_MS, n_intervals=0)
], fluid=True)

# ===========================================================
# CALLBACK NOUVEAU : Chargement dynamique des modèles
# ===========================================================
@app.callback(
    Output('sim-status-notification', 'children', allow_duplicate=True),
    [Input('dropdown-physicien', 'value'),
     Input('dropdown-energetique', 'value')],
    prevent_initial_call=True
)
def update_models_registry(phys_name, ener_name):
    global scaler_X, modele_physicien, modele_financier, m1_ref, m2_ref, _day_cache
    
    try:
        # Nettoyage des noms au cas où
        folder_phys = MODELS_CONFIG['physiciens'][phys_name].strip()
        file_ener = MODELS_CONFIG['energetiques'][ener_name].strip()
        
        print(f"🔄 Registry: Chargement {phys_name} + {ener_name}...")
        
        # Chemins absolus ou relatifs propres
        path_phys_dir = os.path.join("trained_models", "modeles_physiciens", folder_phys)
        path_ener_file = os.path.join("trained_models", "modeles_energetiques", file_ener)

        # Vérification de l'existence des fichiers avant de charger
        if not os.path.exists(os.path.join(path_phys_dir, "scaler.pkl")):
            raise FileNotFoundError(f"Scaler introuvable dans {path_phys_dir}")

        # Chargement
        scaler_X = joblib.load(os.path.join(path_phys_dir, "scaler.pkl"))
        modele_physicien = tf.keras.models.load_model(
            os.path.join(path_phys_dir, f"modele_physicien_{folder_phys}.keras"), 
            compile=False
        )
        
        modele_financier = xgb.XGBRegressor()
        modele_financier.load_model(path_ener_file)
        
        m1_ref, m2_ref = phys_name, ener_name
        
        with _cache_lock:
            _day_cache = {}
            
        print("✅ Tous les modèles et scalers sont chargés !")
        return dbc.Badge("Modèles chargés", color="success", className="w-100 p-2")

    except Exception as e:
        print(f"❌ ERREUR REGISTRY : {e}")
        return dbc.Badge(f"Erreur: {str(e)}", color="danger", className="w-100 p-2")

# ===========================================================
# CALLBACKS EXISTANTS (Adaptés pour les nouveaux modèles)
# ===========================================================

@app.callback(
    [Output('store-daily-data', 'data'), Output('store-idx', 'data'), Output('tab-kpi-content', 'children')],
    Input('date-picker', 'date')
)
def manage_daily_batch(picked_date):
    if not picked_date: return dash.no_update
    dict_data = get_day_data_cached(picked_date)
    if dict_data is None: return None, 1, html.H4("Date non disponible")
    return dict_data, 1, build_tab_stats(pd.DataFrame(dict_data))

@app.callback(
    [Output('graph-meteo', 'figure'), Output('graph-temperature', 'figure'), Output('graph-puissance', 'figure'),
     Output('kpi-energie', 'children'), Output('store-idx', 'data', allow_duplicate=True),
     Output('btn-opti', 'outline'), Output('store-play-state', 'data'), Output('store-opti-state', 'data')],
    [Input('interval', 'n_intervals'), Input('btn-play', 'n_clicks'), Input('btn-opti', 'n_clicks'), Input('btn-reset', 'n_clicks')],
    [State('store-daily-data', 'data'), State('store-idx', 'data'), State('store-play-state', 'data'), State('store-opti-state', 'data')],
    prevent_initial_call=True
)
def update_streaming_graphs(n, p_c, o_c, r_c, data, current_idx, is_playing, is_opti):
    if not data: return dash.no_update
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id']
    if "btn-play" in trigger: is_playing = not is_playing
    if "btn-opti" in trigger: is_opti = not is_opti
    if "btn-reset" in trigger: current_idx, is_playing = 1, False
    if is_playing and "interval" in trigger: current_idx = min(current_idx + 1, len(data))

    df_plot = pd.DataFrame(data[:current_idx])
    df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'])
    target_date = df_plot['timestamp'].iloc[0].date()
    x_range = [target_date.strftime('%Y-%m-%d 00:00:00'), (target_date + timedelta(days=1)).strftime('%Y-%m-%d 00:00:00')]

    f_m, f_t, f_p = build_figures(df_plot, is_opti, target_date, x_range)
    return f_m, f_t, f_p, build_kpi(df_plot, is_opti), current_idx, not is_opti, is_playing, is_opti

# Force le premier chargement AVANT que l'utilisateur ne clique
update_models_registry("LSTM 128u Attention", "XGBoost Depth 5")
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9050, debug=False)
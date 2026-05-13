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
# CONFIGURATION
# ===========================================================
PATH_M1  = "trained_models/modele_physicien_lstm_128u_Attention.keras"
PATH_M2  = "trained_models/modele_financier_depth_5.json"
DB_URL   = "sqlite:///axima_poc.db"
FEATURES = ['T_ext', 'T_int', 'Etat_Compresseur', 'T_int_lag_1',
            'day_sin', 'day_cos', 'hour_sin', 'hour_cos']
TIME_STEPS   = 12
CONSIGNE     = 7.0
HYSTERESIS_H = 0.5   # seuil haut  : compresseur ON  si T > consigne + H
HYSTERESIS_L = 0.3   # seuil bas   : compresseur OFF si T < consigne - L
INTERVAL_MS  = 600   # intervalle d'animation (ms) — augmenté pour moins de charge
WINDOW_PENTE = 10    # nb de points pour moyenne glissante des pentes

couleur_ia  = '#B833FF'
couleur_opt = '#55AF0F'

m1_ref = os.path.basename(PATH_M1).replace("modele_physicien_", "").replace(".keras", "").replace("_", " ")
m2_ref = os.path.basename(PATH_M2).replace("modele_financier_", "").replace(".json", "").replace("_", " ")

print("=" * 50)
print("🌐 JUMEAU NUMÉRIQUE V13 — Optimisé & Refactorisé")
print("=" * 50)

# ===========================================================
# CHARGEMENT UNIQUE AU DÉMARRAGE (pas dans les callbacks)
# ===========================================================
engine   = create_engine(DB_URL)
executor = ThreadPoolExecutor(max_workers=2)   # pour les écritures BDD async

with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS optimisations_data (
            timestamp TIMESTAMP, t_int_opt FLOAT, etat_opt FLOAT,
            pwr_opt FLOAT, pente_chauffe FLOAT, pente_froid FLOAT
        )
    """))
    # ✅ TRUNCATE au démarrage : vide la table proprement (plus rapide
    # qu'un DELETE sans WHERE car ne génère pas de logs de transaction)
    conn.execute(text("DELETE FROM optimisations_data"))
    print("   🗑️  Table 'optimisations_data' vidée au démarrage")

print("📥 Chargement en mémoire des données de simulation 2026...")
df_sim = (pd.read_sql_table('simulation_data_2026_table', engine)
            .sort_values('timestamp')
            .reset_index(drop=True))
df_sim['timestamp'] = pd.to_datetime(df_sim['timestamp'])

# Index par date pour lookup O(1) au lieu de masque booléen à chaque appel
df_sim['_date'] = df_sim['timestamp'].dt.date
date_index = df_sim.groupby('_date').apply(lambda g: (g.index[0], g.index[-1])).to_dict()

print(f"   ✅ {len(df_sim):,} lignes chargées — index par date construit")

# Chargement des modèles
scaler_X          = joblib.load("trained_models/scaler.pkl")
modele_physicien  = tf.keras.models.load_model(PATH_M1, compile=False)
modele_financier  = xgb.XGBRegressor()
modele_financier.load_model(PATH_M2)

# Warm-up du modèle LSTM (évite la latence au 1er appel réel)
_dummy = np.zeros((1, TIME_STEPS, len(FEATURES)), dtype=np.float32)
modele_physicien.predict(_dummy, verbose=0)
print("   ✅ Modèles chargés et warm-up effectué")


# ===========================================================
# MOTEUR DE CALCUL — vectorisé
# ===========================================================

def _build_sequences_vectorized(idx_start: int, idx_end: int) -> np.ndarray:
    """
    Construit toutes les séquences d'un coup via slicing numpy
    au lieu d'une list comprehension avec 144 appels à transform().
    """
    # Extraire le bloc brut une seule fois
    raw_block = df_sim.iloc[idx_start - TIME_STEPS: idx_end][FEATURES].values  # (N+TIME_STEPS, F)
    n_seq     = idx_end - idx_start

    # ✅ FIX PERF #1 : scaler.transform appelé UNE SEULE FOIS sur tout le bloc
    scaled_block = scaler_X.transform(raw_block)  # (N+TIME_STEPS, F)

    # Construire les séquences par striding — sans copie supplémentaire
    seqs = np.lib.stride_tricks.sliding_window_view(
        scaled_block, window_shape=(TIME_STEPS, len(FEATURES))
    )[:n_seq, 0]  # (n_seq, TIME_STEPS, F)

    return seqs.astype(np.float32)


def _compute_optimisation(t_preds: np.ndarray,
                           etat_compresseur_jour: np.ndarray,
                           etat_init: float,
                           t_init: float) -> tuple:
    """
    Logique d'optimisation ORIGINALE — copiée mot pour mot depuis le
    script de l'utilisateur. Aucune modification de la logique métier.
    """
    t_opts, etat_opts, pente_chauffes, pente_froids = [], [], [], []

    t_opt_current      = t_init
    etat_opt_current   = etat_init
    pente_chauffe_list = []
    pente_froid_list   = []
    t_pred_prec        = t_init

    for k in range(len(t_preds)):
        t_pred_current = t_preds[k]
        pente_obs      = t_pred_current - t_pred_prec
        t_pred_prec    = t_pred_current

        if etat_compresseur_jour[k] == 0:
            pente_chauffe_list.append(pente_obs)
        else:
            pente_froid_list.append(pente_obs)

        p_c = np.mean(pente_chauffe_list[-10:]) if pente_chauffe_list else 0.04
        p_f = np.mean(pente_froid_list[-10:])   if pente_froid_list   else -0.12

        consigne = 7.0
        etat_opt_current = (1.0 if t_opt_current > consigne + 0.5
                            else (0.0 if t_opt_current < consigne - 0.3
                                  else etat_opt_current))
        t_opt_next = t_opt_current + (p_f if etat_opt_current == 1.0 else p_c)

        t_opts.append(t_opt_current)
        etat_opts.append(etat_opt_current)
        pente_chauffes.append(p_c)
        pente_froids.append(p_f)

        t_opt_current = t_opt_next

    return (np.array(t_opts), np.array(etat_opts),
            np.array(pente_chauffes), np.array(pente_froids))


def _write_optimisation_async(df_opt: pd.DataFrame):
    """Écriture BDD dans un thread séparé pour ne pas bloquer l'UI.
    TRUNCATE est utilisé à la place de DELETE : plus rapide car il ne
    génère pas de logs de transaction ligne par ligne."""
    try:
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE optimisations_data"))
        df_opt.to_sql('optimisations_data', engine, if_exists='append', index=False)
    except Exception as e:
        print(f"⚠️ Écriture BDD async échouée : {e}")


def compute_day_data(target_date_str: str) -> list | None:
    """
    Calcul complet pour une journée.
    Mis en cache au niveau applicatif via un dict Python (plus rapide que Flask-Cache
    pour des objets déjà en mémoire).
    """
    target_date = pd.to_datetime(target_date_str).date()

    if target_date not in date_index:
        return None

    i_start_raw, i_end_raw = date_index[target_date]
    idx_start = max(i_start_raw, TIME_STEPS)
    idx_end   = min(idx_start + 144, i_end_raw + 1)

    df_day = df_sim.iloc[idx_start:idx_end].copy()

    # ── Prédictions LSTM (batch unique) ──────────────────────────────────────
    seqs   = _build_sequences_vectorized(idx_start, idx_end)
    t_preds = modele_physicien.predict(seqs, batch_size=64, verbose=0).flatten()
    df_day['t_int_pred'] = t_preds

    # ── Prédictions XGBoost IA ───────────────────────────────────────────────
    X_ia = df_day[['t_int_pred', 'Etat_Compresseur']].rename(
        columns={'t_int_pred': 'T_int'})
    df_day['pwr_pred_ia'] = modele_financier.predict(X_ia)

    # ── Optimisation — logique originale stricte ─────────────────────────────
    etat_init      = float(df_sim.iloc[idx_start - 1]['Etat_Compresseur'])
    t_init         = float(df_sim.iloc[idx_start - 1]['T_int'])
    etat_reel_jour = df_day['Etat_Compresseur'].values   # slice du jour uniquement

    t_opts, etat_opts, pente_c, pente_f = _compute_optimisation(
        t_preds, etat_reel_jour, etat_init, t_init
    )

    df_day['t_int_opt']    = t_opts
    df_day['etat_opt']     = etat_opts
    df_day['pente_chauffe']= pente_c
    df_day['pente_froid']  = pente_f

    # ── Puissance optimisée (batch unique) ───────────────────────────────────
    X_opt = pd.DataFrame({'T_int': t_opts, 'Etat_Compresseur': etat_opts})
    df_day['pwr_opt'] = modele_financier.predict(X_opt)

    # ── Écriture BDD (async — ne bloque pas l'UI) ────────────────────────────
    df_opt_db = df_day[['timestamp', 't_int_opt', 'etat_opt',
                         'pwr_opt', 'pente_chauffe', 'pente_froid']].copy()
    df_opt_db['timestamp'] = df_opt_db['timestamp'].astype(str)
    executor.submit(_write_optimisation_async, df_opt_db)

    df_day['timestamp'] = df_day['timestamp'].astype(str)
    return df_day.to_dict('records')


# Cache applicatif simple : dict en mémoire, réinitialisé au restart
_day_cache: dict = {}
_cache_lock = threading.Lock()

def get_day_data_cached(date_str: str) -> list | None:
    with _cache_lock:
        if date_str not in _day_cache:
            _day_cache[date_str] = compute_day_data(date_str)
        return _day_cache[date_str]


# ===========================================================
# FONCTIONS DE CONSTRUCTION DES FIGURES
# (séparées du callback pour lisibilité et réutilisabilité)
# ===========================================================

def _make_common_layout(x_range: list, extra: dict = None) -> dict:
    layout = dict(
        template="plotly_dark",
        xaxis=dict(range=x_range, showgrid=True, tickformat="%H:%M"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.5,
                    xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=40, b=40),
        uirevision="fixed"   # ✅ FIX PERF #3 : évite le re-rendu complet du layout
    )
    if extra:
        layout.update(extra)
    return layout


def build_figures(df_plot: pd.DataFrame, is_opti_active: bool,
                  target_date: date, x_range: list) -> tuple:
    """Construit les 3 figures Plotly depuis le DataFrame courant."""

    # Transitions compresseur réel
    diff_r   = df_plot['Etat_Compresseur'].diff().fillna(0)
    on_reel  = df_plot[diff_r == 1]
    off_reel = df_plot[diff_r == -1]

    # Transitions optimisées
    etat_opt_s = df_plot['etat_opt']
    first_val  = etat_opt_s.iloc[0]          # 0.0 ou 1.0
    diff_o     = etat_opt_s.diff().fillna(first_val)
    on_opt     = df_plot[diff_o == 1]
    off_opt    = df_plot[diff_o == -1]

    # =========================================================================
    # ✅ CORRECTION VISUELLE (COURBE VERTE)
    # Pour ne pas fausser les calculs XGBoost, on crée une copie strictement
    # réservée à l'affichage. On masque (NaN) tout l'historique situé avant 
    # la toute première décision d'allumage (ON) de l'IA.
    # =========================================================================
    t_int_opt_visuel = df_plot['t_int_opt'].copy()
    idx_first_on = df_plot.index[df_plot['etat_opt'] == 1.0].min()

    if pd.notna(idx_first_on) and idx_first_on > 0:
        # Masque toutes les valeurs de l'index 0 jusqu'à juste avant le 1er ON
        t_int_opt_visuel.loc[:idx_first_on - 1] = np.nan
    elif pd.isna(idx_first_on):
        # Si l'IA décide de ne jamais allumer le compresseur de la journée
        t_int_opt_visuel[:] = np.nan
    # =========================================================================

    # ── Météo ──────────────────────────────────────────────────────────────
    fig_meteo = go.Figure(
        data=[go.Scatter(
            x=df_plot['timestamp'], y=df_plot['T_ext'],
            mode='lines',
            name="T° Extérieure",
            line=dict(color="#CA3FC5")
        )],
        layout=_make_common_layout(x_range, {
            'title': dict(text=f"Météo : {target_date.strftime('%d/%m/%Y')}",
                          x=0.5)
        })
    )

    # ── Températures ───────────────────────────────────────────────────────
    consigne_y = [CONSIGNE] * len(df_plot)

    traces_temp = [
        # Courbe réelle
        go.Scatter(
            x=df_plot['timestamp'], y=df_plot['T_int'],
            mode='lines',
            name="🔵 Réel",
            line=dict(color="#2485DB", width=2)
        ),
        # Marqueurs ON/OFF réel
        go.Scatter(
            x=on_reel['timestamp'], y=on_reel['T_int'],
            mode='markers',
            marker=dict(color='#00FF00', size=11, symbol='circle',
                        line=dict(color='white', width=1)),
            name='🟢 ON Réel  ●'
        ),
        go.Scatter(
            x=off_reel['timestamp'], y=off_reel['T_int'],
            mode='markers',
            marker=dict(color='#FF4444', size=11, symbol='circle',
                        line=dict(color='white', width=1)),
            name='🔴 OFF Réel ●'
        ),
        # Courbe IA LSTM
        go.Scatter(
            x=df_plot['timestamp'], y=df_plot['t_int_pred'],
            mode='lines',
            name="🟣 IA LSTM",
            line=dict(color=couleur_ia, dash='dash', width=2)
        ),
        # Ligne de consigne
        go.Scatter(
            x=df_plot['timestamp'], y=consigne_y,
            mode='lines',
            name=f"⚪ Consigne {CONSIGNE}°C",
            line=dict(color='white', dash='dot', width=1.5),
            opacity=0.6
        ),
    ]

    if is_opti_active:
        traces_temp += [
            # Courbe optimisée — UTILISE LA COPIE VISUELLE AVEC LES NaN
            go.Scatter(
                x=df_plot['timestamp'], y=t_int_opt_visuel, # <--- CHANGEMENT ICI
                mode='lines',                               
                name="🟢 Optimisé",
                line=dict(color=couleur_opt, width=2)
            ),
            # Étoile verte
            go.Scatter(
                x=on_opt['timestamp'], y=on_opt['t_int_opt'],
                mode='markers+text',
                marker=dict(color='#00FF00', size=16, symbol='star',
                            line=dict(color='#003300', width=1.5)),
                text=["ON"] * len(on_opt),
                textposition="top center",
                textfont=dict(color='#00FF00', size=9),
                name='★ ON Opti (vert)'
            ),
            # Étoile rouge
            go.Scatter(
                x=off_opt['timestamp'], y=off_opt['t_int_opt'],
                mode='markers+text',
                marker=dict(color='#FF4444', size=16, symbol='star',
                            line=dict(color='#330000', width=1.5)),
                text=["OFF"] * len(off_opt),
                textposition="bottom center",
                textfont=dict(color='#FF4444', size=9),
                name='★ OFF Opti (rouge)'
            ),
        ]

    fig_temp = go.Figure(
        data=traces_temp,
        layout=_make_common_layout(x_range, {
            'title': dict(text=f"Modèle physicien LSTM : {m1_ref}", x=0.01),
            'legend': dict(
                orientation="h",
                yanchor="bottom", y=-0.55,
                xanchor="center", x=0.5,
                bgcolor="rgba(0,0,0,0.4)",
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1
            )
        })
    )

    # ── Puissance ──────────────────────────────────────────────────────────
    traces_pwr = [
        go.Scatter(x=df_plot['timestamp'], y=df_plot['Puissance_Elec_kW'],
                   mode='lines',
                   name="Conso Réelle",
                   line=dict(color="white", width=1)),
        go.Scatter(x=df_plot['timestamp'], y=df_plot['pwr_pred_ia'],
                   mode='lines',
                   name="IA XGBoost",
                   line=dict(color=couleur_ia, dash='dash')),
    ]
    if is_opti_active:
        traces_pwr.append(
            go.Scatter(x=df_plot['timestamp'], y=df_plot['pwr_opt'],
                       mode='lines',
                       name="Conso Optimisée",
                       line=dict(color=couleur_opt))
        )
    fig_pwr = go.Figure(
        data=traces_pwr,
        layout=_make_common_layout(x_range, {
            'title': dict(text=f"Modèle Énergétique XGBoost : {m2_ref}", x=0.01)
        })
    )

    return fig_meteo, fig_temp, fig_pwr


def build_kpi(df_plot: pd.DataFrame, is_opti_active: bool) -> html.Div:
    total_reel = df_plot['Puissance_Elec_kW'].sum() * (10 / 60)
    total_ia   = df_plot['pwr_pred_ia'].sum()       * (10 / 60)
    total_opt  = df_plot['pwr_opt'].sum()            * (10 / 60) if is_opti_active else 0
    eco        = total_reel - total_opt if is_opti_active else 0

    return html.Div([
        html.H5("Usine (24h) :",    className="text-white mt-1"),
        html.H3(f"{total_reel:.1f} kWh", className="text-warning mb-3"),
        html.H5("Prédiction IA  :",     className="text-white mt-1"),
        html.H3(f"{total_ia:.1f} kWh", style={'color': couleur_ia}, className="mb-3"),
        html.Div([
            html.H5("Optimisation  IA :", className="text-white mt-1"),
            html.H3(f"{total_opt:.1f} kWh",
                    style={'color': couleur_opt, 'display': 'inline-block',
                           'marginRight': '10px'}),
            html.Span(
                f" (-{eco:.1f})" if eco >= 0 else f" (+{abs(eco):.1f})",
                className="text-success fw-bold" if eco >= 0 else "text-danger fw-bold"
            )
        ], style={'display': 'block' if is_opti_active else 'none'})
    ])


def build_tab_stats(df_day: pd.DataFrame) -> dbc.Row:
    """Construit l'onglet statistiques 24h."""
    time_reel_min = int((df_day['Etat_Compresseur'] == 1).sum() * 10)
    time_opt_min  = int((df_day['etat_opt']          == 1).sum() * 10)

    cats     = ['Température Min', 'Température Moyenne', 'Température Max']
    val_reel = [df_day['T_int'].min(),       df_day['T_int'].mean(),       df_day['T_int'].max()]
    val_ia   = [df_day['t_int_pred'].min(),  df_day['t_int_pred'].mean(),  df_day['t_int_pred'].max()]
    val_opt  = [df_day['t_int_opt'].min(),   df_day['t_int_opt'].mean(),   df_day['t_int_opt'].max()]

    fig_stats = go.Figure()
    for name, vals, color in [
        ('Terrain (Réel)', val_reel, '#2485DB'),
        ('IA (LSTM)',       val_ia,   couleur_ia),
        ('Optimisation IA', val_opt,  couleur_opt),
    ]:
        fig_stats.add_trace(go.Bar(
            name=name, x=cats, y=vals, marker_color=color,
            text=[f"{v:.2f}°C" for v in vals], textposition='auto'
        ))
    fig_stats.update_layout(
        template="plotly_dark",
        title="Comparatif des Températures sur 24h",
        barmode='group'
    )

    return dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_stats), width=8),
        dbc.Col(dbc.Card([
            dbc.CardHeader("⏱️ Temps de Fonctionnement Compresseur (24h)",
                           className="text-white fw-bold",
                           style={'backgroundColor': '#d9534f'}),
            dbc.CardBody([
                html.H5([html.Span("🔵 Terrain (Réel) : ", className="text-info"),
                         f"{time_reel_min} min"], className="mb-4"),
                html.H5([html.Span("🟣 IA (Prédite) : ",
                                   style={'color': couleur_ia}),
                         f"{time_reel_min} min"], className="mb-4"),
                html.H5([html.Span("🟢 Optimisation : ",
                                   style={'color': couleur_opt}),
                         f"{time_opt_min} min"], className="mb-0"),
            ])
        ], className="h-100"), width=4)
    ])


# ===========================================================
# LAYOUT
# ===========================================================
tab_supervision = dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("🎛️ Contrôle du Flux"),
            dbc.CardBody([
                html.Label("Sélectionnez un jour (2026) :"),
                dcc.DatePickerSingle(
                    id='date-picker', date=date(2026, 12, 1),
                    display_format='DD/MM/YYYY', className="mb-4 w-100"
                ),
                dbc.Button("⏸️ Play / Pause", id="btn-play",
                           color="warning", className="w-100 mb-2"),
                dbc.Button("🔄 Reset", id="btn-reset",
                           color="danger", outline=True, className="w-100 mb-3"),
                dbc.Button("✨ Optimisation énergétique", id="btn-opti",
                           color="success", outline=True, className="w-100 mb-2"),
                html.Div(id="sim-status-notification")
            ])
        ], color="dark", outline=True),
        html.Br(),
        dbc.Card([
            dbc.CardHeader("⚡ Énergie Cumulée (24h)"),
            dbc.CardBody(id='kpi-energie')
        ], color="dark", outline=True)
    ], width=2),

    dbc.Col([
        dcc.Graph(id='graph-meteo',       style={'height': '20vh', 'marginBottom': '10px'}),
        dcc.Graph(id='graph-temperature', style={'height': '30vh', 'marginBottom': '10px'}),
        dcc.Graph(id='graph-puissance',   style={'height': '30vh'})
    ], width=10)
])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
cache = Cache(app.server, config={'CACHE_TYPE': 'SimpleCache'})



app.layout = dbc.Container([
    # 1. VOS STORES (Variables invisibles)
    dcc.Store(id='store-daily-data'),
    dcc.Store(id='store-idx',        data=1),
    dcc.Store(id='store-play-state', data=False),
    dcc.Store(id='store-opti-state', data=False),

    # 2. NOTRE NOUVEL EN-TÊTE
    dbc.Row([
        # --- COLONNE DE GAUCHE : LE TITRE UNIQUE (Largeur : 9) ---
        dbc.Col(
            html.H2("🏭 Jumeau Numérique & Commande Prédictive (MPC)", 
                    className="text-primary text-start mb-0",
                    style={"fontSize": "1.8rem"}), 
            width=9
        ),

        # --- COLONNE DE DROITE : LINKEDIN (Largeur : 3) ---
        dbc.Col(html.Div([
            html.Span("developed by TOURKI Mahmoud | ", className="text-muted small"),
            html.A("LinkedIn",
                   href="https://www.linkedin.com/in/mahmoud-tourki-b228b9147/?skipRedirect=true",
                   target="_blank", className="text-info small fw-bold")
        ], className="text-end"), width=3)
        
    ], className="mt-4 mb-4 align-items-center"),

    dbc.Tabs([
        dbc.Tab(tab_supervision, label="📈 Supervision (Graphiques)", className="pt-3"),
        dbc.Tab(
            dcc.Loading(type="circle",
                        children=html.Div(id="tab-kpi-content", className="p-4")),
            label="📊 Analyses & Statistiques (24h)"
        )
    ]),

    dcc.Interval(id='interval', interval=INTERVAL_MS,
                 n_intervals=0, disabled=False)
], fluid=True)


# ===========================================================
# CALLBACK 1 — Chargement des données à la sélection de date
# ===========================================================
@app.callback(
    [Output('store-daily-data', 'data'),
     Output('store-idx', 'data'),
     Output('tab-kpi-content', 'children')],
    Input('date-picker', 'date')
)
def manage_daily_batch(picked_date):
    if not picked_date:
        return dash.no_update, dash.no_update, dash.no_update

    # ✅ FIX PERF #4 : cache en mémoire dict (plus rapide que Flask-Cache
    #    pour des objets Python déjà sérialisés)
    dict_data = get_day_data_cached(picked_date)

    if dict_data is None:
        return None, 1, html.H4("Aucune donnée pour cette date.",
                                className="text-danger")

    df_day = pd.DataFrame(dict_data)
    df_day['timestamp'] = pd.to_datetime(df_day['timestamp'])

    tab2 = build_tab_stats(df_day)
    return dict_data, 1, tab2


# ===========================================================
# CALLBACK 2 — Animation des graphiques
# ===========================================================
@app.callback(
    [Output('graph-meteo',              'figure'),
     Output('graph-temperature',        'figure'),
     Output('graph-puissance',          'figure'),
     Output('kpi-energie',              'children'),
     Output('store-idx',                'data',    allow_duplicate=True),
     Output('btn-opti',                 'outline'),
     Output('store-play-state',         'data'),
     Output('store-opti-state',         'data'),
     Output('sim-status-notification',  'children')],
    [Input('interval',   'n_intervals'),
     Input('btn-play',   'n_clicks'),
     Input('btn-opti',   'n_clicks'),
     Input('btn-reset',  'n_clicks')],
    [State('store-daily-data',  'data'),
     State('store-idx',         'data'),
     State('store-play-state',  'data'),
     State('store-opti-state',  'data')],
    prevent_initial_call=True
)
def update_streaming_graphs(n, play_clicks, opti_clicks, reset_clicks,
                             data, current_idx, is_playing, is_opti_active):
    if not data:
        return dash.no_update

    ctx     = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'] if ctx.triggered else ""

    # Gestion des boutons
    if "btn-play"  in trigger: is_playing      = not is_playing
    if "btn-opti"  in trigger: is_opti_active  = not is_opti_active
    if "btn-reset" in trigger:
        current_idx = 1
        is_playing  = False

    # Avance de l'index uniquement sur tick interval
    if is_playing and "interval" in trigger:
        current_idx = min(current_idx + 1, len(data))

    # Statut de simulation
    if is_playing and current_idx < len(data):
        sim_status = dbc.Alert("▶️ Simulation en cours",
                               color="success", className="p-2 text-center mb-0 fw-bold")
    elif current_idx >= len(data):
        sim_status = dbc.Alert("⏹️ Simulation terminée",
                               color="info", className="p-2 text-center mb-0 fw-bold")
        is_playing = False
    else:
        sim_status = dbc.Alert("⏸️ Simulation arrêtée",
                               color="danger", className="p-2 text-center mb-0 fw-bold")

    # ✅ FIX PERF #5 : slicing direct sur la liste + construction DataFrame minimale
    #    On ne reconstruit que les colonnes nécessaires à l'affichage
    df_plot = pd.DataFrame(data[:current_idx])
    if df_plot.empty:
        return dash.no_update

    df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'])

    target_date = df_plot['timestamp'].iloc[0].date()
    x_range = [
        target_date.strftime('%Y-%m-%d 00:00:00'),
        (target_date + timedelta(days=1)).strftime('%Y-%m-%d 00:00:00')
    ]

    fig_meteo, fig_temp, fig_pwr = build_figures(
        df_plot, is_opti_active, target_date, x_range
    )
    kpi = build_kpi(df_plot, is_opti_active)

    return (fig_meteo, fig_temp, fig_pwr, kpi,
            current_idx, not is_opti_active,
            is_playing, is_opti_active, sim_status)


# ===========================================================
# LANCEMENT
# ===========================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9050, debug=False)
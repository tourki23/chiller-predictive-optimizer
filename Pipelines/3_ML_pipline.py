import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import mlflow.keras
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import warnings
import time
import joblib

# --- Imports Deep Learning ---
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, MultiHeadAttention, GlobalAveragePooling1D, Dropout, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')


def create_sequences(X: np.ndarray, y: np.ndarray, time_steps: int = 12):
    """
    Crée des séquences temporelles à partir de tableaux numpy.
    Reçoit des arrays (pas des DataFrames) pour éviter toute ambiguïté d'index.
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def compute_permutation_importance(model, X_test: np.ndarray, y_test: np.ndarray,
                                   feature_names: list, base_mae: float) -> list:
    """
    Calcule la permutation importance en shufflant correctement
    les valeurs de chaque feature INDÉPENDAMMENT au sein des séquences.

    FIX : on permute X_test[:, :, i] en reshapant en 1D, shufflant,
    puis en remettant en forme — ce qui mélange les valeurs de la feature i
    entre tous les timesteps et tous les samples.
    """
    importances = []
    n_samples, n_steps, n_features = X_test.shape

    for i, feature_name in enumerate(feature_names):
        X_shuffled = X_test.copy()
        # Extraire les valeurs de la feature i sur tous samples × timesteps
        feature_vals = X_shuffled[:, :, i].flatten()
        np.random.shuffle(feature_vals)
        X_shuffled[:, :, i] = feature_vals.reshape(n_samples, n_steps)

        shuffled_preds = model.predict(X_shuffled, verbose=0)
        shuffled_mae = mean_absolute_error(y_test, shuffled_preds)
        importances.append(shuffled_mae - base_mae)  # delta MAE = importance

    return importances


def build_lstm_model(time_steps: int, n_features: int, units: int, use_attention: bool) -> Model:
    """
    Construit le modèle LSTM avec ou sans attention.
    Ajout de LayerNormalization pour stabiliser l'entraînement.
    """
    inputs = Input(shape=(time_steps, n_features))

    if use_attention:
        x = LSTM(units, return_sequences=True)(inputs)
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)
        attn_out = MultiHeadAttention(num_heads=4, key_dim=units // 4)(x, x)
        x = LayerNormalization()(attn_out)
        x = GlobalAveragePooling1D()(x)
    else:
        x = LSTM(units, return_sequences=True)(inputs)
        x = LayerNormalization()(x)
        x = LSTM(units // 2, return_sequences=False)(x)  # 2ème couche LSTM
        x = Dropout(0.2)(x)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    outputs = Dense(1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
        loss=Huber(delta=1.0),
        metrics=['mae']
    )
    return model


def train_and_log_models():
    start_global = time.time()
    print("\n" + "=" * 60)
    print("🚀 PIPELINE MLOPS V11 : ENTRAÎNEMENT COMPLET (SPLIT 80/20)")
    print("   ✅ Data Leakage corrigé — Scaler fitté sur TRAIN uniquement")
    print("=" * 60 + "\n")

    output_dir = "trained_models"
    os.makedirs(output_dir, exist_ok=True)

    metrics_summary = []

    # ===========================================================
    # ÉTAPE 1 : CHARGEMENT DES DONNÉES
    # ===========================================================
    print("📥 [1/4] Chargement de la table 'training_table' (2024-2025)...")
    engine = create_engine('postgresql://kyc_user:kyc_password@127.0.0.1:5433/axima_poc')
    df = pd.read_sql_table('training_table', engine)
    df.columns = [str(c) for c in df.columns]
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    print(f"✅ {len(df):,} lignes d'entraînement prêtes.\n")

    mlflow.set_experiment("Axima_V11_Digital_Twin_Full_NoLeak")

    # ===========================================================
    # ÉTAPE 2 : SPLIT TEMPOREL EN PREMIER, PUIS NORMALISATION
    # ===========================================================
    print("⚖️ [2/4] Split temporel 80/20 → Normalisation → Séquences...")

    features_m1 = ['T_ext', 'T_int', 'Etat_Compresseur', 'T_int_lag_1',
                   'day_sin', 'day_cos', 'hour_sin', 'hour_cos']
    target_m1 = 'T_int_future'

    TIME_STEPS = 12

    # ✅ FIX #1 : Split RAW avant toute normalisation
    raw_split_idx = int(len(df) * 0.80)
    df_train_raw = df.iloc[:raw_split_idx].copy()
    df_test_raw = df.iloc[raw_split_idx:].copy()

    print(f"   Train brut : {len(df_train_raw):,} lignes | Test brut : {len(df_test_raw):,} lignes")

    # ✅ FIX #2 : Scaler fitté UNIQUEMENT sur le train
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(df_train_raw[features_m1])
    X_test_scaled = scaler_X.transform(df_test_raw[features_m1])  # transform only !

    y_train_raw = df_train_raw[target_m1].values
    y_test_raw = df_test_raw[target_m1].values

    # Sauvegarde du scaler (fitté sans fuite)
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    joblib.dump(scaler_X, scaler_path)
    print(f"   ✅ Scaler sauvegardé : {scaler_path}")

    # ✅ FIX #3 : Séquences créées séparément sur train et test
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_raw, TIME_STEPS)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_raw, TIME_STEPS)

    print(f"   Séquences Train : {X_train_seq.shape} | Test : {X_test_seq.shape}\n")

    # ===========================================================
    # ÉTAPE 3 : MODÈLES LSTM
    # ===========================================================
    print("⚡ [3/4] Entraînement des 4 variantes LSTM...")

    lstm_configs = [
        {"name": "128u_NoAttention", "units": 128, "use_attention": False},
        {"name": "128u_Attention",   "units": 128, "use_attention": True},
        {"name": "64u_NoAttention",  "units": 64,  "use_attention": False},
        {"name": "64u_Attention",    "units": 64,  "use_attention": True},
    ]

    callbacks_lstm = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0),
    ]

    for config in lstm_configs:
        run_name = f"LSTM_{config['name']}"
        with mlflow.start_run(run_name=run_name):
            print(f"\n   ➤ Entraînement Variante : {run_name} ...")
            mlflow.log_artifact(scaler_path)
            mlflow.log_param("units", config["units"])
            mlflow.log_param("use_attention", config["use_attention"])
            mlflow.log_param("time_steps", TIME_STEPS)
            mlflow.log_param("train_samples", len(X_train_seq))
            mlflow.log_param("test_samples", len(X_test_seq))

            model_1 = build_lstm_model(TIME_STEPS, len(features_m1),
                                       config["units"], config["use_attention"])

            history = model_1.fit(
                X_train_seq, y_train_seq,
                epochs=50,
                batch_size=64,
                validation_data=(X_test_seq, y_test_seq),
                callbacks=callbacks_lstm,
                verbose=1
            )

            preds_1 = model_1.predict(X_test_seq, verbose=0)
            mae = mean_absolute_error(y_test_seq, preds_1)
            rmse = np.sqrt(mean_squared_error(y_test_seq, preds_1))

            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("epochs_trained", len(history.history['loss']))

            metrics_summary.append({
                "Modèle": run_name,
                "Cible": "Température (°C)",
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4)
            })

            # 📈 Learning Curve
            fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
            ax_loss.plot(history.history['loss'], label='Train (Huber)')
            ax_loss.plot(history.history['val_loss'], label='Validation (Huber)')
            ax_loss.set_title(f"Learning Curve — {config['name']}")
            ax_loss.set_xlabel("Époques")
            ax_loss.set_ylabel("Erreur")
            ax_loss.legend()
            mlflow.log_figure(fig_loss, f"learning_curve_{config['name']}.png")
            plt.close(fig_loss)

            # 📈 Prédiction vs Réel (200 derniers points)
            fig_val, ax_val = plt.subplots(figsize=(15, 6))
            ax_val.plot(y_test_seq[-200:], label='Réel', color='#1f77b4', linewidth=2)
            ax_val.plot(preds_1[-200:], label='Prédiction IA', color='#d62728',
                        linestyle='--', linewidth=2)
            ax_val.set_title(f"Validation Set (Test 20%) — {config['name']}")
            ax_val.legend()
            mlflow.log_figure(fig_val, f"validation_vs_test_{config['name']}.png")
            plt.close(fig_val)

            # 📈 Feature Importance (Permutation — corrigée)
            # ✅ FIX #4 : permutation correcte feature par feature
            lstm_importances = compute_permutation_importance(
                model_1, X_test_seq, y_test_seq, features_m1, mae
            )

            fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
            sorted_idx = np.argsort(lstm_importances)
            ax_fi.barh(np.array(features_m1)[sorted_idx],
                       np.array(lstm_importances)[sorted_idx], color='#9467bd')
            ax_fi.set_title(f"Feature Importance (Permutation) — {config['name']}")
            ax_fi.set_xlabel("Δ MAE (plus grand = plus important)")
            mlflow.log_figure(fig_fi, f"feature_importance_lstm_{config['name']}.png")
            plt.close(fig_fi)

            model_path = os.path.join(output_dir, f"modele_physicien_lstm_{config['name']}.keras")
            model_1.save(model_path)
            mlflow.keras.log_model(model_1, "model")
            print(f"     ✅ {run_name} — MAE: {mae:.4f} °C | RMSE: {rmse:.4f} °C "
                  f"({len(history.history['loss'])} époques)")

    # ===========================================================
    # ÉTAPE 4 : MODÈLE FINANCIER (XGBOOST)
    # ===========================================================
    print("\n💰 [4/4] Entraînement des variantes XGBoost (Épuré)...")

    features_m2 = ['T_int', 'Etat_Compresseur']

    # ✅ FIX #3 BIS : split XGBoost aligné sur le même raw_split_idx
    # (évite le décalage de TIME_STEPS entre split LSTM et split XGB)
    X_train_xgb = df_train_raw[features_m2].values
    X_test_xgb = df_test_raw[features_m2].values
    y_train_xgb = df_train_raw['Puissance_Elec_kW'].values
    y_test_xgb = df_test_raw['Puissance_Elec_kW'].values

    print(f"   XGBoost — Train : {len(X_train_xgb):,} | Test : {len(X_test_xgb):,}")

    for depth in [3, 5, 7, 9]:
        with mlflow.start_run(run_name=f"XGB_Financier_Depth_{depth}"):
            mlflow.log_param("max_depth", depth)
            mlflow.log_param("n_estimators", 500)
            mlflow.log_param("learning_rate", 0.05)

            model_2 = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=depth,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                early_stopping_rounds=20,
            )
            model_2.fit(
                X_train_xgb, y_train_xgb,
                eval_set=[(X_test_xgb, y_test_xgb)],
                verbose=False
            )

            preds_2 = model_2.predict(X_test_xgb)
            mae_2 = mean_absolute_error(y_test_xgb, preds_2)
            rmse_2 = np.sqrt(mean_squared_error(y_test_xgb, preds_2))

            mlflow.log_metric("mae", mae_2)
            mlflow.log_metric("rmse", rmse_2)
            mlflow.log_metric("best_iteration", model_2.best_iteration)

            metrics_summary.append({
                "Modèle": f"XGBoost_Depth_{depth}",
                "Cible": "Puissance (kW)",
                "MAE": round(mae_2, 4),
                "RMSE": round(rmse_2, 4)
            })

            # 📈 Feature Importance XGBoost
            xgb_importances = model_2.feature_importances_
            fig_fi_xgb, ax_fi_xgb = plt.subplots(figsize=(6, 4))
            sorted_idx_xgb = np.argsort(xgb_importances)
            ax_fi_xgb.barh(np.array(features_m2)[sorted_idx_xgb],
                           np.array(xgb_importances)[sorted_idx_xgb], color='#2ca02c')
            ax_fi_xgb.set_title(f"Feature Importance (Gain) — XGB Depth {depth}")
            mlflow.log_figure(fig_fi_xgb, f"feature_importance_xgb_depth_{depth}.png")
            plt.close(fig_fi_xgb)

            model_2.save_model(os.path.join(output_dir, f"modele_financier_depth_{depth}.json"))
            print(f"   💾 XGBoost Depth {depth} — MAE: {mae_2:.4f} kW | "
                  f"Best iter: {model_2.best_iteration}")

    # ===========================================================
    # ÉTAPE 5 : TABLEAU RÉCAPITULATIF
    # ===========================================================
    df_summary = pd.DataFrame(metrics_summary)

    csv_path = os.path.join(output_dir, "metrics_summary.csv")
    md_path = os.path.join(output_dir, "metrics_summary.md")

    df_summary.to_csv(csv_path, index=False)
    df_summary.to_markdown(md_path, index=False)

    print("\n" + "=" * 55)
    print("📊 RÉCAPITULATIF DES PERFORMANCES DES MODÈLES")
    print("=" * 55)
    print(df_summary.to_markdown(index=False))
    print("=" * 55)
    print(f"Fichiers de synthèse : {csv_path} & {md_path}")

    duration = (time.time() - start_global) / 60
    print(f"\n🎉 PIPELINE V11 TERMINÉ EN {duration:.2f} MINUTES")


if __name__ == "__main__":
    train_and_log_models()
# tests/test_models.py
import os
import pytest

def test_lstm_model_exists():
    """Vérifie que le modèle principal LSTM est bien présent dans le package."""
    model_path = "trained_models/modeles_physiciens/lstm_128u_Attention/modele_physicien_lstm_128u_Attention.keras"
    assert os.path.exists(model_path), f"Le modèle {model_path} est introuvable."

def test_xgboost_model_exists():
    """Vérifie que le modèle énergétique XGBoost est bien présent."""
    model_path = "trained_models/modeles_energetiques/modele_financier_depth_5.json"
    assert os.path.exists(model_path), "Le modèle XGBoost est introuvable."
# tests/test_logic.py
import pytest
import numpy as np
# Importation de ta fonction (ajuste le chemin si nécessaire)
from app.UI import _compute_optimisation

def test_compute_optimisation_heating_logic():
    """
    Vérifie que la fonction d'optimisation déclenche bien le chauffage
    lorsque la température initiale est très basse.
    """
    # 1. Préparation des fausses données de test (Mock data)
    # Imaginons 5 prédictions de température où l'IA dit qu'il va faire très froid
    t_preds_froid = np.array([4.0, 3.8, 3.5, 3.2, 3.0])
    # Un état de compresseur réel qui était à l'arrêt
    etat_reel_jour = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    etat_init = 0.0 # Compresseur initialement éteint
    t_init = 3.0    # Température initiale très basse (en dessous de la consigne de 7.0)

    # 2. Appel de la fonction à tester
    t_opts, etat_opts, pente_c, pente_f = _compute_optimisation(
        t_preds_froid, 
        etat_reel_jour, 
        etat_init, 
        t_init
    )

    # 3. Vérifications (Assertions)
    # Si la température initiale est à 3.0 (consigne 7.0 - Hysteresis L),
    # le jumeau d'optimisation DOIT ordonner d'éteindre le froid (état = 0.0)
    assert etat_opts[0] == 0.0, "Le compresseur devrait être coupé car il fait déjà trop froid."
    
    # On vérifie que la taille des tableaux de retour est bonne
    assert len(t_opts) == 5, "La fonction doit renvoyer autant de points que la prédiction."
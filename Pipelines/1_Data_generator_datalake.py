import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from datetime import datetime
import math
import os

def generate_raw_iot_data():
    print("=========================================")
    print("🌊 DATA LAKE : GÉNÉRATION IOT BRUTE (2024-2026 | 5 sec)")
    print("=========================================")

    spark = SparkSession.builder.appName("Axima_RawDataGen_V10_5s").master("local[*]").config("spark.driver.memory", "4g").getOrCreate()
    output_path = "./datalake/raw_iot_data"
    
    import shutil
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2026, 12, 31, 23, 59, 59)
    
    consigne = 7.0
    hysteresis = 0.5
    T_int = 7.2
    etat = 0
    
    refroidissement_5s = 0.15 / 120.0
    rechauffement_5s = 0.05 / 120.0
    facteur_fuite_5s = 0.009 / 120.0 
    
    total_lignes = 0
    current_date = start_date

    print("⏳ Génération par batch mensuel en cours...")

    while current_date <= end_date:
        next_month = (current_date.replace(day=28) + pd.Timedelta(days=4)).replace(day=1)
        chunk_end = min(next_month - pd.Timedelta(seconds=5), end_date)
        
        timestamps = pd.date_range(start=current_date, end=chunk_end, freq='5s')
        data = []
        
        for t in timestamps:
            jour_yday = t.dayofyear
            annees_ecoulees = (t.year - 2024) + (jour_yday / 365.25)
            
            # 💡 CORRECTION DÉFINITIVE DE LA MÉTÉO (Hiver Froid, Été Chaud)
            T_saison = 12.5 - 10.0 * math.cos(2 * math.pi * (jour_yday - 15) / 365.25)
            T_journaliere = 5.0 * math.sin(2 * math.pi * (t.hour - 6) / 24.0)
            T_ext = T_saison + T_journaliere + (annees_ecoulees * 0.2)
            
            if T_int > consigne + hysteresis: etat = 1
            elif T_int < consigne - hysteresis: etat = 0

            delta_t = T_ext - T_int
            usure = 1.0 + (0.15 * (annees_ecoulees / 3.0)) 

            if etat == 1:
                puissance = (15.0 + delta_t * 0.4) * usure
                T_int -= refroidissement_5s 
            else:
                puissance = 0.5 
                T_int += rechauffement_5s + (delta_t * facteur_fuite_5s)
                
            bruit_T_int = T_int + np.random.normal(0, 0.05)
            bruit_puissance = max(puissance + np.random.normal(0, 0.5), 0)

            data.append({
                "timestamp": t,
                "t_ext_raw": round(T_ext, 3), 
                "t_int_raw": round(bruit_T_int, 3),
                "puissance_elec_kw": round(bruit_puissance, 3),
                "etat_compresseur": etat
            })

        df_spark = spark.createDataFrame(pd.DataFrame(data))
        df_spark.write.mode("append").parquet(output_path)
        
        total_lignes += len(data)
        print(f"✔️ Batch {current_date.strftime('%Y-%m')} généré ({len(data):,} lignes)")
        current_date = next_month

    print(f"\n✅ Terminé ! Data Lake construit : {total_lignes:,} lignes.")
    spark.stop()

if __name__ == "__main__":
    generate_raw_iot_data()
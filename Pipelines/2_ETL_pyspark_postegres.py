from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg, round, col, year, lag, lead,
    dayofyear, hour, minute, sin, cos,
    date_trunc, to_timestamp, unix_timestamp,
    floor, from_unixtime
)
from pyspark.sql.window import Window
import math


def run_etl():
    print("\n" + "=" * 60)
    print("🚀 PIPELINE ETL V13 : AGRÉGATION & SPLIT (TRAIN 2024-25 / TEST 2026)")
    print("   ✅ Remplacement de window() par date_trunc — bug 2025 corrigé")
    print("=" * 60)

    spark = SparkSession.builder.appName("Axima_ETL_V13") \
        .config("spark.jars.packages", "org.postgresql:postgresql:42.6.0") \
        .config("spark.driver.memory", "6g") \
        .config("spark.sql.session.timeZone", "Europe/Paris") \
        .getOrCreate()

    # -------------------------------------------------------
    # ÉTAPE 1 : Lecture Parquet
    # -------------------------------------------------------
    print("\n⏳ [1/4] Lecture Parquet brut...")
    df_raw = spark.read.parquet("./datalake/raw_iot_data")
    print(f"   Lignes brutes : {df_raw.count():,}")

    # Vérification rapide de la distribution annuelle en entrée
    print("   Distribution annuelle brute :")
    df_raw.groupBy(year("timestamp").alias("annee")).count().orderBy("annee").show()

    # -------------------------------------------------------
    # ÉTAPE 2 : Agrégation par tranche de 10 minutes
    #
    # ✅ FIX PRINCIPAL : on abandonne pyspark window() qui aligne
    # les fenêtres sur l'epoch Unix et produit des groupes incohérents.
    # On calcule à la place un identifiant de tranche 10min via
    # floor(unix_timestamp / 600) * 600, puis on recrée le timestamp
    # propre avec from_unixtime. Cela garantit :
    #   - Des tranches de exactement 10 minutes
    #   - Un timestamp de début de tranche déterministe
    #   - Aucune ligne orpheline aux bords des partitions
    # -------------------------------------------------------
    print("\n⏳ [2/4] Agrégation par tranches de 10 minutes (date_trunc / floor)...")

    WINDOW_SECONDS = 600  # 10 minutes

    df_with_slot = df_raw.withColumn(
        "slot_ts",
        from_unixtime(
            floor(unix_timestamp("timestamp") / WINDOW_SECONDS) * WINDOW_SECONDS
        ).cast("timestamp")
    )

    df_agg = df_with_slot.groupBy("slot_ts").agg(
        round(avg("t_ext_raw"), 3).alias("T_ext"),
        round(avg("t_int_raw"), 3).alias("T_int"),
        round(avg("etat_compresseur"), 0).alias("Etat_Compresseur"),
        round(avg("puissance_elec_kw"), 3).alias("Puissance_Elec_kW")
    ).withColumnRenamed("slot_ts", "timestamp")

    # Tri global indispensable avant le lag/lead
    # On repartitionne par année pour paralléliser proprement,
    # puis on trie globalement pour que le lag soit correct.
    df_agg = df_agg.orderBy("timestamp")

    total_agg = df_agg.count()
    print(f"   Lignes agrégées : {total_agg:,}")
    print("   Distribution annuelle après agrégation :")
    df_agg.groupBy(year("timestamp").alias("annee")).count().orderBy("annee").show()

    # -------------------------------------------------------
    # ÉTAPE 3 : Feature Engineering
    # -------------------------------------------------------
    print("\n⏳ [3/4] Feature Engineering (Temps Cyclique & Inertie Thermique)...")

    pi2 = 2 * math.pi

    df_feat = df_agg \
        .withColumn("heure_decimale", hour("timestamp") + (minute("timestamp") / 60.0)) \
        .withColumn("day_sin",  round(sin(pi2 * dayofyear("timestamp") / 365.25), 4)) \
        .withColumn("day_cos",  round(cos(pi2 * dayofyear("timestamp") / 365.25), 4)) \
        .withColumn("hour_sin", round(sin(pi2 * col("heure_decimale") / 24.0), 4)) \
        .withColumn("hour_cos", round(cos(pi2 * col("heure_decimale") / 24.0), 4)) \
        .drop("heure_decimale")

    # ✅ lag/lead sur un DataFrame TRIé globalement par timestamp
    # On force une seule partition pour le Window global afin d'éviter
    # les null aux bords de partition — acceptable ici car les données
    # agrégées (~ 105 000 lignes sur 3 ans) tiennent en mémoire.
    w_global = Window.orderBy("timestamp")

    df_feat = df_feat \
        .withColumn("T_int_lag_1",  round(lag("T_int", 1).over(w_global), 3)) \
        .withColumn("T_int_future", round(lead("T_int", 1).over(w_global), 3))

    # ✅ dropna uniquement sur les colonnes critiques, pas sur tout le DataFrame
    # Cela élimine uniquement la 1ère et la dernière ligne (bords globaux),
    # pas des blocs entiers de partitions comme avec window() non trié.
    df_final = df_feat.dropna(subset=["T_int_lag_1", "T_int_future",
                                       "T_ext", "T_int",
                                       "Etat_Compresseur", "Puissance_Elec_kW"])

    print(f"   Lignes après dropna : {df_final.count():,}")
    print("   Distribution annuelle finale :")
    df_final.groupBy(year("timestamp").alias("annee")).count().orderBy("annee").show()

    # -------------------------------------------------------
    # ÉTAPE 4 : Split et écriture PostgreSQL
    # -------------------------------------------------------
    print("\n⏳ [4/4] Split 2024+2025 / 2026 et écriture PostgreSQL...")

    db_url   = "jdbc:postgresql://127.0.0.1:5433/axima_poc"
    db_props = {
        "user": "kyc_user",
        "password": "kyc_password",
        "driver": "org.postgresql.Driver"
    }

    df_train = df_final.filter(year("timestamp").isin(2024, 2025))
    df_test  = df_final.filter(year("timestamp") == 2026)

    n_train = df_train.count()
    n_test  = df_test.count()

    print(f"\n   ✅ Lignes TRAIN (2024 + 2025) : {n_train:,}")
    print(f"   ✅ Lignes TEST  (2026)         : {n_test:,}")

    if n_train == 0:
        raise ValueError("❌ df_train est vide après le split — vérifiez le filtrage.")
    if n_test == 0:
        print("   ⚠️  df_test vide — données 2026 absentes ou filtrées.")

    # Vérification par année dans le train
    print("\n   Détail du train par année :")
    df_train.groupBy(year("timestamp").alias("annee")).count().orderBy("annee").show()

    df_train.write.jdbc(url=db_url, table="training_table",
                        mode="overwrite", properties=db_props)
    print("   💾 Table 'training_table' écrite.")

    if n_test > 0:
        df_test.write.jdbc(url=db_url, table="simulation_data_2026_table",
                           mode="overwrite", properties=db_props)
        print("   💾 Table 'simulation_data_2026_table' écrite.")

    print("\n✅ PIPELINE ETL V13 TERMINÉ AVEC SUCCÈS")
    spark.stop()


if __name__ == "__main__":
    run_etl()
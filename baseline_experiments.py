import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, mean_absolute_error
from catboost import CatBoostClassifier, CatBoostRegressor
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

    # === PATH CONFIG ===
    result_path = "baseline_results"
    os.makedirs(result_path, exist_ok=True)

    data_folder = "data"
    emb_folder = "data_embedding"
    os.makedirs(emb_folder, exist_ok=True)

    # === METRIC ===
    regression_metric = lambda y_true, y_pred: f"MAE: {mean_absolute_error(y_true, y_pred):.4f}"

    # === TASKS DEFINITION ===
    tasks = {
        "job_state": {
            "type": "classification",
            "target": lambda j: 1 if j["job_state"] == "COMPLETED" else 0
        },
        "run_time": {
            "type": "regression",
            "target": lambda j: int(j["run_time"] / 60)
        }
    }

    # === FEATURES DEFINITION ===
    features = {
        "int_anon": lambda df: df[["user_id", "group_id", "partition", "qos"]].astype(int).values,
        "sb_anon": lambda df: np.vstack(df["embedding_anon"].values),
    }

    # === MODELS ===
    model_candidates = {
        "classification": {
            "KNN": KNeighborsClassifier,
            "RF": RandomForestClassifier,
            "XGB": XGBClassifier,
            "CatBoost": CatBoostClassifier
        },
        "regression": {
            "KNN": KNeighborsRegressor,
            "RF": RandomForestRegressor,
            "XGB": XGBRegressor,
            "CatBoost": CatBoostRegressor
        }
    }

    # === LOAD SBERT MODEL ONCE ===
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

    # === PROCESS ONE MONTH'S DATA ===
    file_name = "job_table.parquet"
    ym = file_name.replace(".parquet", "")
    df_path = os.path.join(data_folder, file_name)
    emb_save_path = os.path.join(emb_folder, f"{ym}_embedding.parquet")

    print(f"\n=== Processing {ym} ===")

    # Load dataset
    df = pd.read_parquet(df_path)

    # === GENERATE OR LOAD EMBEDDINGS ===
    if os.path.exists(emb_save_path):
        print(f"🔁 Loading existing embeddings: {emb_save_path}")
        emb_df = pd.read_parquet(emb_save_path)
        df["embedding_anon"] = emb_df["embedding_anon"]
    else:
        print(f"Generating SBert embeddings for {ym} ...")
        df["merged_text"] = df.apply(
            lambda r: f"{r['user_id']}, {r['group_id']}, {r['partition']}, {r['qos']}", axis=1
        )
        embeddings = sbert_model.encode(
            df["merged_text"].tolist(),
            batch_size=256,
            show_progress_bar=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        df["embedding_anon"] = list(embeddings)

        emb_df = df[["embedding_anon"]].copy()
        emb_df.to_parquet(emb_save_path)
        print(f"✅ Saved new embeddings: {emb_save_path}")

    # === BUILD TARGETS ===
    for task_name, task_info in tasks.items():
        df[task_name] = df.apply(task_info["target"], axis=1)

    # === SPLIT DATA ===
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    print("\nTraining and evaluating models...\n")

    # === TRAIN AND EVALUATE ===
    for feat_name, feat_func in features.items():
        xtr = feat_func(train_df)
        xte = feat_func(test_df)

        for task_name, task_info in tasks.items():
            task_type = task_info["type"]
            ytr = train_df[task_name].values
            yte = test_df[task_name].values

            for model_name, model_cls in model_candidates[task_type].items():
                print(f"Running {model_name} on {task_name} ({task_type}) with features {feat_name}...")

                # Init model
                if model_name != "CatBoost":
                    model_instance = model_cls(n_jobs=-1)
                else:
                    model_instance = model_cls(task_type="GPU") if torch.cuda.is_available() else model_cls(thread_count=-1)

                # Train
                model_instance.fit(xtr, ytr)
                y_pred = model_instance.predict(xte)

                # Evaluate
                if task_type == "classification":
                    report = classification_report(yte, y_pred)
                else:
                    report = regression_metric(yte, y_pred)

                # Save results
                result_file = os.path.join(result_path, f"{model_name}_{feat_name}_{task_name}.txt")
                with open(result_file, "w", encoding="utf-8") as f:
                    f.write(report)

                print(f"✅ Saved {result_file}")
    
    print("\nAll experiments completed successfully!")

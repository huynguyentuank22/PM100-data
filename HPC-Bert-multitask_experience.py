#!/usr/bin/env python3
"""
HPC-Bert_experience.py

Phiên bản đã chỉnh sửa cho 1 file dữ liệu duy nhất:
- Tải file parquet từ args.input
- Split train/test trực tiếp bằng train_test_split
- Giữ nguyên toàn bộ logic embedding, model training, lưu kết quả
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from catboost import CatBoostClassifier, CatBoostRegressor
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HPC-BERT baseline on single input file.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input parquet file.")
    args = parser.parse_args()

    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

    # === Paths ===
    result_path = "baseline_results"
    os.makedirs(result_path, exist_ok=True)

    emb_folder = "data_embedding_semantic_hpcb-multitask"
    os.makedirs(emb_folder, exist_ok=True)

    regression_metric = lambda y_true, y_pred: f"MAE: {mean_absolute_error(y_true, y_pred):.4f}"

    # === Tasks ===
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

    # === Features ===
    features = {
        "hpcb-multitask_anon": lambda df: np.vstack(df["embedding_anon"].values)
    }

    # === Model candidates ===
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

    # === Semantic template ===
    semantic_templates = [
        lambda r: f"Job {r['jnam']}, which will be executed by {r['usr']}, requires exclusive access to the infrastructure {r['jobenv_req']}."
    ]

    # === Load dataset ===
    df = pd.read_parquet(args.input)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    ym = os.path.splitext(os.path.basename(args.input))[0]

    # === Loop qua từng task ===
    for task, task_info in tasks.items():
        print(f"\n🔹 Loading SBERT for task: {task}")
        model_path = f"models/finetuned_all-MiniLM-L6-v2_multitask/{task}_triplet/model"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model path not found: {model_path}")

        sbert_model = SentenceTransformer(model_path)
        print(f"✅ Loaded SBERT from {model_path}")

        # === Loop qua từng template ===
        for template_idx, template_fn in enumerate(semantic_templates, start=1):
            print(f"\n🔹 Running template {template_idx} for task {task}...")

            emb_save_path_train = os.path.join(
                emb_folder, f"{ym}_train_template{template_idx}_{task}.parquet"
            )
            emb_save_path_test = os.path.join(
                emb_folder, f"{ym}_test_template{template_idx}_{task}.parquet"
            )

            # === TRAIN EMBEDDINGS ===
            if os.path.exists(emb_save_path_train) and os.path.exists(emb_save_path_test):
                print(f"🔁 Loading cached embeddings for {task} ...")
                train_df["embedding_anon"] = pd.read_parquet(emb_save_path_train)["embedding_anon"]
                test_df["embedding_anon"] = pd.read_parquet(emb_save_path_test)["embedding_anon"]
            else:
                print(f"Generating embeddings using SBERT ({task}) ...")
                for split_name, split_df, save_path in [
                    ("train", train_df, emb_save_path_train),
                    ("test", test_df, emb_save_path_test)
                ]:
                    print(f" → Encoding {split_name} split ...")
                    split_df["merged_text"] = split_df.apply(template_fn, axis=1)
                    embeddings = sbert_model.encode(
                        split_df["merged_text"].tolist(),
                        batch_size=256,
                        show_progress_bar=True,
                        device="cuda" if torch.cuda.is_available() else "cpu"
                    )
                    split_df["embedding_anon"] = list(embeddings)
                    split_df[["embedding_anon"]].to_parquet(save_path)
                    print(f"✅ Saved {split_name} embeddings: {save_path}")

            # === Train & Evaluate Models ===
            task_type = task_info["type"]
            y_train = train_df.apply(task_info["target"], axis=1).tolist()
            y_test = test_df.apply(task_info["target"], axis=1).tolist()

            for feat in features:
                x_train = features[feat](train_df)
                x_test = features[feat](test_df)

                for model_name, model_cls in model_candidates[task_type].items():
                    print(f"\n▶ Training {model_name} for {task} ({task_type})...")

                    model_instance = (
                        model_cls(n_jobs=-1)
                        if model_name != "CatBoost"
                        else model_cls(task_type="GPU")
                        if torch.cuda.is_available()
                        else model_cls(thread_count=-1)
                    )

                    model_instance.fit(x_train, y_train)
                    y_pred = model_instance.predict(x_test)

                    if task_type == "classification":
                        report = classification_report(y_test, y_pred)
                    else:
                        report = regression_metric(y_test, y_pred)

                    result_file = os.path.join(
                        result_path,
                        f"{model_name}_{feat}_template{template_idx}_{task}.txt"
                    )
                    with open(result_file, "w", encoding="utf-8") as f:
                        f.write(report)

                    print(f"✅ Saved result to {result_file}")

    print("\n🎯 All tasks completed successfully!")

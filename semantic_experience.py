import os
import torch
import random
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
    emb_folder = "data_embedding_semantic"
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
        },
        "node_pcon": {
            "type": "regression",
            "target": lambda j: np.mean(j["node_power_consumption"]) if len(j["node_power_consumption"]) > 0 else 0
        },
        "mem_pcon": {
            "type": "regression",
            "target": lambda j: np.mean(j["mem_power_consumption"]) if len(j["mem_power_consumption"]) > 0 else 0
        },
        "cpu_pcon": {
            "type": "regression",
            "target": lambda j: np.mean(j["cpu_power_consumption"]) if len(j["cpu_power_consumption"]) > 0 else 0
        },
    }

    # === FEATURES DEFINITION ===
    features = {
        "sb_anon": lambda df: np.vstack(df["embedding_anon"].values)
    }

    # === MODELS ===
    model_candidates = {
        "classification": {
            # "KNN": KNeighborsClassifier,
            # "RF": RandomForestClassifier,
            # "XGB": XGBClassifier,
            "CatBoost": CatBoostClassifier
        },
        "regression": {
            # "KNN": KNeighborsRegressor,
            # "RF": RandomForestRegressor,
            # "XGB": XGBRegressor,
            "CatBoost": CatBoostRegressor
        }
    }

    # === SEMANTIC TEMPLATES ===
    semantic_templates = [
        # lambda r: f"This job was submitted by user {r['user_id']} in group {r['group_id']}, assigned to partition {r['partition']} with quality of service {r['qos']}.",
        # lambda r: f"User {r['user_id']} from group {r['group_id']} executed a job under the {r['partition']} partition using QoS level {r['qos']}.",
        # lambda r: f"Job submitted under partition {r['partition']} by user {r['user_id']} (group {r['group_id']}), with QoS setting {r['qos']}.",
        # lambda r: f"A job associated with user {r['user_id']} and group {r['group_id']} was scheduled on partition {r['partition']}, quality of service: {r['qos']}.",
        # lambda r: f"Job configuration — user: {r['user_id']}, group: {r['group_id']}, partition: {r['partition']}, QoS: {r['qos']}."
        lambda r: f"HPC Job {r['job_id']} in partition {r['partition']} submitted by user {r['user_id']} with group {r['group_id']} requested {r['num_cores_req']} cores across {r['num_nodes_req']} nodes with {r['num_tasks']} tasks, {r['cores_per_task']} cores per task and {r['threads_per_core']} threads per core. The job required {r['mem_req']:.2f} GB memory and {r['num_gpus_req']} GPUs. Job was submitted at {r['submit_time']}, became eligible at {r['eligible_time']}, and started execution at {r['start_time']} with time limit of {r['time_limit']:.2f} seconds. The job operated with priority level {r['priority']}, QoS setting {r['qos']}, sharing mode {r['shared']}, specific node requirements {r['req_nodes']}, switch requirements {r['req_switch']}."
    ]

    # === LOAD SBERT MODEL ONCE ===
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

    # === PROCESS ONE DATA FILE ===
    file_name = "job_table.parquet"
    ym = file_name.replace(".parquet", "")
    df_path = os.path.join(data_folder, file_name)

    print(f"\n=== Processing {ym} ===")
    df = pd.read_parquet(df_path)

    # === BUILD TARGETS ===
    for task_name, task_info in tasks.items():
        df[task_name] = df.apply(task_info["target"], axis=1)

    # === LOOP OVER SEMANTIC TEMPLATES ===
    for tidx, template_func in enumerate(semantic_templates, start=1):
        print(f"\n🧠 Running SBERT embedding with template {tidx} ...")

        # emb_save_path = os.path.join(emb_folder, f"{ym}_template{tidx}_embedding.parquet")
        emb_save_path = os.path.join(emb_folder, f"{ym}_template_full_embedding.parquet")

        # === GENERATE OR LOAD EMBEDDINGS ===
        if os.path.exists(emb_save_path):
            print(f"🔁 Loading cached embeddings: {emb_save_path}")
            emb_df = pd.read_parquet(emb_save_path)
            df["embedding_anon"] = emb_df["embedding_anon"]
        else:
            df["merged_text"] = df.apply(template_func, axis=1)
            embeddings = sbert_model.encode(
                df["merged_text"].tolist(),
                batch_size=256,
                show_progress_bar=True,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            df["embedding_anon"] = list(embeddings)
            df[["embedding_anon"]].to_parquet(emb_save_path)
            print(f"✅ Saved embeddings: {emb_save_path}")

        # === SPLIT TRAIN/TEST ===
        # test_df = df[df['submit_time'].dt.to_period('M') == '2020-10']
        # train_df = df[df['submit_time'].dt.to_period('M') < '2020-10']

        # Giả sử df đã có cột submit_time dạng datetime có timezone
        df['submit_time'] = pd.to_datetime(df['submit_time'], utc=True)

        # B1. Sort theo thời gian (quan trọng)
        df = df.sort_values(by='submit_time')

        # B2. Xác định mốc thời gian để tách (1 tháng cuối)
        split_date = df['submit_time'].max() - pd.DateOffset(months=1)

        # B3. Tạo train/test theo điều kiện thời gian
        train_df = df[df['submit_time'] < split_date]
        test_df  = df[df['submit_time'] >= split_date]

        print("Train set:", train_df['submit_time'].min(), "→", train_df['submit_time'].max())
        print("Number of training samples:", len(train_df))
        print("Test set: ", test_df['submit_time'].min(), "→", test_df['submit_time'].max())
        print("Number of test samples:", len(test_df))
        print(f"\nTraining and evaluating models for Template {tidx}...\n")

        # === TRAIN AND EVALUATE ===
        for feat_name, feat_func in features.items():
            xtr = feat_func(train_df)
            xte = feat_func(test_df)

            for task_name, task_info in tasks.items():
                task_type = task_info["type"]
                ytr = train_df[task_name].values
                yte = test_df[task_name].values

                for model_name, model_cls in model_candidates[task_type].items():
                    print(f"🚀 {model_name} on {task_name} ({task_type}) with {feat_name} [Template {tidx}]")

                    # Init model
                    if model_name == "CatBoost":
                        model_instance = model_cls(task_type="GPU") if torch.cuda.is_available() else model_cls(thread_count=-1)
                    else:
                        model_instance = model_cls(n_jobs=-1)

                    # Train
                    model_instance.fit(xtr, ytr)
                    y_pred = model_instance.predict(xte)

                    # Evaluate
                    if task_type == "classification":
                        report = classification_report(yte, y_pred)
                    else:
                        report = regression_metric(yte, y_pred)

                    # Save results
                    # result_file = os.path.join(
                    #     result_path,
                    #     f"{model_name}_{feat_name}_template{tidx}_{task_name}.txt"
                    # )
                    result_file = os.path.join(
                        result_path,
                        f"{model_name}_{feat_name}_template_full_{task_name}.txt"
                    )

                    with open(result_file, "w", encoding="utf-8") as f:
                        f.write(report)

                    print(f"✅ Saved {result_file}")

    print("\n🎯 All experiments with all templates completed successfully!")

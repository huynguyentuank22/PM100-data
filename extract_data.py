import os
from multiprocessing import Pool
from typing import Iterable, Literal
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import ast

def round_to_closest_second(timestamp: datetime, sampling_time: int = 20, mode: Literal["ceil", "floor"] = "ceil") -> datetime:
    """
    Round timestamp to the closest fixed interval.
    """
    if (timestamp.second % sampling_time) == 0:
        return timestamp

    for i in range(int(60 / sampling_time)):
        values = list(range(i * sampling_time, (i + 1) * sampling_time))
        s = timestamp.second
        if s in values:
            if mode == "ceil":
                return timestamp + timedelta(seconds=values[-1] - s + 1)
            else:
                return timestamp - timedelta(seconds=s - values[0])
    return timestamp


def create_node_hashmap(node: str, jobs: pd.DataFrame, job_start_field: str = "start_time", job_end_field: str = "end_time",
                        job_id_field: str = "job_id", job_nodes_field: str = "nodes", sampling_time: int = 20) -> dict:
    """
    Creates the occupancy hashmap for a given node.
    """
    node_jobs = jobs.loc[jobs[job_nodes_field].apply(lambda ns: isinstance(ns, (list, set, tuple)) and node in ns)]

    hashmap = {}
    for job_id, start, end in node_jobs[[job_id_field, job_start_field, job_end_field]].values:
        for t in pd.date_range(
            round_to_closest_second(start, sampling_time=sampling_time, mode="ceil"),
            round_to_closest_second(end, sampling_time=sampling_time, mode="floor"),
            freq=f"{sampling_time}s"
        ).to_pydatetime():
            hashmap.setdefault(str(t), []).append(job_id)
    return hashmap


def get_non_exclusive_ids(nodes_global_hashmaps: list) -> Iterable:
    """
    Returns job IDs that are non-exclusive (i.e., jobs running concurrently on the same node).
    """
    non_exclusive_set = set()
    for node_hashmap in nodes_global_hashmaps:
        for ts_jobs in node_hashmap.values():
            if len(ts_jobs) > 1:
                non_exclusive_set.update(ts_jobs)
    return non_exclusive_set

def extract_job_power(job_data: pd.Series, ps0: pd.DataFrame, ps1: pd.DataFrame,
                      job_start_field: str = "start_time", job_end_field: str = "end_time",
                      job_id_field: str = "job_id", job_nodes_field: str = "nodes",
                      save_path: str = None) -> dict:
    """
    Extract the job's power consumption from power tables.
    """
    job_id = job_data[job_id_field]

    try:
        # Parse node list
        nodes_raw = job_data[job_nodes_field]
        nodes = ast.literal_eval(nodes_raw) if isinstance(nodes_raw, str) else nodes_raw

        if not isinstance(nodes, (list, set, tuple)) or len(nodes) == 0:
            raise ValueError("Invalid node list for job")

        # Extract time range
        start_time = job_data[job_start_field]
        end_time = job_data[job_end_field]

        # Filter by node
        ps0_nodes = ps0.loc[ps0["node"].isin(nodes)]
        ps1_nodes = ps1.loc[ps1["node"].isin(nodes)]

        # Filter by time range
        ps0_filtered = ps0_nodes.loc[
            (ps0_nodes["timestamp"] >= start_time) & (ps0_nodes["timestamp"] <= end_time)
        ]
        ps1_filtered = ps1_nodes.loc[
            (ps1_nodes["timestamp"] >= start_time) & (ps1_nodes["timestamp"] <= end_time)
        ]

        # Aggregate power values
        ps0_power = ps0_filtered.groupby("timestamp").sum()["value"].values
        ps1_power = ps1_filtered.groupby("timestamp").sum()["value"].values

        power = ps0_power + ps1_power

        if len(power) == 0:
            raise ValueError("No power data found.")

        # Optionally save to pickle
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, f"{job_id}.pkl"), "wb") as f:
                pickle.dump(power, f)

        return {
            "job_id": job_id,
            "start_time": start_time,
            "end_time": end_time,
            "nodes": nodes,
            "power_consumption": power
        }

    except Exception as e:
        return {
            "job_id": job_id,
            "start_time": job_data.get(job_start_field, None),
            "end_time": job_data.get(job_end_field, None),
            "nodes": job_data.get(job_nodes_field, []),
            "power_consumption": [],
            "error": str(e)
        }

if __name__ == "__main__":
    num_cores = os.cpu_count()
    print(f"Number of cores available: {num_cores}")
    n_threads = max(1, num_cores - 1)

    month = "20-12"

    job_table_path = os.path.join("data", month, "year_month=20-12", "plugin=job_table",
                                  "metric=job_info_marconi100", "a_0.parquet")
    ps0_table_path = os.path.join("data", month, "year_month=20-12", "plugin=ipmi_pub",
                                  "metric=ps0_input_power", "a_0.parquet")
    ps1_table_path = os.path.join("data", month, "year_month=20-12", "plugin=ipmi_pub",
                                  "metric=ps1_input_power", "a_0.parquet")

    final_table_path = os.path.join("final_data", f"{month}.parquet")

    # Load data
    job_table = pd.read_parquet(job_table_path)
    ps0 = pd.read_parquet(ps0_table_path)
    ps1 = pd.read_parquet(ps1_table_path)
 
    ps0["node"] = pd.to_numeric(ps0["node"], errors="coerce")
    ps1["node"] = pd.to_numeric(ps1["node"], errors="coerce")
    ps0 = ps0.dropna(subset=["node"]).astype({"node": int})
    ps1 = ps1.dropna(subset=["node"]).astype({"node": int})

    # Collect all nodes
    job_table["nodes"] = job_table["nodes"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    nodes = set()
    for na in job_table["nodes"].values:
        if isinstance(na, (list, set, tuple)):
            nodes.update(na)

    # Create hashmaps
    with Pool(n_threads) as p:
        hashmaps_list = p.starmap(create_node_hashmap, [(node, job_table) for node in nodes])

    # Detect overlapping jobs
    ids_to_exclude = get_non_exclusive_ids(hashmaps_list)
    job_table_exclusive = job_table[~job_table["job_id"].isin(ids_to_exclude)]

    # Extract job power
    with Pool(n_threads) as p:
        job_list = p.starmap(extract_job_power, [(j, ps0, ps1) for _, j in job_table_exclusive.iterrows()])

    # Filter valid jobs and save
    job_table_valid = [j for j in job_list if len(j["power_consumption"]) > 0]
    os.makedirs(os.path.dirname(final_table_path), exist_ok=True)
    pd.DataFrame(job_table_valid).to_parquet(final_table_path)

    print(f"✅ Saved final table to {final_table_path}")

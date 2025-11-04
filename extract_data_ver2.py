#!/usr/bin/env python3
# coding: utf-8

import os
import glob
import ast
import pickle
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Iterable

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

# --------- CONFIG ----------
month = "20-12"

job_table_path = os.path.join("data", month, "year_month=20-12", "plugin=job_table",
                              "metric=job_info_marconi100", "a_0.parquet")
ps0_base_path = os.path.join("data", month, "year_month=20-12", "plugin=ipmi_pub",
                             "metric=ps0_input_power")
ps1_base_path = os.path.join("data", month, "year_month=20-12", "plugin=ipmi_pub",
                             "metric=ps1_input_power")

final_table_path = os.path.join("final_data", f"{month}.parquet")
n_threads = max(1, (os.cpu_count() or 2) - 1)  # used for ThreadPoolExecutor
sampling_time = 20  # seconds, adjust if desired
# ----------------------------

def parse_nodes_field(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return x if x is not None else []

def is_partitioned_by_node(base_path: str) -> bool:
    """Check if parquet dir contains node=* subfolders (simple heuristic)."""
    if not os.path.isdir(base_path):
        return False
    children = os.listdir(base_path)
    for c in children:
        if c.startswith("node="):
            return True
    return False

def read_parquet_partitioned_node(base_path: str, node: int) -> pd.DataFrame:
    """Read parquet files under base_path/node=<node>/* into pandas DataFrame."""
    node_dir = os.path.join(base_path, f"node={node}")
    if not os.path.isdir(node_dir):
        # Try slightly different patterns (some setups use node=0001 etc.)
        matches = glob.glob(os.path.join(base_path, f"**/node={node}*",), recursive=True)
        if matches:
            node_dir = matches[0]
        else:
            return pd.DataFrame(columns=["timestamp", "node", "value"])
    # read all parquet files under node_dir
    files = glob.glob(os.path.join(node_dir, "*.parquet"))
    if len(files) == 0:
        files = glob.glob(os.path.join(node_dir, "*.parq"))  # fallback
    if len(files) == 0:
        return pd.DataFrame(columns=["timestamp", "node", "value"])
    # use pyarrow to read faster then to_pandas
    tables = [pq.read_table(f) for f in files]
    combined = pq.concat_tables(tables)
    df = combined.to_pandas()
    return df

def read_parquet_filter_by_node_stream(base_path: str, node: int, batch_size: int = 1_000_000) -> pd.DataFrame:
    """
    If data is NOT partitioned, read file in batches and filter rows where node == node.
    This avoids loading the entire file into memory.
    Assumes single file under base_path (or multiple - will iterate).
    """
    # gather parquet files under base_path
    candidates = []
    if os.path.isdir(base_path):
        # find parquet files recursively
        candidates = glob.glob(os.path.join(base_path, "**", "*.parquet"), recursive=True)
    else:
        if os.path.exists(base_path):
            candidates = [base_path]

    frames = []
    for f in candidates:
        try:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=batch_size):
                tb = batch.to_table()
                df_batch = tb.to_pandas()
                if "node" in df_batch.columns:
                    df_f = df_batch.loc[df_batch["node"] == node]
                    if not df_f.empty:
                        frames.append(df_f)
        except Exception:
            # fallback to pandas read for this file if pyarrow fails
            try:
                df_iter = pd.read_parquet(f, engine="pyarrow")
                df_f = df_iter.loc[df_iter["node"] == node]
                if not df_f.empty:
                    frames.append(df_f)
            except Exception:
                continue
    if frames:
        return pd.concat(frames, ignore_index=True)
    else:
        return pd.DataFrame(columns=["timestamp", "node", "value"])

def prepare_node_power(ps_df: pd.DataFrame, sampling_time: int = 20) -> pd.Series:
    """
    Given ps dataframe with 'timestamp' and 'value', aggregate by timestamp (floor to sampling_time).
    Return a Series indexed by timestamp (as datetime) with summed 'value'.
    """
    if ps_df.empty:
        return pd.Series(dtype=float)

    # ensure timestamp is datetime
    if not np.issubdtype(ps_df["timestamp"].dtype, np.datetime64):
        ps_df["timestamp"] = pd.to_datetime(ps_df["timestamp"], utc=False, errors="coerce")
    # floor timestamps to nearest sampling_time-second boundary
    # pandas supports .dt.floor with string like '20s'
    ps_df["ts_floor"] = ps_df["timestamp"].dt.floor(f"{sampling_time}s")
    agg = ps_df.groupby("ts_floor", observed=True)["value"].sum()
    # ensure index is sorted datetime
    agg = agg.sort_index()
    return agg  # Series indexed by ts_floor

def extract_job_power_from_series(job_row: pd.Series, ps0_series: pd.Series, ps1_series: pd.Series,
                                  job_start_field="start_time", job_end_field="end_time",
                                  job_id_field="job_id", job_nodes_field="nodes") -> dict:
    """
    Extract job power given aggregated series for ps0 and ps1 (index = floored timestamps).
    Returns dict with power vector (aligned by timestamps present).
    """
    job_id = job_row[job_id_field]
    try:
        start_time = pd.to_datetime(job_row[job_start_field], errors="coerce")
        end_time = pd.to_datetime(job_row[job_end_field], errors="coerce")
        if pd.isna(start_time) or pd.isna(end_time):
            raise ValueError("Invalid start/end time")

        # floor start/end to sampling_time multiples using same approach as aggregated series
        start_floor = start_time
        end_floor = end_time

        # Select values between start and end
        # The series index is floored timestamps; select index between start_floor and end_floor inclusive
        ps0_sel = ps0_series.loc[(ps0_series.index >= start_floor) & (ps0_series.index <= end_floor)]
        ps1_sel = ps1_series.loc[(ps1_series.index >= start_floor) & (ps1_series.index <= end_floor)]

        # Align indices: union of timestamps
        if ps0_sel.empty and ps1_sel.empty:
            raise ValueError("No power data found for this job/time window.")

        all_idx = ps0_sel.index.union(ps1_sel.index).sort_values()
        ps0_aligned = ps0_sel.reindex(all_idx, fill_value=0.0)
        ps1_aligned = ps1_sel.reindex(all_idx, fill_value=0.0)
        power_values = (ps0_aligned.values + ps1_aligned.values).tolist()

        return {
            "job_id": job_id,
            "start_time": start_time,
            "end_time": end_time,
            "nodes": job_row.get(job_nodes_field, []),
            "power_consumption": power_values,
            "timestamps": [ts.to_pydatetime() for ts in all_idx]
        }
    except Exception as e:
        return {
            "job_id": job_id,
            "start_time": job_row.get(job_start_field, None),
            "end_time": job_row.get(job_end_field, None),
            "nodes": job_row.get(job_nodes_field, []),
            "power_consumption": [],
            "error": str(e)
        }

def process_node(node: int, node_jobs: pd.DataFrame, ps0_path: str, ps1_path: str,
                 partitioned0: bool, partitioned1: bool, sampling_time: int, max_workers: int):
    """
    Load ps0/ps1 for node (once), prepare aggregated series, then parallelize job extraction using threads.
    Returns list of job dicts for this node.
    """
    # 1) Load ps0_node
    if partitioned0:
        ps0_node_df = read_parquet_partitioned_node(ps0_path, node)
    else:
        ps0_node_df = read_parquet_filter_by_node_stream(ps0_path, node)

    if partitioned1:
        ps1_node_df = read_parquet_partitioned_node(ps1_path, node)
    else:
        ps1_node_df = read_parquet_filter_by_node_stream(ps1_path, node)

    # Ensure columns exist
    for df in (ps0_node_df, ps1_node_df):
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.NaT
        if "value" not in df.columns:
            df["value"] = 0.0
        if "node" not in df.columns:
            df["node"] = node

    # 2) prepare aggregated time series per node
    ps0_series = prepare_node_power(ps0_node_df, sampling_time=sampling_time)
    ps1_series = prepare_node_power(ps1_node_df, sampling_time=sampling_time)

    # 3) For each job in node_jobs, extract power (parallel by threads)
    results = []
    jobs_iter = list(node_jobs.itertuples(index=False, name="JobRow"))
    if len(jobs_iter) == 0:
        return results

    workers = min(max_workers, len(jobs_iter))
    with ThreadPoolExecutor(max_workers=workers) as exc:
        futures = {exc.submit(extract_job_power_from_series, pd.Series(row._asdict()), ps0_series, ps1_series): row for row in jobs_iter}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"jobs node={node}", leave=False):
            res = fut.result()
            results.append(res)
    return results

def main():
    os.makedirs(os.path.dirname(final_table_path), exist_ok=True)
    print(f"Loading job table from {job_table_path} ...")
    job_table = pd.read_parquet(job_table_path)
    # Parse nodes field
    job_table["nodes"] = job_table["nodes"].apply(parse_nodes_field)
    # Ensure job_id col exists
    if "job_id" not in job_table.columns:
        job_table = job_table.reset_index().rename(columns={"index": "job_id"})

    # Build node -> job rows mapping (only exclusive jobs assumed already filtered)
    # If your script still has overlap detection, run that prior to this script.
    # Explode nodes to have one row per (job, node)
    exploded = job_table.explode("nodes").dropna(subset=["nodes"])
    # cast node to int if possible
    try:
        exploded["nodes"] = exploded["nodes"].astype(int)
    except Exception:
        pass

    nodes = exploded["nodes"].unique().tolist()
    print(f"Found {len(nodes)} unique nodes in job table.")

    partitioned0 = is_partitioned_by_node(ps0_base_path)
    partitioned1 = is_partitioned_by_node(ps1_base_path)
    if partitioned0 or partitioned1:
        print(f"Detected partitioned data: ps0 partitioned={partitioned0}, ps1 partitioned={partitioned1}")

    final_results = []
    # Node-level loop with tqdm
    for node in tqdm(nodes, desc="Processing nodes"):
        node_jobs = exploded.loc[exploded["nodes"] == node]
        # process node (load ps0/ps1 once, then parallelize job extraction via threads)
        node_results = process_node(node=int(node),
                                    node_jobs=node_jobs,
                                    ps0_path=ps0_base_path,
                                    ps1_path=ps1_base_path,
                                    partitioned0=partitioned0,
                                    partitioned1=partitioned1,
                                    sampling_time=sampling_time,
                                    max_workers=n_threads)
        final_results.extend(node_results)

    # Filter only valid ones with power data
    job_table_valid = [r for r in final_results if r.get("power_consumption")]
    if len(job_table_valid) == 0:
        print("No valid jobs with power data found.")
    else:
        df_out = pd.DataFrame(job_table_valid)
        # Save parquet
        df_out.to_parquet(final_table_path, index=False)
        print(f"✅ Saved final table to {final_table_path} - {len(df_out)} jobs")

if __name__ == "__main__":
    main()

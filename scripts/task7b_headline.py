"""Task 7B corrected-randomized T-Drive headline battery (restart safe)."""
from __future__ import annotations

import argparse, json, logging, sys
from pathlib import Path

import psycopg
import torch
from psycopg import sql

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.multiseed_accuracy_battery import (consolidate, estimator_configs, fingerprint,
    predicate_dict, query_dict, run_estimator)
from src.eval_module.scalability import bulk_load_trajectories, generate_predicates, label_predicates
from src.results_io import write_results_json
from src.utils.db import get_connection

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "run/real_experiments/task7_dataset_validity/prepared/tdrive_corrected__max_1000000.parquet"
OUT = ROOT / "run/real_experiments/task7b/headline"
TABLE = "tdrive_traj"
MODEL_SEEDS=(1,2,3,4,5); SAMPLE_SEEDS=(17,23,47,59,71)
TRAIN_SEED=42; EVAL_SEED=43; N_EVAL=300
SPECS={"fixed_extent":(800,0.45,24.0), "mixed_extent":(6400,0.45,24.0)}

def prepare_table(conn, pool):
    with conn.cursor() as cur:
        cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(TABLE)))
        cur.execute(sql.SQL("CREATE TABLE {} (trajectory_id int, traj tgeompoint)").format(sql.Identifier(TABLE)))
    n=bulk_load_trajectories(conn,TABLE,pool[pool.trajectory_id < 100000])
    with conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE INDEX tdrive_traj_gist ON {} USING GIST (traj)").format(sql.Identifier(TABLE)))
        cur.execute(sql.SQL("ALTER TABLE {} ALTER COLUMN traj SET STATISTICS 1000").format(sql.Identifier(TABLE)))
        cur.execute(sql.SQL("ANALYZE {}").format(sql.Identifier(TABLE)))
    if n != 100000: raise RuntimeError(f"expected 100000 trajectories, loaded {n}")

def workload(conn, out, name, bbox, times, ntrain, extent, hours):
    path=out/"workloads"/f"tdrive__{name}.json"
    if path.exists():
        from scripts.multiseed_accuracy_battery import query_from_dict
        p=json.loads(path.read_text()); return [query_from_dict(x) for x in p["training_queries"]], [query_from_dict(x) for x in p["evaluation_queries"]], p
    tp=generate_predicates(name,ntrain,TRAIN_SEED,bbox,times,extent,hours)
    ep=generate_predicates(name,N_EVAL,EVAL_SEED,bbox,times,extent,hours)
    train,tms=label_predicates(conn,TABLE,tp); ev,ems=label_predicates(conn,TABLE,ep)
    tr=[query_dict(x) for x in train]; er=[query_dict(x) for x in ev]
    p={"dataset":"tdrive","table":TABLE,"workload":name,"training_seed":TRAIN_SEED,
       "evaluation_seed":EVAL_SEED,"n_train":len(tr),"n_eval":len(er),
       "training_workload_id":fingerprint(tr),"evaluation_workload_id":fingerprint(er),
       "training_queries":tr,"evaluation_queries":er,
       "training_label_ms":tms,"evaluation_label_ms":ems,
       "training_zero_count":sum(x["true_cardinality"]==0 for x in tr),
       "evaluation_zero_count":sum(x["true_cardinality"]==0 for x in er)}
    path.parent.mkdir(parents=True,exist_ok=True); write_results_json(p,path); return train,ev,p

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",type=Path,default=OUT); ap.add_argument("--dsn",default="postgresql://postgres@localhost:5434/mobilitydb"); ap.add_argument("--epochs",type=int,default=1000); ap.add_argument("--skip-table-load",action="store_true"); a=ap.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s"); torch.set_num_threads(4)
    import pandas as pd
    pool=pd.read_parquet(POOL); bbox=(float(pool.lon.min()),float(pool.lon.max()),float(pool.lat.min()),float(pool.lat.max())); times=(pool.timestamp.min().to_pydatetime(),pool.timestamp.max().to_pydatetime())
    a.output_dir.mkdir(parents=True,exist_ok=True); (a.output_dir/"runs").mkdir(exist_ok=True); (a.output_dir/"models").mkdir(exist_ok=True)
    conn=get_connection(a.dsn)
    try:
        if not a.skip_table_load: prepare_table(conn,pool)
        write_results_json({"task":"7B","dataset":"corrected randomized T-Drive","pool":str(POOL.relative_to(ROOT)),"sampling_seed":70707,"source_taxis_selected":10047,"contributing_taxis":9992,"headline_scale":100000,"model_seeds":list(MODEL_SEEDS),"sampling_seeds":list(SAMPLE_SEEDS),"training_seed":TRAIN_SEED,"evaluation_seed":EVAL_SEED},a.output_dir/"experiment_metadata.json")
        for name,(budget,extent,hours) in SPECS.items():
            train,ev,meta=workload(conn,a.output_dir,name,bbox,times,budget,extent,hours)
            for est,seed,factory in estimator_configs(MODEL_SEEDS,SAMPLE_SEEDS,a.epochs):
                sl="deterministic" if seed is None else str(seed); rp=a.output_dir/"runs"/f"tdrive__{name}__{est}__{sl}.json"
                if rp.exists(): continue
                logging.info("%s %s seed=%s",name,est,sl)
                run=run_estimator(conn,TABLE,train,ev,meta,"tdrive",name,est,seed,factory); write_results_json(run,rp)
        consolidate(a.output_dir)
    finally: conn.close()

if __name__ == "__main__": main()

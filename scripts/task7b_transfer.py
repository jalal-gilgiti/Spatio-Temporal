"""Two-direction Task 7B transfer sanity check (not broad generalization)."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.real_experiment_steps import run_cross_city_transfer_matrix
from src.results_io import write_results_json
from src.utils.db import get_connection

def main():
    p=argparse.ArgumentParser(); p.add_argument("--dsn",default="postgresql://postgres@localhost:5434/mobilitydb"); p.add_argument("--output",type=Path,default=Path("run/real_experiments/task7b/transfer_sanity.json")); p.add_argument("--n-train",type=int,default=800); p.add_argument("--n-eval",type=int,default=300); a=p.parse_args()
    c=get_connection(a.dsn)
    try: result=run_cross_city_transfer_matrix(c,{"tdrive":"tdrive_traj","porto":"porto_traj"},n_train=a.n_train,n_eval=a.n_eval,epochs=1000,seed=42)
    finally: c.close()
    result["claim_scope"]="two-direction cross-dataset transfer sanity check; not evidence of broad cross-city generalization"
    a.output.parent.mkdir(parents=True,exist_ok=True); write_results_json(result,a.output)
if __name__=="__main__": main()

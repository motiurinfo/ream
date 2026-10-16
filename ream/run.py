"""CLI: python -m ream.run --dataset maldroid [--quick]"""
import argparse
from .core import config as C
from .contributions import c1_reliability as c1


def main():
    ap=argparse.ArgumentParser(description="Run corrected reliability pipeline.")
    ap.add_argument("--dataset", required=True, choices=["maldroid","cccs","krono_real","krono_emu"])
    ap.add_argument("--quick", action="store_true", help="reduced-scale validation run")
    args=ap.parse_args()
    if args.quick:
        q=C.quick_mode()
        c1.run_dataset(args.dataset, models=q["MODELS"], seeds=q["SEEDS"],
                       n_shap=q["N_SHAP"], n_lime=q["N_LIME"])
    else:
        c1.run_dataset(args.dataset)


if __name__=="__main__":
    main()

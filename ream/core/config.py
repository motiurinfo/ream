"""Central configuration. Edit paths/scale here; everything else imports from this."""
import os

SEED = 42
TEST_SIZE = 0.30
DATA_DIR = os.environ.get("XAI_DATA_DIR", "../data")
RESULTS_DIR = os.environ.get("XAI_RESULTS_DIR", "results")

# ---- experiment scale (reduce for smoke tests) ----
MODELS = ["lgbm", "rf", "mlp"]        # detector families
N_REPEATS = 5                          # seeds -> confidence intervals
BASELINES = ["median", "zero", "classmean", "random"]   # faithfulness masking
N_SHAP = 300                           # instances explained per cell (from common set)
N_LIME = 60                            # subset also getting LIME
LIME_MODES = [True, False]             # discretized / continuous
SEEDS = [SEED + i for i in range(N_REPEATS)]

# ---- MalDroid ----
MALDROID_CLASS_NAMES = {1:"Adware",2:"Banking",3:"SMS",4:"Riskware",5:"Benign"}
MALDROID_MODALITIES = {
    "M1":"feature_vectors_syscalls_frequency_5_Cat.csv",
    "M2":"feature_vectors_syscallsbinders_frequency_5_Cat.csv",
    "M3":"feature_vectors_static.csv",
}
STATIC_SELECT = "freq"; STATIC_MIN_COUNT = 10; STATIC_TOPK = 5000

# ---- CCCS (dynamic-only) ----
CCCS_BLOCKS = {"D1_API":["API_"], "D2_Memory":["Memory_"],
               "D3_NetSys":["Network_","Logcat_","Battery_","Process_"]}
CCCS_META = ["Hash","Category","Family"]
CCCS_DROP = {"No_Category","Zero_Day","Zeroday","NoCategory"}
CCCS_MINCLASS = 50


def quick_mode():
    """Return a reduced-scale config dict for validation runs."""
    return dict(MODELS=["lgbm","rf"], SEEDS=[SEED,SEED+1], N_SHAP=40, N_LIME=15)

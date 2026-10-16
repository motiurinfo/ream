"""Contribution 3 — adversarial explanation robustness.

Threat model: attacker perturbs an app to PRESERVE the malware verdict (same
predicted class, within a confidence tolerance) but MAXIMALLY DESTABILIZE its
explanation (change the top-k feature set). An analyst reviewing the alert is then
misled about *why* it fired, even though detection is unaffected.

We use a validity-preserving random-search attack (works for any model, incl.
tree ensembles where gradients are unavailable). For each instance:
  - propose perturbations that keep the predicted class and |Δp| <= conf_tol,
  - keep the one that most reduces top-k Jaccard of the explanation,
  - iterate for a budget of queries.
Report: attack success (top-k Jaccard drop), per modality/model, vs a benign
(non-adversarial) perturbation baseline of equal magnitude.
"""
import numpy as np
from ..core.metrics import perturb_instance, topk_jaccard, adjusted_jaccard
from ..core.explainers import ShapExplainer


def _pred_class_prob(clf, x, c):
    p=clf.predict_proba(x.reshape(1,-1))[0]; return int(np.argmax(p)), float(p[c])


def attack_instance(clf, explain_fn, x, c, is_binary, d, rng,
                    budget=40, frac=0.05, conf_tol=0.10, k=10):
    """Return (best_adv_jaccard, benign_jaccard, success) for one instance.
    Lower adversarial Jaccard = more successful explanation destabilization while
    keeping the prediction."""
    base_attr=explain_fn(x)
    pred0,p0=_pred_class_prob(clf,x,c)
    best_j=1.0; found=False
    for _ in range(budget):
        xp=perturb_instance(x,is_binary,rng,frac=frac)
        pred,p=_pred_class_prob(clf,xp,c)
        if pred!=pred0 or abs(p-p0)>conf_tol:   # verdict must be preserved
            continue
        j=topk_jaccard(base_attr, explain_fn(xp), k)
        if j<best_j: best_j=j; found=True
    # benign baseline: one random valid perturbation's explanation change
    xb=perturb_instance(x,is_binary,rng,frac=frac)
    benign_j=topk_jaccard(base_attr, explain_fn(xb), k)
    return best_j, benign_j, found


def run_attack(clf, model_kind, X_train_np, x_list, c_list, is_binary, d,
               budget=40, seed=0):
    """Run the attack over a list of instances; return summary stats."""
    se=ShapExplainer(clf, model_kind, X_background=X_train_np)
    rng=np.random.default_rng(seed)
    adv, ben = [], []
    for x,c in zip(x_list,c_list):
        aj,bj,ok=attack_instance(clf, lambda z: se.attributions(z,c), x, c, is_binary, d, rng, budget=budget)
        adv.append(aj); ben.append(bj)
    adv=np.array(adv); ben=np.array(ben)
    return dict(mean_adv_jaccard=float(adv.mean()), mean_benign_jaccard=float(ben.mean()),
                mean_drop=float(ben.mean()-adv.mean()),
                # attack "success rate": fraction where adversarial top-10 overlap < 0.5
                attack_success_rate=float((adv<0.5).mean()), n=len(adv))

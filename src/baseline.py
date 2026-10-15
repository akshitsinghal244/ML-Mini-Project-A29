"""Classical baseline for bird species identification (CUB-200-2011).

Features : HSV colour histogram + HOG + LBP (texture)
Models   : Linear SVM, RBF SVM, Random Forest, k-NN
Metrics  : top-1 / top-5 accuracy, confusion matrix (best model), results.json

Usage:
    python src/baseline.py --root data/CUB_200_2011 --num-classes 20 --crop
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

SIZE = 128  # images are resized to SIZE x SIZE before feature extraction


# ----------------------------------------------------------------- data ----
def read_pairs(path):
    with open(path) as f:
        return [line.split(maxsplit=1) for line in f.read().splitlines()]


def load_metadata(root: Path):
    """Return dict: id -> (relative path, class id, is_train, bbox)."""
    paths = {int(i): p for i, p in read_pairs(root / "images.txt")}
    labels = {int(i): int(c) for i, c in read_pairs(root / "image_class_labels.txt")}
    split = {int(i): int(t) for i, t in read_pairs(root / "train_test_split.txt")}
    bbox = {int(i): tuple(map(float, b.split())) for i, b in read_pairs(root / "bounding_boxes.txt")}
    classes = {int(i): n for i, n in read_pairs(root / "classes.txt")}
    meta = {i: (paths[i], labels[i], split[i], bbox[i]) for i in paths}
    return meta, classes


def select_classes(classes, n, seed):
    ids = sorted(classes)
    if n <= 0 or n >= len(ids):
        return ids
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(ids, size=n, replace=False).tolist())


# ------------------------------------------------------------- features ----
def extract(img_path, bbox, crop):
    img = Image.open(img_path).convert("RGB")
    if crop:  # bbox = x, y, w, h
        x, y, w, h = bbox
        img = img.crop((int(x), int(y), int(x + w), int(y + h)))
    img = img.resize((SIZE, SIZE))

    # 1) colour: 3D HSV histogram, 8 bins per channel -> 512 dims
    hsv = np.asarray(img.convert("HSV"))
    hist, _ = np.histogramdd(hsv.reshape(-1, 3), bins=(8, 8, 8), range=((0, 256),) * 3)
    hist = hist.ravel() / hist.sum()

    gray = np.asarray(img.convert("L"))

    # 2) shape/edges: HOG
    h = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))

    # 3) texture: uniform LBP histogram (P=8 -> 10 bins)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)

    return np.concatenate([hist, h, lbp_hist]).astype(np.float32)


def build_features(root, meta, ids, crop, cache):
    if cache.exists():
        print(f"Loading cached features: {cache}")
        d = np.load(cache)
        if list(d["ids"]) == ids:
            return d["X"]
    print(f"Extracting features for {len(ids)} images ...")
    t = time.time()
    X = Parallel(n_jobs=-1)(
        delayed(extract)(root / "images" / meta[i][0], meta[i][3], crop) for i in ids
    )
    X = np.stack(X)
    np.savez_compressed(cache, X=X, ids=np.array(ids))
    print(f"  done in {time.time() - t:.1f}s, feature dim = {X.shape[1]}")
    return X


# ------------------------------------------------------------- evaluate ----
def scores(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    return clf.decision_function(X)


def topk(clf, X, y, k=5):
    s = scores(clf, X)
    k = min(k, s.shape[1])
    top = np.argsort(-s, axis=1)[:, :k]
    classes = clf.classes_
    return float(np.mean([y[i] in classes[top[i]] for i in range(len(y))]))


def models(seed, n_pc):
    return {
        "LinearSVM": make_pipeline(StandardScaler(), PCA(n_components=n_pc, random_state=seed),
                                   LinearSVC(C=0.01, max_iter=1000, random_state=seed)),
        "RBF-SVM": make_pipeline(StandardScaler(), PCA(n_components=0.95, random_state=seed), SVC(C=10, gamma="scale")),
        "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed),
        "kNN": make_pipeline(StandardScaler(), PCA(n_components=0.95, random_state=seed),
                             KNeighborsClassifier(n_neighbors=5, weights="distance")),
    }


def save_confusion(cm, names, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (best baseline)")
    if len(names) <= 25:
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=6)
        ax.set_yticks(range(len(names)), names, fontsize=6)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def top_confused(cm, names, n=5):
    cm = cm.copy()
    np.fill_diagonal(cm, 0)
    flat = np.dstack(np.unravel_index(np.argsort(-cm.ravel())[:n], cm.shape))[0]
    return [(names[a], names[b], int(cm[a, b])) for a, b in flat if cm[a, b] > 0]


# ----------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path to CUB_200_2011 folder")
    ap.add_argument("--num-classes", type=int, default=20, help="0 = all 200")
    ap.add_argument("--crop", action="store_true", help="crop to bounding box")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out)
    out.mkdir(exist_ok=True, parents=True)

    meta, classes = load_metadata(root)
    chosen = select_classes(classes, a.num_classes, a.seed)
    ids = sorted(i for i, m in meta.items() if m[1] in chosen)
    print(f"{len(chosen)} classes, {len(ids)} images")

    tag = f"n{len(chosen)}_s{a.seed}_{'crop' if a.crop else 'full'}"
    X = build_features(root, meta, ids, a.crop, out / f"features_{tag}.npz")
    y = np.array([meta[i][1] for i in ids])
    is_train = np.array([meta[i][2] for i in ids]) == 1
    Xtr, ytr, Xte, yte = X[is_train], y[is_train], X[~is_train], y[~is_train]
    print(f"train={len(ytr)}  test={len(yte)}  chance={1 / len(chosen):.3f}")

    results, best = {}, (None, -1, None)
    n_pc = min(256, Xtr.shape[0] - 1, Xtr.shape[1])
    for name, clf in models(a.seed, n_pc).items():
        print(f"fitting {name} ...", flush=True)
        t = time.time()
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        r = {"top1": float(np.mean(pred == yte)), "top5": topk(clf, Xte, yte), "fit_s": round(time.time() - t, 1)}
        results[name] = r
        print(f"{name:13s} top1={r['top1']:.3f}  top5={r['top5']:.3f}  ({r['fit_s']}s)")
        if r["top1"] > best[1]:
            best = (name, r["top1"], pred)

    names = [classes[c].split(".", 1)[-1] for c in chosen]
    cm = confusion_matrix(yte, best[2], labels=chosen)
    save_confusion(cm, names, out / f"confusion_{tag}.png")
    results["_meta"] = {"classes": len(chosen), "train": int(len(ytr)), "test": int(len(yte)),
                        "crop": a.crop, "best": best[0], "most_confused": top_confused(cm, names)}
    (out / f"results_{tag}.json").write_text(json.dumps(results, indent=2))
    print(f"\nBest: {best[0]} ({best[1]:.3f}). Saved to {out}/")


if __name__ == "__main__":
    main()

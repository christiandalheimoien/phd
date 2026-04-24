# coeff_field_simple.py  — now with dataset-size dependent normalization
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

_EPS = 1e-12

# ---------- utilities ----------

def _iqr(a, axis=0):
    q75 = np.percentile(a, 75, axis=axis)
    q25 = np.percentile(a, 25, axis=axis)
    return q75 - q25

def _robust_center_scale(X):
    center = np.median(X, axis=0)
    scale = _iqr(X, axis=0)
    scale = np.where(scale <= 1e-12, 1.0, scale)  # avoid zeros
    return center, scale

def _transform(X, center, scale):
    return (X - center) / scale

def _map_far_to_close(distances, d_close, d_far, *, use_log=True, sharpness=1.0):
    d = np.maximum(distances, _EPS) # avoid log(0)
    if use_log:
        d = np.log(d)
        d_close = np.log(max(d_close, _EPS))
        d_far = np.log(max(d_far, _EPS))
    t = (d_far - d) / max(d_far - d_close, 1e-12)   # 1 when very close, 0 when very far
    t = np.clip(t, 0.0, 1.0)
    t = t ** float(sharpness)                   # faster drop with larger sharpness
    return t

def _dataset_range_scale(n_rows: int, n1: int, n2: int) -> float:
    """
    Log-linear scaling of usable coefficient range with dataset size N:
      N < n1      -> 0
      N >= n2     -> 1
      log-linear curve in between.
    """
    if n_rows < n1:
        return 0.0
    if n_rows >= n2:
        return 1.0  
    y = (np.log10(n_rows)-np.log10(n1))/ (np.log10(n2)-np.log10(n1))
    return y

# ---------- plain model container ----------

class CoeffModel:
    def __init__(self, *, knn, center, scale, k, infer_min, infer_max,
                 d_min, d_max, use_log=True, sharpness=1.0, size_scale=1.0):
        self.knn = knn
        self.center = center
        self.scale = scale
        self.k = int(k)
        self.infer_min = infer_min
        self.infer_max = infer_max
        self.d_min = float(d_min)  # global closest avg kNN distance (widened box)
        self.d_max = float(d_max)  # global median avg kNN distance (widened box)
        self.use_log = bool(use_log)
        self.sharpness = float(sharpness)
        self.size_scale = float(size_scale)  # dataset-size dependent scaling

# ---------- build ----------

def build_coeff_model(data, k=12, sample_size=10_000, *, use_log=True, sharpness=1.0, n1=50, n2=5000, verbose=True):
    data = np.asarray(data, float)
    if not (data.ndim == 2 and data.shape[1] == 4):
        raise ValueError("Expect data of shape (N, 4)")

    # Robust scaling
    center, scale = _robust_center_scale(data)
    Z = _transform(data, center, scale)

    # kNN
    k = int(max(1, min(k, max(1, len(data) - 1))))
    knn = NearestNeighbors(metric="euclidean").fit(Z)

    # Global inference box: 25% widened min and max per dimension
    widen = 0.25
    mins = data.min(axis=0); maxs = data.max(axis=0)
    width = np.maximum(maxs - mins, 1e-12)
    pad = widen * width
    infer_min = mins - pad
    infer_max = maxs + pad

    # Sample uniformly in widened box and get avg kNN distances
    rng = np.random.default_rng(42)
    U = rng.random((sample_size, 4))
    samples = infer_min + U * (infer_max - infer_min)
    Zs = _transform(samples, center, scale)
    dists, _ = knn.kneighbors(Zs, n_neighbors=k)
    avg_d = dists.mean(axis=1)
    d_min = float(np.min(avg_d))
    d_max = float(np.percentile(avg_d, 50)) # <-- Use 50th percentile ()= median) for d_far
    if not np.isfinite(d_min) or not np.isfinite(d_max) or d_max <= d_min:
        d_min = float(np.median(avg_d)); d_max = float(np.max(avg_d))

    # dataset-size dependent scaling factor
    size_scale = _dataset_range_scale(len(data), n1, n2)
    if verbose:
        print("=== Build summary ===")
        print(f"k={k}, samples={sample_size}, sharpness={sharpness}")
        print(f"Inference box min: {infer_min}")
        print(f"Inference box max: {infer_max}")
        print(f"Avg kNN distance: min={d_min:.4g}, max={d_max:.4g}")
        print(f"Dataset size={len(data)} -> size_scale={size_scale:.3f} (0..1 multiplier)")

    return CoeffModel(knn=knn, center=center, scale=scale, k=k,
                      infer_min=infer_min, infer_max=infer_max,
                      d_min=d_min, d_max=d_max,
                      use_log=use_log, sharpness=sharpness, size_scale=size_scale)

# ---------- inference ----------

def coefficient_for_point(model: CoeffModel, x):
    x = np.asarray(x, float).reshape(1, 4)
    z = _transform(x, model.center, model.scale)
    dists, _ = model.knn.kneighbors(z, n_neighbors=model.k)
    avg_d = dists.mean(axis=1)
    base = _map_far_to_close(avg_d, model.d_min, model.d_max,
                             use_log=model.use_log, sharpness=model.sharpness)[0]
    return float(model.size_scale * base)  # scaled by dataset size

def coefficient_for_point_no_size_scale(model: CoeffModel, x):
    x = np.asarray(x, float).reshape(1, 4)
    z = _transform(x, model.center, model.scale)
    dists, _ = model.knn.kneighbors(z, n_neighbors=model.k)
    avg_d = dists.mean(axis=1)
    base = _map_far_to_close(avg_d, model.d_min, model.d_max,
                             use_log=model.use_log, sharpness=model.sharpness)[0]
    return float(base)  # without scaling by dataset size

# ---------- plotting ----------

def plot_coefficient_field(model: CoeffModel, data, dim1, dim2,
                           fixed_dim_values, resolution=120, cmap="gray"):
    data = np.asarray(data, float)
    dims = [0, 1, 2, 3]
    if dim1 == dim2 or dim1 not in dims or dim2 not in dims:
        raise ValueError("dim1 and dim2 must be two different integers in {0,1,2,3}.")
    other = [d for d in dims if d not in (dim1, dim2)]
    for d in other:
        if d not in fixed_dim_values:
            raise ValueError(f"Provide fixed value for dimension {d} via fixed_dim_values.")

    # Grid from the global widened box
    x_min, x_max = model.infer_min[dim1], model.infer_max[dim1]
    y_min, y_max = model.infer_min[dim2], model.infer_max[dim2]
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    XX, YY = np.meshgrid(xs, ys)

    # Build queries
    grid = np.zeros((resolution * resolution, 4))
    grid[:, dim1] = XX.ravel()
    grid[:, dim2] = YY.ravel()
    for d in other:
        grid[:, d] = fixed_dim_values[d]

    Zq = _transform(grid, model.center, model.scale)
    dists, _ = model.knn.kneighbors(Zq, n_neighbors=model.k)
    avg_d = dists.mean(axis=1)
    base = _map_far_to_close(avg_d, model.d_min, model.d_max,
                             use_log=model.use_log, sharpness=model.sharpness)
    coefs = model.size_scale * base                    # scaled by dataset size
    field = coefs.reshape(resolution, resolution)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(xs, ys, field, shading="auto", cmap=cmap)
    cbar = plt.colorbar(im, ax=ax); cbar.set_label("Coefficient (0=far, 1=close) × size_scale")
    ax.set_xlabel(f"Dim {dim1}"); ax.set_ylabel(f"Dim {dim2}")
    ax.set_title(
        "Coefficient field; fixed dims {} at ".format(other) +
        ", ".join([f"dim{d}={fixed_dim_values[d]}" for d in other]) +
        f"   (sharpness={model.sharpness:g}, size_scale={model.size_scale:.3f})"
    )
    ax.scatter(data[:, dim1], data[:, dim2], s=6, c="tab:blue", alpha=0.7, edgecolors="none")
    plt.tight_layout()
    return fig, ax

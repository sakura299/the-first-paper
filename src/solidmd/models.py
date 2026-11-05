"""Small ridge model, using only training statistics and the standard library."""
import math


def solve(matrix, rhs):
    """Gaussian elimination with partial pivoting for small dense systems."""
    size = len(rhs)
    augmented = [list(row) + [value] for row, value in zip(matrix, rhs)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(augmented[row][col]))
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        if abs(scale) < 1e-14:
            raise ValueError("Singular system; use positive ridge_alpha")
        augmented[col] = [value / scale for value in augmented[col]]
        for row in range(size):
            if row != col:
                factor = augmented[row][col]
                augmented[row] = [a - factor * b for a, b in zip(augmented[row], augmented[col])]
    return [row[-1] for row in augmented]


def fit_ridge(x, y, alpha=0.01):
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("ridge_alpha must be finite and >0")
    width, count = len(x[0]), len(x)
    means = [sum(row[k] for row in x) / count for k in range(width)]
    scales = [max(math.sqrt(sum((row[k] - means[k]) ** 2 for row in x) / count), 1e-12)
              for k in range(width)]
    design = [[1.0] + [(v - means[k]) / scales[k] for k, v in enumerate(row)] for row in x]
    gram = [[sum(row[i] * row[j] for row in design) / count
             + (alpha if i == j and i > 0 else 0.0)
             for j in range(width + 1)] for i in range(width + 1)]
    weights = [solve(gram, [sum(row[i] * target[k] for row, target in zip(design, y)) / count
                            for i in range(width + 1)]) for k in range(3)]
    return {"kind": "ridge", "alpha": alpha, "feature_mean": means,
            "feature_scale": scales, "weights": weights,
            "objective": "mean_squared_error + alpha * squared_standardized_weights"}


def predict_ridge(model, x):
    rows = [[1.0] + [(v - model["feature_mean"][k]) / model["feature_scale"][k]
                     for k, v in enumerate(row)] for row in x]
    return [[sum(a * b for a, b in zip(row, weights)) for weights in model["weights"]]
            for row in rows]

"""Per-ion displacement histories with strict, disjoint frame blocks."""
import math


def make_splits(positions, species, selected="Li", history=4,
                train_fraction=0.6, validation_fraction=0.2):
    if not isinstance(history, int) or history < 1:
        raise ValueError("history must be a positive integer")
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1
            and train_fraction + validation_fraction < 1):
        raise ValueError("Invalid train/validation fractions")
    indices = [i for i, symbol in enumerate(species) if symbol == selected]
    if not indices:
        raise ValueError("Selected species is absent: " + selected)
    n = len(positions)
    a, b = int(n * train_fraction), int(n * (train_fraction + validation_fraction))
    splits = {}
    for name, start, stop in [("train", 0, a), ("validation", a, b), ("test", b, n)]:
        if stop - start < history + 2:
            raise ValueError("Too few frames in %s block for history=%s" % (name, history))
        x, y = [], []
        # An example uses frames [target-history-1, target], all within its block.
        for target in range(start + history + 1, stop):
            for atom in indices:
                x.append([positions[t][atom][k] - positions[t - 1][atom][k]
                          for t in range(target - history, target) for k in range(3)])
                y.append([positions[target][atom][k] - positions[target - 1][atom][k]
                          for k in range(3)])
        splits[name] = {"x": x, "y": y, "frame_range": [start, stop],
                        "examples": len(x)}
    return splits


def score(predictions, targets):
    if len(predictions) != len(targets) or not targets:
        raise ValueError("Prediction/target lengths differ or are empty")
    errors = [p[k] - t[k] for p, t in zip(predictions, targets) for k in range(3)]
    return {"rmse_angstrom_per_component": math.sqrt(sum(e * e for e in errors) / len(errors)),
            "mae_angstrom_per_component": sum(abs(e) for e in errors) / len(errors),
            "examples": len(targets)}

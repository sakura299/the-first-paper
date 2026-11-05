#!/usr/bin/env python3
"""Optional small PyTorch next-displacement predictor. CPU is the default."""
import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solidmd.features import make_splits, score
from solidmd.trajectory import read, unwrap
from solidmd.pipeline import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/demo.json"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/runs/transformer-demo")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    if args.output.exists() and any(args.output.iterdir()) and not args.force:
        parser.error("Output is not empty; use --force or select another directory")
    try:
        import torch
        from torch import nn
    except ImportError:
        parser.exit(2, "Install optional PyTorch in a Python 3.11/3.12 venv: pip install -e '.[transformer]'\n")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable; use --device cpu")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    source = ROOT / config["input"]
    data = read(source)
    splits = make_splits(unwrap(data), data["species"], config["species"], config["history"],
                         config["train_fraction"], config["validation_fraction"])
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    # Only training examples contribute to scaling. Scale only, retaining physical zero.
    train_x = torch.tensor(splits["train"]["x"], dtype=torch.float32).reshape(-1, config["history"], 3)
    scale = train_x.square().mean(dim=(0, 1)).sqrt().clamp_min(1e-8)
    tensors = {name: (torch.tensor(split["x"], dtype=torch.float32).reshape(-1, config["history"], 3).div(scale).to(args.device),
                      torch.tensor(split["y"], dtype=torch.float32).div(scale).to(args.device))
               for name, split in splits.items()}

    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Linear(3, 24)
            positional = torch.zeros(config["history"], 24)
            time_index = torch.arange(config["history"]).unsqueeze(1)
            frequency = torch.exp(torch.arange(0, 24, 2) * (-math.log(10000.0) / 24))
            positional[:, 0::2] = torch.sin(time_index * frequency)
            positional[:, 1::2] = torch.cos(time_index * frequency)
            self.register_buffer("positional", positional)
            layer = nn.TransformerEncoderLayer(24, 2, dim_feedforward=48, dropout=0.0, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
            self.head = nn.Linear(24, 3)

        def forward(self, x):
            # All input tokens precede the one future target; no future tokens are present.
            return self.head(self.encoder(self.project(x) + self.positional)[:, -1])

    model = TinyTransformer().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    best_loss, best_state, best_epoch = float("inf"), None, None
    losses = []
    for epoch in range(args.epochs):
        model.train()
        x, y = tensors["train"]
        order = torch.randperm(len(x))
        for batch in order.split(64):
            batch = batch.to(args.device)
            optimizer.zero_grad()
            loss = (model(x[batch]) - y[batch]).square().mean()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vx, vy = tensors["validation"]
            validation_loss = (model(vx) - vy).square().mean().item()
        if not math.isfinite(validation_loss):
            raise ValueError("Nonfinite validation loss")
        losses.append(validation_loss)
        if validation_loss < best_loss:
            best_loss, best_state, best_epoch = validation_loss, copy.deepcopy(model.state_dict()), epoch + 1
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = (model(tensors["test"][0]).cpu() * scale).tolist()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "scale": scale, "history": config["history"], "d_model": 24, "nhead": 2,
                "dim_feedforward": 48, "num_layers": 1}, args.output / "model.pt")
    report = {"config": config, "seed": seed, "torch_version": torch.__version__,
              "python": sys.version, "device": args.device, "epochs": args.epochs,
              "selected_epoch": best_epoch, "validation_normalized_mse": losses,
              "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "metadata": data["metadata"],
              "splits": {name: {"frame_range_half_open": part["frame_range"], "examples": part["examples"]}
                         for name, part in splits.items()},
              "test": score(predictions, splits["test"]["y"]),
              "limitations": "Tiny temporal predictor; no environment or symmetry encoding; synthetic smoke test is not scientific validation"}
    save_json(args.output / "report.json", report)
    print(json.dumps(report["test"], indent=2))


if __name__ == "__main__":
    main()

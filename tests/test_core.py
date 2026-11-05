import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solidmd.adapters import convert, read_lammps
from solidmd.analysis import diffusion_fit, msd
from solidmd.features import make_splits, score
from solidmd.models import fit_ridge, predict_ridge
from solidmd.pipeline import run
from solidmd.trajectory import synthetic, unwrap, validate, write


class PhysicsTests(unittest.TestCase):
    def test_boundary_crossing_in_both_directions(self):
        data = synthetic(frames=3)
        data["species"] = ["Li"]
        data["positions_angstrom"] = [[[9.8, 0.2, 0]], [[0.1, 9.9, 0]], [[0.4, 9.6, 0]]]
        data["coordinate_mode"] = "wrapped"
        result = unwrap(data)
        self.assertAlmostEqual(result[1][0][0], 10.1)
        self.assertAlmostEqual(result[2][0][1], -0.4)
        self.assertAlmostEqual(msd(result, ["Li"], 2, max_lag=2)[1]["msd_angstrom2"], 0.18)

    def test_half_box_jump_rejected(self):
        data = synthetic(frames=3)
        data["coordinate_mode"] = "wrapped"
        data["positions_angstrom"][1][0][0] = data["positions_angstrom"][0][0][0] + 5
        with self.assertRaises(ValueError):
            unwrap(data)

    def test_unwrapped_long_jump_preserved(self):
        data = synthetic(frames=3)
        data["positions_angstrom"][1][0][0] += 30
        self.assertEqual(unwrap(data), data["positions_angstrom"])

    def test_constant_velocity_msd(self):
        positions = [[[float(t), 0.0, 0.0]] for t in range(10)]
        rows = msd(positions, ["Li"], 1000, max_lag=4)
        self.assertEqual([row["msd_angstrom2"] for row in rows], [0, 1, 4, 9, 16])
        self.assertEqual(rows[2]["time_ps"], 2)

    def test_diffusion_units_and_window(self):
        rows = [{"time_ps": float(i), "msd_angstrom2": 12.0 * i + 3} for i in range(10)]
        fit = diffusion_fit(rows, 2, 8)
        self.assertAlmostEqual(fit["D_cm2_per_s"], 2e-4)
        self.assertAlmostEqual(fit["intercept_angstrom2"], 3)
        self.assertAlmostEqual(fit["r_squared"], 1)

    def test_invalid_scientific_input(self):
        for field, value in [("timestep_fs", 0), ("timestep_fs", float("nan")),
                             ("pbc", [True, True, False])]:
            data = synthetic(frames=3)
            data[field] = value
            with self.assertRaises(ValueError):
                validate(data)
        data = synthetic(frames=3)
        data["cell_angstrom"][0][1] = 1.0
        with self.assertRaises(ValueError):
            validate(data)


class LearningTests(unittest.TestCase):
    def test_split_does_not_share_any_raw_frame(self):
        # x coordinate t^2 gives unique odd displacements and identifiable frame IDs.
        positions = [[[float(t * t), 0, 0]] for t in range(100)]
        parts = make_splits(positions, ["Li"], history=2)
        frame_sets = []
        for part in parts.values():
            used = set()
            for x, y in zip(part["x"], part["y"]):
                target = int((y[0] + 1) / 2)
                used.update(range(target - 3, target + 1))
                self.assertEqual(x[0], 2 * (target - 2) - 1)
                self.assertGreaterEqual(min(used), part["frame_range"][0])
                self.assertLess(max(used), part["frame_range"][1])
            frame_sets.append(used)
        self.assertFalse(frame_sets[0] & frame_sets[1])
        self.assertFalse(frame_sets[0] & frame_sets[2])
        self.assertFalse(frame_sets[1] & frame_sets[2])

    def test_ridge_learns_known_mapping(self):
        x = [[float(i)] for i in range(-20, 21)]
        y = [[2 * row[0] + 3, -row[0], 0.5] for row in x]
        model = fit_ridge(x, y, alpha=1e-8)
        prediction = predict_ridge(model, [[2.5]])[0]
        self.assertAlmostEqual(prediction[0], 8, places=5)
        self.assertAlmostEqual(prediction[1], -2.5, places=5)
        self.assertAlmostEqual(prediction[2], 0.5, places=5)

    def test_score_units(self):
        self.assertEqual(score([[3, 0, 0]], [[0, 0, 0]])["rmse_angstrom_per_component"], math.sqrt(3))

    def test_short_split_and_missing_species_rejected(self):
        data = synthetic(frames=20)
        with self.assertRaises(ValueError):
            make_splits(unwrap(data), data["species"], history=4)
        with self.assertRaises(ValueError):
            make_splits(unwrap(data), data["species"], selected="Na")


def dump_text(steps=(0, 10, 20), unstable=False):
    blocks = []
    for index, step in enumerate(steps):
        second_id = 3 if unstable and index == 1 else 2
        blocks.append("ITEM: TIMESTEP\n%s\nITEM: NUMBER OF ATOMS\n2\n"
                      "ITEM: BOX BOUNDS pp pp pp\n-1 9\n0 10\n0 10\n"
                      "ITEM: ATOMS id type xu yu zu\n%s 2 2 3 4\n1 1 %s 1 1\n"
                      % (step, second_id, index + 8))
    return "".join(blocks)


class ImportTests(unittest.TestCase):
    def test_lammps_sorts_ids_and_preserves_unwrapped_crossing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trajectory.dump"
            path.write_text(dump_text())
            data = convert(path, "lammps", "lammps-dump", 20, "unwrapped", "run-1", type_map={"1": "Li", "2": "O"})
            self.assertEqual(data["species"], ["Li", "O"])
            self.assertEqual(data["positions_angstrom"][2][0][0], 11)
            self.assertEqual(data["timestep_fs"], 20)

    def test_lammps_rejects_irregular_time_and_changed_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trajectory.dump"
            for text in [dump_text((0, 10, 30)), dump_text(unstable=True)]:
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_lammps(path, {"1": "Li", "2": "O"})

    def test_xyz_requires_cell_and_preserves_species(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trajectory.xyz"
            path.write_text("".join("1\nsynthetic test fixture\nLi %s 0 0\n" % i for i in range(3)))
            with self.assertRaises(ValueError):
                convert(path, "cp2k", "xyz", 2, "unwrapped", "fixture")
            data = convert(path, "cp2k", "xyz", 2, "unwrapped", "fixture", box=[10, 10, 10])
            self.assertEqual(data["positions_angstrom"][2][0], [2, 0, 0])


class PipelineTests(unittest.TestCase):
    def test_end_to_end_reproducible_and_protects_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = json.loads((ROOT / "configs/demo.json").read_text())
            data = synthetic(seed=config["seed"], frames=120)
            write(root / config["input"], data)
            first = run(config, root)
            with self.assertRaises(ValueError):
                run(config, root)
            second = run(config, root, force=True)
            self.assertEqual(first["metrics"], second["metrics"])
            self.assertEqual(first["input_sha256"], second["input_sha256"])
            report = json.loads((root / config["output"] / "report.json").read_text())
            self.assertEqual(report["metadata"]["data_kind"], "synthetic")
            self.assertTrue((root / config["output"] / "test_predictions.csv").exists())


if __name__ == "__main__":
    unittest.main()

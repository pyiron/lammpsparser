# Streaming Dump Parser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `iter_lammps_frames()`, a memory-efficient public generator that yields one parsed, unit-converted `LammpsFrame` per timestep from a LAMMPS text dump file.

**Architecture:** A private `_iter_raw_frames()` generator in `output_raw.py` reads the dump file line-by-line and yields one raw frame dict at a time. The existing `parse_raw_dump_from_text` is refactored to call it internally (no behavioral change). A public `iter_lammps_frames()` generator in `output.py` wraps the raw generator and applies per-frame coordinate rotation, index remapping, and unit conversion, yielding a `LammpsFrame` dataclass.

**Tech Stack:** Python 3.9+, numpy, pandas, ase, existing `UnfoldingPrism` / `UnitConverter` / `remap_indices_ase` from this repo.

---

## File Map

| File | Change |
|---|---|
| `src/lammpsparser/output_raw.py` | Add `_iter_raw_frames`; refactor `parse_raw_dump_from_text` to use it |
| `src/lammpsparser/output.py` | Add `LammpsFrame` dataclass; add `iter_lammps_frames` generator |
| `src/lammpsparser/__init__.py` | Export `iter_lammps_frames` and `LammpsFrame` |
| `tests/test_output_raw.py` | Add tests for `_iter_raw_frames` |
| `tests/test_output.py` | Add tests for `iter_lammps_frames` |

---

## Task 1: Add `_iter_raw_frames` to `output_raw.py`

**Files:**
- Modify: `src/lammpsparser/output_raw.py`
- Test: `tests/test_output_raw.py`

The goal is a private generator that reads a dump file frame-by-frame. It accepts `start`, `stop`, `step` slice parameters (same semantics as Python's `range`). The existing `parse_raw_dump_from_text` will be refactored to call it in Task 2.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_output_raw.py`:

```python
from src.lammpsparser.output_raw import _iter_raw_frames

class TestIterRawFrames(unittest.TestCase):
    def test_yields_all_frames(self):
        frames = list(_iter_raw_frames("tests/static/jagged_dump/dump.out"))
        self.assertEqual(len(frames), 2)

    def test_frame_keys(self):
        frame = next(_iter_raw_frames("tests/static/jagged_dump/dump.out"))
        for key in ["steps", "natoms", "cells", "indices", "forces", "velocities",
                    "positions", "unwrapped_positions", "computes"]:
            self.assertIn(key, frame)

    def test_start_stop(self):
        frames = list(_iter_raw_frames("tests/static/jagged_dump/dump.out", start=1))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0]["steps"], 1)

    def test_step_slice(self):
        # jagged_dump has 2 frames; step=2 should yield only frame 0
        frames = list(_iter_raw_frames("tests/static/jagged_dump/dump.out", step=2))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0]["steps"], 0)

    def test_truncated_file_raises(self):
        import tempfile, os
        # Write a dump with a started-but-unfinished frame
        content = (
            "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\n"
            "ITEM: BOX BOUNDS pp pp pp\n-1 1\n-1 1\n-1 1\n"
            "ITEM: ATOMS id type xsu ysu zsu fx fy fz vx vy vz\n"
            "1 1 0 0 0 0 0 0 0 0 0\n"
            # second atom line missing — truncated
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                list(_iter_raw_frames(path))
        finally:
            os.unlink(path)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_output_raw.py::TestIterRawFrames -v
```

Expected: all 5 tests FAIL with `ImportError` or `AttributeError` — `_iter_raw_frames` doesn't exist yet.

- [ ] **Step 3: Implement `_iter_raw_frames` in `output_raw.py`**

Add the following function to `src/lammpsparser/output_raw.py`, just above `parse_raw_dump_from_text`:

```python
def _iter_raw_frames(
    file_name: str,
    start: int = 0,
    stop: Optional[int] = None,
    step: int = 1,
) -> "Iterator[Dict]":
    """
    Yield one raw frame dict at a time from a LAMMPS text dump file.

    Each yielded dict has the same keys as DumpData fields, but values are
    for a single frame (scalars or arrays, not lists).

    Args:
        file_name: Path to the LAMMPS text dump file.
        start: First frame index to yield (0-based, default 0).
        stop: Stop before this frame index. None means read to end.
        step: Yield every ``step``-th frame (default 1).

    Yields:
        dict with keys: steps, natoms, cells, indices, forces, mean_forces,
        velocities, mean_velocities, unwrapped_positions,
        mean_unwrapped_positions, positions, computes.

    Raises:
        ValueError: If a frame is malformed or the file is truncated mid-frame.
    """
    frame_index = 0
    with open(file_name, "r") as f:
        line = f.readline()
        while line:
            if "ITEM: TIMESTEP" not in line:
                line = f.readline()
                continue

            # --- read header ---
            try:
                timestep = int(f.readline())
            except ValueError as e:
                raise ValueError(f"Malformed TIMESTEP at frame {frame_index}") from e

            line = f.readline()
            if "ITEM: NUMBER OF ATOMS" not in line:
                raise ValueError(f"Expected NUMBER OF ATOMS at frame {frame_index}, got: {line!r}")
            try:
                n = int(f.readline())
            except ValueError as e:
                raise ValueError(f"Malformed NUMBER OF ATOMS at frame {frame_index}") from e

            line = f.readline()
            if "ITEM: BOX BOUNDS" not in line:
                raise ValueError(f"Expected BOX BOUNDS at frame {frame_index}, got: {line!r}")
            try:
                c1 = np.fromstring(f.readline(), dtype=float, sep=" ")
                c2 = np.fromstring(f.readline(), dtype=float, sep=" ")
                c3 = np.fromstring(f.readline(), dtype=float, sep=" ")
            except Exception as e:
                raise ValueError(f"Malformed BOX BOUNDS at frame {frame_index}") from e
            cell = to_amat(np.concatenate([c1, c2, c3]))

            line = f.readline()
            if "ITEM: ATOMS" not in line:
                raise ValueError(f"Expected ITEM: ATOMS at frame {frame_index}, got: {line!r}")
            columns = line.lstrip("ITEM: ATOMS").split()

            # --- read atom data ---
            buf = StringIO()
            for i in range(n):
                atom_line = f.readline()
                if not atom_line:
                    raise ValueError(
                        f"Truncated dump file: expected {n} atoms at frame {frame_index} "
                        f"(step {timestep}), got {i}"
                    )
                buf.write(atom_line)
            buf.seek(0)

            # --- decide whether to yield this frame ---
            in_range = (
                frame_index >= start
                and (stop is None or frame_index < stop)
                and (frame_index - start) % step == 0
            )

            if in_range:
                df = pd.read_csv(
                    buf,
                    nrows=n,
                    sep="\\s+",
                    header=None,
                    names=columns,
                    engine="c",
                )
                df.sort_values(by="id", ignore_index=True, inplace=True)

                frame: Dict = {
                    "steps": timestep,
                    "natoms": n,
                    "cells": cell,
                    "indices": df["type"].array.astype(int),
                    "forces": np.stack([df["fx"].array, df["fy"].array, df["fz"].array], axis=1),
                    "mean_forces": np.stack([
                        df["f_mean_forces[1]"].array,
                        df["f_mean_forces[2]"].array,
                        df["f_mean_forces[3]"].array,
                    ], axis=1) if "f_mean_forces[1]" in columns else np.array([]),
                    "velocities": np.stack([
                        df["vx"].array, df["vy"].array, df["vz"].array
                    ], axis=1) if all(c in columns for c in ("vx", "vy", "vz")) else np.array([]),
                    "mean_velocities": np.stack([
                        df["f_mean_velocities[1]"].array,
                        df["f_mean_velocities[2]"].array,
                        df["f_mean_velocities[3]"].array,
                    ], axis=1) if "f_mean_velocities[1]" in columns else np.array([]),
                    "computes": {},
                }

                if "xsu" in columns:
                    direct = np.stack([df["xsu"].array, df["ysu"].array, df["zsu"].array], axis=1)
                    frame["unwrapped_positions"] = direct
                    frame["positions"] = direct - np.floor(direct)
                else:
                    frame["unwrapped_positions"] = np.array([])
                    frame["positions"] = np.array([])

                if "f_mean_positions[1]" in columns:
                    frame["mean_unwrapped_positions"] = np.stack([
                        df["f_mean_positions[1]"].array,
                        df["f_mean_positions[2]"].array,
                        df["f_mean_positions[3]"].array,
                    ], axis=1)
                else:
                    frame["mean_unwrapped_positions"] = np.array([])

                for k in columns:
                    if k.startswith("c_"):
                        frame["computes"][k.replace("c_", "")] = df[k].array

                yield frame

            frame_index += 1
            line = f.readline()
```

Also add `Optional` and `Iterator` to the imports at the top of `output_raw.py`:

```python
from typing import Dict, Iterator, List, Optional, Union
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_output_raw.py::TestIterRawFrames -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lammpsparser/output_raw.py tests/test_output_raw.py
git commit -m "feat: add _iter_raw_frames generator to output_raw"
```

---

## Task 2: Refactor `parse_raw_dump_from_text` to use `_iter_raw_frames`

**Files:**
- Modify: `src/lammpsparser/output_raw.py`
- Test: `tests/test_output_raw.py` (existing tests must still pass)

- [ ] **Step 1: Replace the body of `parse_raw_dump_from_text`**

In `src/lammpsparser/output_raw.py`, replace the entire body of `parse_raw_dump_from_text` (everything after the docstring) with:

```python
    dump = DumpData()
    for frame in _iter_raw_frames(file_name):
        dump.steps.append(frame["steps"])
        dump.natoms.append(frame["natoms"])
        dump.cells.append(frame["cells"])
        dump.indices.append(frame["indices"])
        if len(frame["forces"]):
            dump.forces.append(frame["forces"])
        if len(frame["mean_forces"]):
            dump.mean_forces.append(frame["mean_forces"])
        if len(frame["velocities"]):
            dump.velocities.append(frame["velocities"])
        if len(frame["mean_velocities"]):
            dump.mean_velocities.append(frame["mean_velocities"])
        if len(frame["unwrapped_positions"]):
            dump.unwrapped_positions.append(frame["unwrapped_positions"])
            dump.positions.append(frame["positions"])
        if len(frame["mean_unwrapped_positions"]):
            dump.mean_unwrapped_positions.append(frame["mean_unwrapped_positions"])
        for k, v in frame["computes"].items():
            if k not in dump.computes:
                dump.computes[k] = []
            dump.computes[k].append(v)
    return asdict(dump)
```

- [ ] **Step 2: Run the full output_raw test suite**

```bash
python -m pytest tests/test_output_raw.py -v
```

Expected: all tests PASS (both the new `TestIterRawFrames` tests and the existing tests).

- [ ] **Step 3: Commit**

```bash
git add src/lammpsparser/output_raw.py
git commit -m "refactor: parse_raw_dump_from_text uses _iter_raw_frames internally"
```

---

## Task 3: Add `LammpsFrame` dataclass and `iter_lammps_frames` to `output.py`

**Files:**
- Modify: `src/lammpsparser/output.py`
- Test: `tests/test_output.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_output.py`:

```python
from lammpsparser.output import iter_lammps_frames, LammpsFrame
from lammpsparser import parse_lammps_output_files

class TestIterLammpsFrames(unittest.TestCase):
    def setUp(self):
        self.static_folder = os.path.abspath(os.path.join(__file__, "..", "static"))
        self.structure = bulk("Al", cubic=True)
        self.potential_elements = ["Al"]
        self.units = "metal"

    def _full_job_dir(self):
        return os.path.join(self.static_folder, "full_job")

    def test_yields_lammps_frame_instances(self):
        frames = list(iter_lammps_frames(
            working_directory=self._full_job_dir(),
            structure=self.structure,
            potential_elements=self.potential_elements,
            units=self.units,
        ))
        self.assertGreater(len(frames), 0)
        self.assertIsInstance(frames[0], LammpsFrame)

    def test_frame_has_required_fields(self):
        frame = next(iter(iter_lammps_frames(
            working_directory=self._full_job_dir(),
            structure=self.structure,
            potential_elements=self.potential_elements,
            units=self.units,
        )))
        self.assertIsInstance(frame.step, int)
        self.assertEqual(frame.cell.shape, (3, 3))
        self.assertEqual(frame.positions.ndim, 2)
        self.assertEqual(frame.positions.shape[1], 3)
        self.assertEqual(frame.forces.ndim, 2)
        self.assertEqual(frame.indices.ndim, 1)

    def test_equivalence_with_batch_parser(self):
        """Streaming all frames must produce the same positions and forces as parse_lammps_output_files."""
        frames = list(iter_lammps_frames(
            working_directory=self._full_job_dir(),
            structure=self.structure,
            potential_elements=self.potential_elements,
            units=self.units,
        ))
        batch = parse_lammps_output_files(
            working_directory=self._full_job_dir(),
            structure=self.structure,
            potential_elements=self.potential_elements,
            units=self.units,
        )
        streamed_positions = np.stack([f.positions for f in frames])
        np.testing.assert_allclose(streamed_positions, batch["generic"]["positions"], rtol=1e-10)
        streamed_forces = np.stack([f.forces for f in frames])
        np.testing.assert_allclose(streamed_forces, batch["generic"]["forces"], rtol=1e-10)

    def test_start_stop_slicing(self):
        all_frames = list(iter_lammps_frames(
            working_directory=self._full_job_dir(),
            structure=self.structure,
            potential_elements=self.potential_elements,
            units=self.units,
        ))
        sliced = list(iter_lammps_frames(
            working_directory=self._full_job_dir(),
            structure=self.structure,
            potential_elements=self.potential_elements,
            units=self.units,
            start=1,
            stop=3,
        ))
        self.assertEqual(len(sliced), 2)
        np.testing.assert_array_equal(sliced[0].positions, all_frames[1].positions)
        np.testing.assert_array_equal(sliced[1].positions, all_frames[2].positions)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_output.py::TestIterLammpsFrames -v
```

Expected: all 4 tests FAIL with `ImportError` — `iter_lammps_frames` and `LammpsFrame` don't exist yet.

- [ ] **Step 3: Check what `full_job` static fixture contains**

```bash
ls tests/static/full_job/
```

If `dump.out` exists there, you're good. If the directory only has `dump.h5`, use a directory that has a text dump (e.g. `mean_values`) and update the `_full_job_dir` method in the test to point to it. Streaming only supports text dumps, not H5MD.

- [ ] **Step 4: Add `LammpsFrame` dataclass and `iter_lammps_frames` to `output.py`**

At the top of `src/lammpsparser/output.py`, add `Iterator` to the typing import:

```python
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
```

Also add the `dataclass` import:

```python
from dataclasses import dataclass
```

Then add the following just below the existing imports, before `remap_indices_ase`:

```python
@dataclass
class LammpsFrame:
    """A single parsed, unit-converted frame from a LAMMPS dump file."""
    step: int
    cell: np.ndarray
    positions: np.ndarray
    forces: np.ndarray
    indices: np.ndarray
    velocities: Optional[np.ndarray] = None
    computes: Optional[Dict[str, np.ndarray]] = None
```

Then add `iter_lammps_frames` at the bottom of `output.py`, after `_check_ortho_prism`:

```python
def iter_lammps_frames(
    working_directory: str,
    structure: Atoms,
    potential_elements: Union[np.ndarray, List],
    units: str,
    dump_out_file_name: str = "dump.out",
    prism: Optional[UnfoldingPrism] = None,
    remap_indices_funct: Callable[..., np.ndarray] = remap_indices_ase,
    start: int = 0,
    stop: Optional[int] = None,
    step: int = 1,
) -> Iterator[LammpsFrame]:
    """
    Yield one parsed, unit-converted frame at a time from a LAMMPS text dump file.

    Memory usage is proportional to a single frame, not the entire trajectory.
    Coordinates are rotated to the ASE frame, indices are remapped, and all
    quantities are converted to pyiron/ASE units (Å, eV, ps).

    Args:
        working_directory: Directory containing the LAMMPS output files.
        structure: Input ASE Atoms used as template for index remapping.
        potential_elements: Ordered list of element symbols matching the potential.
        units: LAMMPS unit system (e.g. ``"metal"``, ``"real"``).
        dump_out_file_name: Name of the text dump file (default ``"dump.out"``).
        prism: Pre-built UnfoldingPrism. Built from ``structure.cell`` if None.
        remap_indices_funct: Index remapping function (default: remap_indices_ase).
        start: First frame index to yield (0-based, default 0).
        stop: Stop before this frame index. None means read to end.
        step: Yield every ``step``-th frame (default 1).

    Yields:
        LammpsFrame: One frame with fields step, cell, positions, forces,
        indices, velocities (or None), computes (or None).

    Raises:
        FileNotFoundError: If the dump file does not exist.
        ValueError: If a frame is malformed or the file is truncated.
    """
    from lammpsparser.output_raw import _iter_raw_frames

    file_path = os.path.join(working_directory, dump_out_file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dump file not found: {file_path}")

    if prism is None:
        prism = UnfoldingPrism(structure.cell)

    convert_units = UnitConverter(units).convert_array_to_pyiron_units
    rotation_lammps2orig = prism.R.T

    for raw in _iter_raw_frames(file_path, start=start, stop=stop, step=step):
        cell = np.array(prism.unfold_cell(cell=raw["cells"]))

        positions_frac = raw["positions"]
        positions = np.matmul(
            np.matmul(positions_frac, np.array(raw["cells"])), rotation_lammps2orig
        )
        positions = convert_units(positions, label="positions")

        forces = np.matmul(raw["forces"], rotation_lammps2orig)
        forces = convert_units(forces, label="forces")

        indices = remap_indices_funct(
            lammps_indices=raw["indices"],
            potential_elements=potential_elements,
            structure=structure,
        )

        velocities = None
        if len(raw["velocities"]):
            velocities = convert_units(
                np.matmul(raw["velocities"], rotation_lammps2orig),
                label="velocities",
            )

        computes = None
        if raw["computes"]:
            computes = {
                k: convert_units(np.array(v), label=k)
                for k, v in raw["computes"].items()
            }

        yield LammpsFrame(
            step=int(raw["steps"]),
            cell=cell,
            positions=positions,
            forces=forces,
            indices=indices,
            velocities=velocities,
            computes=computes,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_output.py::TestIterLammpsFrames -v
```

Expected: all 4 tests PASS. If the `full_job` fixture only has H5MD, update `_full_job_dir` to point to a directory with a `dump.out` (e.g. `mean_values` or `mean_values_non_ortho`).

- [ ] **Step 6: Run the full test suite to check for regressions**

```bash
python -m pytest tests/ -v
```

Expected: all existing tests still PASS.

- [ ] **Step 7: Commit**

```bash
git add src/lammpsparser/output.py tests/test_output.py
git commit -m "feat: add LammpsFrame dataclass and iter_lammps_frames generator"
```

---

## Task 4: Export from `__init__.py`

**Files:**
- Modify: `src/lammpsparser/__init__.py`
- Test: none needed (import test is implicit in Task 3's tests)

- [ ] **Step 1: Add exports**

In `src/lammpsparser/__init__.py`, add the following import after the existing `output` import:

```python
from lammpsparser.output import iter_lammps_frames, LammpsFrame
```

And add both names to `__all__`:

```python
__all__ = [
    "calc_md",
    "calc_minimize",
    "calc_static",
    "get_potential_by_name",
    "get_potential_dataframe",
    "iter_lammps_frames",
    "LammpsFrame",
    "lammps_file_initialization",
    "lammps_file_interface_function",
    "parse_lammps_output_files",
    "validate_potential_dataframe",
    "write_lammps_structure",
]
```

- [ ] **Step 2: Verify the public import works**

```bash
python -c "from lammpsparser import iter_lammps_frames, LammpsFrame; print('OK')"
```

Expected output: `OK`

- [ ] **Step 3: Run full test suite one final time**

```bash
python -m pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/lammpsparser/__init__.py
git commit -m "feat: export iter_lammps_frames and LammpsFrame from package root"
```

---

## Task 5: Add streaming example to the notebook

**Files:**
- Modify: `notebooks/example.ipynb`

- [ ] **Step 1: Add a new cell at the end of `notebooks/example.ipynb` showing streaming usage**

Open the notebook and append a new markdown cell:

```markdown
## Streaming large trajectories

For large trajectories that don't fit in memory, use `iter_lammps_frames` to process one frame at a time:
```

Followed by a code cell:

```python
from lammpsparser import iter_lammps_frames

# Process frames one at a time — memory usage stays constant
for frame in iter_lammps_frames(
    working_directory=working_directory,
    structure=structure,
    potential_elements=potential_elements,
    units="metal",
    start=0,   # skip warmup frames by setting start > 0
):
    # frame.step, frame.positions, frame.forces, frame.indices are all available
    print(f"Step {frame.step}: mean force magnitude = {(frame.forces**2).sum(axis=1).mean():.4f} eV/Å")
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/example.ipynb
git commit -m "docs: add streaming iter_lammps_frames example to notebook"
```

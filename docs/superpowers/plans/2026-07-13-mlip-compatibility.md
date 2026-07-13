# MLIP Compatibility Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-implement the useful parts of pyiron_atomistics' legacy `LammpsMlip` job class (MTP/MLIP potential support: `mlip.ini` generation, convergence checking, `.cfg` parsing) as standalone, stateless functions in `lammpsparser`, with zero pyiron dependencies.

**Architecture:** A single new module, `src/lammpsparser/compatibility/mlip.py`, exposing five public names (`write_mlip_input_file`, `check_mlip_convergence`, `MlipConfiguration`, `load_mlip_cfgs`, `get_mlip_selected_structures`). It's standalone and composable — the caller invokes it before/after `lammps_file_interface_function`, mirroring the existing relationship between `file.py` and `calculate.py`/`constraints.py`. A from-scratch, dependency-free parser reads the plain-text MTP `.cfg` format.

**Tech Stack:** Python 3.9+, `numpy`, `ase.atoms.Atoms`, stdlib `os`/`dataclasses`. Tests use `unittest` (matches this repo's existing test suite and CI, which runs `python -m unittest discover tests`).

## Global Constraints

- No new dependencies — only `os`, `dataclasses`, `typing`, `numpy`, `ase.atoms.Atoms` (already project dependencies).
- No `pyiron_base`/`pyiron_potentialfit`/`StructureStorage` — this module must work standalone.
- No `SinglePointCalculator` — per-structure MLIP predictions go in `atoms.arrays`/`atoms.info` directly (per approved design).
- No wiring into `lammps_file_interface_function` — this stays a standalone, composable module.
- Follow the existing code style in `src/lammpsparser/compatibility/`: Google-style docstrings with `Args:`/`Returns:`/`Raises:`, type hints on all public functions.
- Test runner is `unittest` (e.g. `python -m unittest tests/test_compatibility_mlip.py -v`), not `pytest` — this repo does not depend on pytest.

---

### Task 1: `write_mlip_input_file` and `check_mlip_convergence`

**Files:**
- Create: `src/lammpsparser/compatibility/mlip.py`
- Create: `tests/test_compatibility_mlip.py`

**Interfaces:**
- Consumes: nothing from other tasks (first task).
- Produces: `write_mlip_input_file(working_directory: str, mtp_filename: str, active_learning: bool = False, threshold: float = 2.0, threshold_break: float = 5.0, save_selected: str = "selected.cfg", load_state: str = "state.mvs", log: str = "selection.log", file_name: str = "mlip.ini") -> str` and `check_mlip_convergence(working_directory: str, error_file_name: str = "error.out") -> bool`, both importable from `lammpsparser.compatibility.mlip`. Later tasks add more functions to this same file/test file — do not overwrite these two.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_compatibility_mlip.py`:

```python
import os
import shutil
import unittest

from lammpsparser.compatibility.mlip import (
    check_mlip_convergence,
    write_mlip_input_file,
)


class TestWriteMlipInputFile(unittest.TestCase):
    def setUp(self):
        self.working_directory = os.path.abspath("mlip_working_directory")

    def tearDown(self):
        if os.path.exists(self.working_directory):
            shutil.rmtree(self.working_directory)

    def test_creates_working_directory(self):
        self.assertFalse(os.path.exists(self.working_directory))
        write_mlip_input_file(
            working_directory=self.working_directory,
            mtp_filename="/abs/path/to/pot.mtp",
        )
        self.assertTrue(os.path.exists(self.working_directory))

    def test_default_content(self):
        file_path = write_mlip_input_file(
            working_directory=self.working_directory,
            mtp_filename="/abs/path/to/pot.mtp",
        )
        self.assertEqual(file_path, os.path.join(self.working_directory, "mlip.ini"))
        with open(file_path) as f:
            content = f.read()
        self.assertEqual(
            content,
            "mtp-filename /abs/path/to/pot.mtp\nselect FALSE\n",
        )

    def test_active_learning_content(self):
        file_path = write_mlip_input_file(
            working_directory=self.working_directory,
            mtp_filename="/abs/path/to/pot.mtp",
            active_learning=True,
            threshold=2.5,
            threshold_break=6.0,
        )
        with open(file_path) as f:
            content = f.read()
        self.assertEqual(
            content,
            "mtp-filename /abs/path/to/pot.mtp\n"
            "calculate-efs TRUE\n"
            "select TRUE\n"
            "select:threshold 2.5\n"
            "select:threshold-break 6.0\n"
            "select:save-selected selected.cfg\n"
            "select:load-state state.mvs\n"
            "select:log selection.log\n"
            "write-cfgs:skip 0\n",
        )

    def test_custom_file_name(self):
        file_path = write_mlip_input_file(
            working_directory=self.working_directory,
            mtp_filename="/abs/path/to/pot.mtp",
            file_name="custom_mlip.ini",
        )
        self.assertEqual(
            file_path, os.path.join(self.working_directory, "custom_mlip.ini")
        )
        self.assertTrue(os.path.exists(file_path))


class TestCheckMlipConvergence(unittest.TestCase):
    def setUp(self):
        self.working_directory = os.path.abspath(
            "mlip_convergence_working_directory"
        )
        os.makedirs(self.working_directory, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.working_directory):
            shutil.rmtree(self.working_directory)

    def test_no_error_file_is_converged(self):
        self.assertTrue(
            check_mlip_convergence(working_directory=self.working_directory)
        )

    def test_error_file_without_breaking_line_is_converged(self):
        with open(os.path.join(self.working_directory, "error.out"), "w") as f:
            f.write("Some unrelated warning\n")
        self.assertTrue(
            check_mlip_convergence(working_directory=self.working_directory)
        )

    def test_error_file_with_breaking_line_is_not_converged(self):
        with open(os.path.join(self.working_directory, "error.out"), "w") as f:
            f.write("MLIP: Breaking threshold exceeded, stopping\n")
        self.assertFalse(
            check_mlip_convergence(working_directory=self.working_directory)
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests/test_compatibility_mlip.py -v`
Expected: `ModuleNotFoundError: No module named 'lammpsparser.compatibility.mlip'` (the module doesn't exist yet).

- [ ] **Step 3: Create `src/lammpsparser/compatibility/mlip.py` with the two functions**

```python
import os
from typing import Optional


def write_mlip_input_file(
    working_directory: str,
    mtp_filename: str,
    active_learning: bool = False,
    threshold: float = 2.0,
    threshold_break: float = 5.0,
    save_selected: str = "selected.cfg",
    load_state: str = "state.mvs",
    log: str = "selection.log",
    file_name: str = "mlip.ini",
) -> str:
    """
    Write the ``mlip.ini`` control file consumed by LAMMPS' ``pair_style mlip``.

    Args:
        working_directory (str): Directory the file is written into. Created
            if it does not already exist.
        mtp_filename (str): Absolute path to the underlying ``.mtp`` potential
            file. Resolve this the same way other potential file paths are
            resolved elsewhere in lammpsparser (e.g. via
            ``potential["Filename"][0]`` after
            :func:`~lammpsparser.potential.update_potential_paths`).
        active_learning (bool): If ``True``, enable MLIP's active-learning
            selection mode: LAMMPS flags (and optionally stops on) structures
            where the potential is extrapolating.
        threshold (float): Extrapolation grade above which a structure is
            selected for the training set. Only used when
            ``active_learning=True``.
        threshold_break (float): Extrapolation grade above which the LAMMPS
            run is stopped. Only used when ``active_learning=True``.
        save_selected (str): Filename (relative to ``working_directory``)
            that MLIP writes selected configurations to. Only used when
            ``active_learning=True``.
        load_state (str): Filename (relative to ``working_directory``) for
            MLIP's selection state file. Only used when
            ``active_learning=True``.
        log (str): Filename (relative to ``working_directory``) for MLIP's
            selection log. Only used when ``active_learning=True``.
        file_name (str): Name of the file to write (default: ``"mlip.ini"``).

    Returns:
        str: Absolute path to the written file.
    """
    lines = ["mtp-filename " + mtp_filename]
    if active_learning:
        lines += [
            "calculate-efs TRUE",
            "select TRUE",
            "select:threshold " + str(threshold),
            "select:threshold-break " + str(threshold_break),
            "select:save-selected " + save_selected,
            "select:load-state " + load_state,
            "select:log " + log,
            "write-cfgs:skip 0",
        ]
    else:
        lines.append("select FALSE")

    os.makedirs(working_directory, exist_ok=True)
    file_path = os.path.join(working_directory, file_name)
    with open(file_path, "w") as f:
        f.writelines([line + "\n" for line in lines])
    return file_path


def check_mlip_convergence(
    working_directory: str, error_file_name: str = "error.out"
) -> bool:
    """
    Check whether an MLIP active-learning run converged (did not break early).

    Args:
        working_directory (str): Directory containing the LAMMPS run's
            ``error.out`` file, if any.
        error_file_name (str): Filename (relative to ``working_directory``)
            to check (default: ``"error.out"``).

    Returns:
        bool: ``True`` if ``error_file_name`` does not exist, or exists but
        contains no line starting with ``"MLIP: Breaking threshold
        exceeded"``. ``False`` otherwise.
    """
    error_file_path = os.path.join(working_directory, error_file_name)
    if not os.path.exists(error_file_path):
        return True
    with open(error_file_path) as f:
        for line in f:
            if line.startswith("MLIP: Breaking threshold exceeded"):
                return False
    return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests/test_compatibility_mlip.py -v`
Expected: All 7 tests `OK`.

- [ ] **Step 5: Commit**

```bash
git add src/lammpsparser/compatibility/mlip.py tests/test_compatibility_mlip.py
git commit -m "$(cat <<'EOF'
Add write_mlip_input_file and check_mlip_convergence

First piece of the standalone MLIP compatibility module: generating
the mlip.ini control file for pair_style mlip (including active
learning mode) and checking error.out for the MLIP breaking-threshold
condition. Ports LammpsMlip.write_input/enable_active_learning and
LammpsMlip.convergence_check from pyiron_atomistics without any
pyiron dependency.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `MlipConfiguration` and `load_mlip_cfgs`

**Files:**
- Modify: `src/lammpsparser/compatibility/mlip.py`
- Modify: `tests/test_compatibility_mlip.py`
- Create: `tests/static/mlip/selected.cfg`

**Interfaces:**
- Consumes: nothing from Task 1's functions directly (independent addition to the same file).
- Produces: `MlipConfiguration` dataclass with fields `cell: np.ndarray`, `positions: np.ndarray`, `types: np.ndarray`, `forces: Optional[np.ndarray] = None`, `energy: Optional[float] = None`, `stress: Optional[np.ndarray] = None`, `grade: Optional[float] = None`; and `load_mlip_cfgs(file_name: str) -> List[MlipConfiguration]`. Task 3's `get_mlip_selected_structures` calls `load_mlip_cfgs` and reads these exact field names.

- [ ] **Step 1: Create the fixture file**

Create `tests/static/mlip/selected.cfg` with exactly this content (two configurations: one with full data, one with only the required `Size`/`Supercell`/`AtomData` sections):

```
BEGIN_CFG
 Size
    2
 Supercell
         5.680600067138671875         0.000000000000000000         0.000000000000000000
         0.000000000000000000         5.680600067138671875         0.000000000000000000
         0.000000000000000000         0.000000000000000000         5.680600067138671875
 AtomData:  id type       cartes_x      cartes_y      cartes_z           fx          fy          fz
             1    0      0.000000000000000      0.000000000000000      0.000000000000000     -0.010000000000000     -0.020000000000000     -0.030000000000000
             2    1      2.840300033569336      2.840300033569336      2.840300033569336      0.010000000000000      0.020000000000000      0.030000000000000
 Energy
   -14.4939880371093750
 PlusStress:  xx          yy          zz          yz          xz          xy
       -0.239923946063382      -0.239923946063382      -0.239923946063382       0.000000000000000       0.000000000000000       0.000000000000000
 Feature   EFS_by       MTP
 Feature   MV_grade       3.19183351
END_CFG
BEGIN_CFG
 Size
    1
 Supercell
         4.000000000000000000         0.000000000000000000         0.000000000000000000
         0.000000000000000000         4.000000000000000000         0.000000000000000000
         0.000000000000000000         0.000000000000000000         4.000000000000000000
 AtomData:  id type       cartes_x      cartes_y      cartes_z
             1    0      1.000000000000000      1.000000000000000      1.000000000000000
END_CFG
```

- [ ] **Step 2: Add the failing tests**

In `tests/test_compatibility_mlip.py`, replace the existing top-of-file import block (the `import os` / `import shutil` / `import unittest` / `from lammpsparser.compatibility.mlip import (...)` lines Task 1 wrote) with:

```python
import os
import shutil
import unittest

import numpy as np

from lammpsparser.compatibility.mlip import (
    MlipConfiguration,
    check_mlip_convergence,
    load_mlip_cfgs,
    write_mlip_input_file,
)

STATIC_MLIP_DIR = os.path.join(os.path.dirname(__file__), "static", "mlip")
```

Then add this new test class after `TestCheckMlipConvergence` and before the `if __name__ == "__main__":` block at the end of the file:

```python
class TestLoadMlipCfgs(unittest.TestCase):
    def test_parses_two_configurations(self):
        configurations = load_mlip_cfgs(os.path.join(STATIC_MLIP_DIR, "selected.cfg"))
        self.assertEqual(len(configurations), 2)

    def test_first_configuration_has_full_data(self):
        cfg = load_mlip_cfgs(os.path.join(STATIC_MLIP_DIR, "selected.cfg"))[0]
        self.assertIsInstance(cfg, MlipConfiguration)
        np.testing.assert_allclose(cfg.cell, np.eye(3) * 5.680600067138671875)
        np.testing.assert_allclose(
            cfg.positions,
            [
                [0.0, 0.0, 0.0],
                [2.840300033569336, 2.840300033569336, 2.840300033569336],
            ],
        )
        np.testing.assert_array_equal(cfg.types, [0, 1])
        np.testing.assert_allclose(
            cfg.forces, [[-0.01, -0.02, -0.03], [0.01, 0.02, 0.03]]
        )
        self.assertAlmostEqual(cfg.energy, -14.493988037109375)
        np.testing.assert_allclose(
            cfg.stress,
            [
                -0.239923946063382,
                -0.239923946063382,
                -0.239923946063382,
                0.0,
                0.0,
                0.0,
            ],
        )
        self.assertAlmostEqual(cfg.grade, 3.19183351)

    def test_second_configuration_has_only_required_data(self):
        cfg = load_mlip_cfgs(os.path.join(STATIC_MLIP_DIR, "selected.cfg"))[1]
        np.testing.assert_allclose(cfg.cell, np.eye(3) * 4.0)
        np.testing.assert_allclose(cfg.positions, [[1.0, 1.0, 1.0]])
        np.testing.assert_array_equal(cfg.types, [0])
        self.assertIsNone(cfg.forces)
        self.assertIsNone(cfg.energy)
        self.assertIsNone(cfg.stress)
        self.assertIsNone(cfg.grade)
```

Replace the existing `import unittest` block's imports at the top of the file with the merged import list above (keep `os`, `shutil`, `unittest` imports already present from Task 1).

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m unittest tests/test_compatibility_mlip.py -v`
Expected: `ImportError: cannot import name 'MlipConfiguration' from 'lammpsparser.compatibility.mlip'`.

- [ ] **Step 4: Add `MlipConfiguration` and `load_mlip_cfgs` to `src/lammpsparser/compatibility/mlip.py`**

Add these imports at the top of the file (extending Task 1's `import os`):

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
```

Add this module-level constant and the dataclass and parser functions (append to the end of the file):

```python
MLIP_STRESS_COMPONENTS: Tuple[str, ...] = ("xx", "yy", "zz", "yz", "xz", "xy")


@dataclass
class MlipConfiguration:
    """
    One parsed configuration from an MLIP ``.cfg`` file.

    Attributes:
        cell (numpy.ndarray): Shape ``(3, 3)`` simulation cell, from the
            ``Supercell`` section.
        positions (numpy.ndarray): Shape ``(N, 3)`` Cartesian atomic
            positions.
        types (numpy.ndarray): Shape ``(N,)`` integer species-type indices,
            0-based, as written by MLIP.
        forces (numpy.ndarray, optional): Shape ``(N, 3)`` if the
            ``AtomData:`` header includes ``fx fy fz``, else ``None``.
        energy (float, optional): From the ``Energy`` section, if present.
        stress (numpy.ndarray, optional): Shape ``(6,)`` in ASE Voigt order
            ``(xx, yy, zz, yz, xz, xy)``, from ``PlusStress:``, if present.
        grade (float, optional): Extrapolation grade from a
            ``Feature   MV_grade   <value>`` line, if present.
    """

    cell: np.ndarray
    positions: np.ndarray
    types: np.ndarray
    forces: Optional[np.ndarray] = None
    energy: Optional[float] = None
    stress: Optional[np.ndarray] = None
    grade: Optional[float] = None


def load_mlip_cfgs(file_name: str) -> List[MlipConfiguration]:
    """
    Parse an MLIP ``.cfg`` file into a list of configurations.

    Args:
        file_name (str): Path to a ``.cfg`` file containing one or more
            ``BEGIN_CFG``/``END_CFG`` blocks (e.g. MLIP's
            ``select:save-selected`` output).

    Returns:
        list[MlipConfiguration]: One entry per ``BEGIN_CFG``/``END_CFG``
        block, in file order.

    Raises:
        ValueError: If an ``END_CFG`` is found without a matching
            ``BEGIN_CFG``, or a block is missing its required ``Supercell``
            or ``AtomData:`` section.
    """
    with open(file_name) as f:
        raw_lines = f.readlines()

    configurations: List[MlipConfiguration] = []
    block: Optional[List[str]] = None
    for line in raw_lines:
        stripped = line.strip()
        if stripped == "BEGIN_CFG":
            block = []
        elif stripped == "END_CFG":
            if block is None:
                raise ValueError("Found 'END_CFG' without a matching 'BEGIN_CFG'.")
            configurations.append(_parse_cfg_block(block))
            block = None
        elif block is not None:
            block.append(line)
    return configurations


def _parse_cfg_block(lines: List[str]) -> MlipConfiguration:
    """
    Parse the lines between one ``BEGIN_CFG``/``END_CFG`` pair.

    Args:
        lines (list[str]): Raw lines strictly between ``BEGIN_CFG`` and
            ``END_CFG`` (exclusive of both markers).

    Returns:
        MlipConfiguration: Parsed configuration.

    Raises:
        ValueError: If the block is missing its required ``Supercell`` or
            ``AtomData:`` section, or ``AtomData:`` appears before ``Size``.
    """
    cell: Optional[np.ndarray] = None
    size: Optional[int] = None
    atom_columns: Optional[List[str]] = None
    atom_rows: List[List[float]] = []
    energy: Optional[float] = None
    stress: Optional[np.ndarray] = None
    grade: Optional[float] = None

    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()
        if stripped.startswith("Size"):
            i += 1
            size = int(lines[i].split()[0])
        elif stripped.startswith("Supercell"):
            cell = np.array(
                [[float(v) for v in lines[i + 1 + row].split()] for row in range(3)]
            )
            i += 3
        elif stripped.startswith("AtomData"):
            if size is None:
                raise ValueError("MLIP cfg block has 'AtomData' before 'Size'.")
            atom_columns = stripped.split(":", 1)[1].split()
            for _ in range(size):
                i += 1
                atom_rows.append([float(v) for v in lines[i].split()])
        elif stripped.startswith("Energy"):
            i += 1
            energy = float(lines[i].split()[0])
        elif stripped.startswith("PlusStress"):
            header = stripped.split(":", 1)[1].split()
            i += 1
            values = [float(v) for v in lines[i].split()]
            stress_by_component = dict(zip(header, values))
            stress = np.array(
                [stress_by_component[component] for component in MLIP_STRESS_COMPONENTS]
            )
        elif stripped.startswith("Feature") and "MV_grade" in stripped:
            grade = float(stripped.split()[-1])
        i += 1

    if cell is None or atom_columns is None:
        raise ValueError(
            "MLIP cfg block is missing a required 'Supercell' or 'AtomData' section."
        )

    atom_array = np.array(atom_rows)
    column_index = {name: idx for idx, name in enumerate(atom_columns)}
    types = atom_array[:, column_index["type"]].astype(int)
    if "cartes_x" in column_index:
        positions = atom_array[
            :,
            [column_index["cartes_x"], column_index["cartes_y"], column_index["cartes_z"]],
        ]
    else:
        fractional = atom_array[
            :,
            [column_index["direct_x"], column_index["direct_y"], column_index["direct_z"]],
        ]
        positions = fractional @ cell

    forces = None
    if "fx" in column_index:
        forces = atom_array[:, [column_index["fx"], column_index["fy"], column_index["fz"]]]

    return MlipConfiguration(
        cell=cell,
        positions=positions,
        types=types,
        forces=forces,
        energy=energy,
        stress=stress,
        grade=grade,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m unittest tests/test_compatibility_mlip.py -v`
Expected: All tests `OK` (10 tests total: 7 from Task 1 + 3 new).

- [ ] **Step 6: Commit**

```bash
git add src/lammpsparser/compatibility/mlip.py tests/test_compatibility_mlip.py tests/static/mlip/selected.cfg
git commit -m "$(cat <<'EOF'
Add MlipConfiguration and load_mlip_cfgs parser

Implements a from-scratch parser for the plain-text MTP .cfg format
(BEGIN_CFG/Size/Supercell/AtomData/Energy/PlusStress/Feature/END_CFG),
replacing the pyiron_potentialfit.mlip.cfgs.loadcfgs dependency the
legacy LammpsMlip job relied on. Stress is parsed directly into ASE's
Voigt order (xx, yy, zz, yz, xz, xy) by column name, so the parser is
robust to header reordering.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `get_mlip_selected_structures`

**Files:**
- Modify: `src/lammpsparser/compatibility/mlip.py`
- Modify: `tests/test_compatibility_mlip.py`

**Interfaces:**
- Consumes: `load_mlip_cfgs(file_name: str) -> List[MlipConfiguration]` and the `MlipConfiguration` fields from Task 2 (`cell`, `positions`, `types`, `forces`, `energy`, `stress`, `grade`).
- Produces: `get_mlip_selected_structures(file_name: str, species: List[str]) -> List[Atoms]`.

- [ ] **Step 1: Add the failing test**

In `tests/test_compatibility_mlip.py`, replace the `from lammpsparser.compatibility.mlip import (...)` block (written in Task 2) with:

```python
from lammpsparser.compatibility.mlip import (
    MlipConfiguration,
    check_mlip_convergence,
    get_mlip_selected_structures,
    load_mlip_cfgs,
    write_mlip_input_file,
)
```

Then add this new test class after `TestLoadMlipCfgs` and before the `if __name__ == "__main__":` block at the end of the file:

```python
class TestGetMlipSelectedStructures(unittest.TestCase):
    def test_builds_atoms_with_species_and_metadata(self):
        structures = get_mlip_selected_structures(
            file_name=os.path.join(STATIC_MLIP_DIR, "selected.cfg"),
            species=["Al", "Ni"],
        )
        self.assertEqual(len(structures), 2)

        first = structures[0]
        self.assertEqual(first.get_chemical_symbols(), ["Al", "Ni"])
        np.testing.assert_allclose(
            first.arrays["forces"], [[-0.01, -0.02, -0.03], [0.01, 0.02, 0.03]]
        )
        self.assertAlmostEqual(first.info["energy"], -14.493988037109375)
        np.testing.assert_allclose(
            first.info["stress"],
            [
                -0.239923946063382,
                -0.239923946063382,
                -0.239923946063382,
                0.0,
                0.0,
                0.0,
            ],
        )
        self.assertAlmostEqual(first.info["mv_grade"], 3.19183351)

        second = structures[1]
        self.assertEqual(second.get_chemical_symbols(), ["Al"])
        self.assertNotIn("forces", second.arrays)
        self.assertNotIn("energy", second.info)
        self.assertNotIn("stress", second.info)
        self.assertNotIn("mv_grade", second.info)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests/test_compatibility_mlip.py -v`
Expected: `ImportError: cannot import name 'get_mlip_selected_structures' from 'lammpsparser.compatibility.mlip'`.

- [ ] **Step 3: Add `get_mlip_selected_structures` to `src/lammpsparser/compatibility/mlip.py`**

Add this import at the top of the file (alongside the existing `numpy as np` import):

```python
from ase.atoms import Atoms
```

Append this function to the end of the file:

```python
def get_mlip_selected_structures(file_name: str, species: List[str]) -> List[Atoms]:
    """
    Build structures from an MLIP ``.cfg`` file (typically ``selected.cfg``).

    Args:
        file_name (str): Path to a ``.cfg`` file, as accepted by
            :func:`load_mlip_cfgs`.
        species (list[str]): Chemical symbols in the same order as the
            interatomic potential declares them (e.g. the potential
            dataframe's ``"Species"`` column), used to map each
            configuration's integer type indices to element symbols.

    Returns:
        list[ase.atoms.Atoms]: One ``Atoms`` object per configuration in
        ``file_name``, with ``pbc=True``. When present in the source
        configuration: per-atom forces are stored in ``atoms.arrays["forces"]``,
        and the total energy, stress (ASE Voigt order), and MLIP
        extrapolation grade are stored in ``atoms.info["energy"]``,
        ``atoms.info["stress"]``, and ``atoms.info["mv_grade"]``
        respectively.
    """
    structures = []
    for cfg in load_mlip_cfgs(file_name):
        symbols = np.asarray(species)[cfg.types]
        atoms = Atoms(
            symbols=symbols,
            positions=cfg.positions,
            cell=cfg.cell,
            pbc=True,
        )
        if cfg.forces is not None:
            atoms.arrays["forces"] = cfg.forces
        if cfg.energy is not None:
            atoms.info["energy"] = cfg.energy
        if cfg.stress is not None:
            atoms.info["stress"] = cfg.stress
        if cfg.grade is not None:
            atoms.info["mv_grade"] = cfg.grade
        structures.append(atoms)
    return structures
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests/test_compatibility_mlip.py -v`
Expected: All tests `OK` (11 tests total).

- [ ] **Step 5: Commit**

```bash
git add src/lammpsparser/compatibility/mlip.py tests/test_compatibility_mlip.py
git commit -m "$(cat <<'EOF'
Add get_mlip_selected_structures

Builds ase.atoms.Atoms from parsed MLIP .cfg configurations, mapping
integer type indices to chemical symbols via the potential's species
list and attaching forces/energy/stress/mv_grade as plain arrays/info
entries (no SinglePointCalculator). Replaces
LammpsMlip.collect_output()'s .cfg-parsing loop, the
selected_structures property, and the StructureStorage dependency.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Export from `lammpsparser` top level and run full suite

**Files:**
- Modify: `src/lammpsparser/__init__.py`

**Interfaces:**
- Consumes: `write_mlip_input_file`, `check_mlip_convergence`, `load_mlip_cfgs`, `get_mlip_selected_structures` from `lammpsparser.compatibility.mlip` (Tasks 1-3). `MlipConfiguration` is intentionally NOT re-exported at top level (matches `CalcMDInput`/`CalcMinimizeInput` in `data.py`, which also stay module-scoped).
- Produces: `lammpsparser.write_mlip_input_file`, `lammpsparser.check_mlip_convergence`, `lammpsparser.load_mlip_cfgs`, `lammpsparser.get_mlip_selected_structures`.

- [ ] **Step 1: Write the failing test**

Add a new test file `tests/test_top_level_exports.py`:

```python
import unittest


class TestTopLevelMlipExports(unittest.TestCase):
    def test_mlip_functions_importable_from_top_level(self):
        from lammpsparser import (
            check_mlip_convergence,
            get_mlip_selected_structures,
            load_mlip_cfgs,
            write_mlip_input_file,
        )

        self.assertTrue(callable(write_mlip_input_file))
        self.assertTrue(callable(check_mlip_convergence))
        self.assertTrue(callable(load_mlip_cfgs))
        self.assertTrue(callable(get_mlip_selected_structures))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_top_level_exports.py -v`
Expected: `ImportError: cannot import name 'check_mlip_convergence' from 'lammpsparser'`.

- [ ] **Step 3: Update `src/lammpsparser/__init__.py`**

Current content (read before editing — do not guess at unshown lines):

```python
import lammpsparser._version
from lammpsparser.compatibility.calculate import calc_md, calc_minimize, calc_static
from lammpsparser.compatibility.file import (
    lammps_file_initialization,
    lammps_file_interface_function,
)
from lammpsparser.output import parse_lammps_output as parse_lammps_output_files
from lammpsparser.potential import (
    get_potential_by_name,
    get_potential_dataframe,
    validate_potential_dataframe,
)
from lammpsparser.structure import write_lammps_datafile as write_lammps_structure

DUMP_COMMANDS = [
    "dump 1 all custom 100 dump.out id type xsu ysu zsu fx fy fz vx vy vz\n",
    'dump_modify 1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"\n',
]

THERMO_COMMANDS = [
    "thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol\n",
    "thermo_modify format float %20.15g\n",
    "thermo 100\n",
]

__version__ = lammpsparser._version.__version__
__all__ = [
    "calc_md",
    "calc_minimize",
    "calc_static",
    "get_potential_by_name",
    "get_potential_dataframe",
    "lammps_file_initialization",
    "lammps_file_interface_function",
    "parse_lammps_output_files",
    "validate_potential_dataframe",
    "write_lammps_structure",
]
```

Add the new import line after the `lammpsparser.compatibility.file` import block:

```python
from lammpsparser.compatibility.mlip import (
    check_mlip_convergence,
    get_mlip_selected_structures,
    load_mlip_cfgs,
    write_mlip_input_file,
)
```

Update `__all__` to (alphabetically ordered, matching the existing style):

```python
__all__ = [
    "calc_md",
    "calc_minimize",
    "calc_static",
    "check_mlip_convergence",
    "get_mlip_selected_structures",
    "get_potential_by_name",
    "get_potential_dataframe",
    "lammps_file_initialization",
    "lammps_file_interface_function",
    "load_mlip_cfgs",
    "parse_lammps_output_files",
    "validate_potential_dataframe",
    "write_lammps_structure",
    "write_mlip_input_file",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_top_level_exports.py -v`
Expected: `OK`.

- [ ] **Step 5: Run the full test suite**

Run: `python -m unittest discover tests`
Expected: All tests `OK` (pre-existing tests plus the 11 new MLIP tests plus the 1 new export test), no failures or errors. Note: `tests/test_compatibility_integration.py` will be skipped unless a real `lammps` Python module is importable in this environment — that's expected and unrelated to this change.

- [ ] **Step 6: Commit**

```bash
git add src/lammpsparser/__init__.py tests/test_top_level_exports.py
git commit -m "$(cat <<'EOF'
Export MLIP compatibility functions from top-level lammpsparser

Makes write_mlip_input_file, check_mlip_convergence, load_mlip_cfgs,
and get_mlip_selected_structures importable directly from
lammpsparser, matching how the other compatibility functions
(calc_md, lammps_file_interface_function, etc.) are already
re-exported.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

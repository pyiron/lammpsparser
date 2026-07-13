# MLIP compatibility module

## Context

`pyiron_atomistics` previously had a `LammpsMlip` job class (subclassing `LammpsInteractive`) that added support for MTP/MLIP interatomic potentials on top of the generic LAMMPS job: writing an `mlip.ini` control file, enabling "active learning" (LAMMPS stops or flags configurations when the potential extrapolates), checking convergence from `error.out`, and parsing MLIP's `selected.cfg` output into structures for retraining.

That implementation depended on `pyiron_base.GenericParameters`, `pyiron_potentialfit.mlip.cfgs.loadcfgs`, `StructureStorage`, and pyiron's job/HDF5 status model — none of which exist in `lammpsparser`, which is a small, stateless, function-based library (`ase`, `numpy`, `pandas`, `scipy` only). This spec re-implements the useful parts of that functionality as plain functions that fit `lammpsparser`'s existing `compatibility` module style.

## Goals

- Generate the `mlip.ini` file consumed by LAMMPS's `pair_style mlip`.
- Support "active learning" mode (extrapolation-grade thresholds, selection state/log files).
- Check post-run convergence from `error.out`.
- Parse MLIP `.cfg` files (in particular `selected.cfg`) into `ase.atoms.Atoms` objects.

## Non-goals

- No job/HDF5/status machinery — this is pure input generation + output parsing.
- No `subprocess` return-code handling for the MLIP "breaking threshold exceeded" exit code (8). That's an orchestration concern; the caller of `lammps_file_interface_function` is responsible for catching `subprocess.CalledProcessError` when active learning is enabled and then consulting `check_mlip_convergence`.
- No dependency on `pyiron_potentialfit` or any other pyiron package — the `.cfg` parser is implemented from scratch against the plain-text MTP cfg format.
- No wiring into `lammps_file_interface_function` itself — this module is standalone and composable, matching the existing relationship between `file.py` and `calculate.py`/`constraints.py`.

## Design

### New file: `src/lammpsparser/compatibility/mlip.py`

#### `write_mlip_input_file(working_directory, mtp_filename, active_learning=False, threshold=2.0, threshold_break=5.0, save_selected="selected.cfg", load_state="state.mvs", log="selection.log", file_name="mlip.ini") -> str`

Writes the `mlip.ini` control file. `mtp_filename` is the absolute path to the underlying `.mtp` potential file (resolved by the caller the same way other potential file paths are resolved elsewhere in `lammpsparser`, e.g. via `potential["Filename"][0]` after `update_potential_paths`). Creates `working_directory` if it doesn't exist. Returns the path to the written file.

Content when `active_learning=False`:
```
mtp-filename <mtp_filename>
select FALSE
```

Content when `active_learning=True`:
```
mtp-filename <mtp_filename>
calculate-efs TRUE
select TRUE
select:threshold <threshold>
select:threshold-break <threshold_break>
select:save-selected <save_selected>
select:load-state <load_state>
select:log <log>
write-cfgs:skip 0
```

Replaces `MlipParameter`, `enable_active_learning()`, and the `mlip.ini`-writing part of `write_input()` from the legacy job.

#### `check_mlip_convergence(working_directory, error_file_name="error.out") -> bool`

Direct port of `LammpsMlip.convergence_check()`. Returns `True` if `error.out` doesn't exist in `working_directory`. Otherwise scans line-by-line for a line starting with `"MLIP: Breaking threshold exceeded"`; returns `False` if found, `True` otherwise.

#### `MlipConfiguration` dataclass

Represents one parsed `.cfg` entry:

- `cell: np.ndarray` — shape `(3, 3)`, from the `Supercell` section.
- `positions: np.ndarray` — shape `(N, 3)`, Cartesian.
- `types: np.ndarray` — shape `(N,)`, integer species-type indices (0-based, as written by MLIP).
- `forces: Optional[np.ndarray]` — shape `(N, 3)` if the `AtomData:` header includes `fx fy fz`, else `None`.
- `energy: Optional[float]` — from the `Energy` section, if present.
- `stress: Optional[np.ndarray]` — shape `(6,)` in ASE Voigt order `(xx, yy, zz, yz, xz, xy)`, from `PlusStress:`, if present. The `PlusStress:` header is parsed to map columns regardless of order.
- `grade: Optional[float]` — from a `Feature   MV_grade   <value>` line, if present.

#### `load_mlip_cfgs(file_name) -> list[MlipConfiguration]`

Parses a `.cfg` file (one or more `BEGIN_CFG` / `END_CFG` blocks) into a list of `MlipConfiguration`. Implemented as plain line-by-line parsing (no regex needed): each block is scanned for `Size`, `Supercell`, `AtomData:`, `Energy`, `PlusStress:`, and `Feature ... MV_grade ...` sections. `Supercell` and `AtomData:` are required in every block (`ValueError` if missing); everything else is optional. Supports both `cartes_x/y/z` (Cartesian, used as-is) and `direct_x/y/z` (fractional, converted via `positions = fractional @ cell`) atom-data columns.

This replaces the dependency on `pyiron_potentialfit.mlip.cfgs.loadcfgs`.

#### `get_mlip_selected_structures(file_name, species) -> list[ase.atoms.Atoms]`

Calls `load_mlip_cfgs(file_name)` and builds one `Atoms` object per configuration:

- `symbols = np.asarray(species)[cfg.types.astype(int)]` (mirrors the legacy `self.potential.Species.iloc[0]` lookup).
- `positions=cfg.positions`, `cell=cfg.cell`, `pbc=True`.
- If `cfg.forces is not None`: `atoms.arrays["forces"] = cfg.forces`.
- If `cfg.energy is not None`: `atoms.info["energy"] = cfg.energy`.
- If `cfg.stress is not None`: `atoms.info["stress"] = cfg.stress`.
- If `cfg.grade is not None`: `atoms.info["mv_grade"] = cfg.grade`.

No `SinglePointCalculator` is attached — per-structure MLIP predictions are exposed as plain `info`/`arrays` entries.

Replaces `LammpsMlip.collect_output()`'s `.cfg`-parsing loop, the `selected_structures` property, and `StructureStorage`.

### `src/lammpsparser/__init__.py`

Add `write_mlip_input_file`, `check_mlip_convergence`, `load_mlip_cfgs`, and `get_mlip_selected_structures` to the top-level imports and `__all__`, matching how other `compatibility` functions (`calc_md`, `lammps_file_interface_function`, etc.) are already re-exported. `MlipConfiguration` stays module-scoped (not re-exported at top level), consistent with e.g. `CalcMDInput`/`CalcMinimizeInput` in `data.py` not being re-exported either.

### Tests: `tests/test_compatibility_mlip.py`

- `write_mlip_input_file`: assert file contents for `active_learning=False` and `active_learning=True`, and that `working_directory` is created if missing.
- `check_mlip_convergence`: no `error.out` → `True`; `error.out` present without the breaking-threshold line → `True`; `error.out` with the breaking-threshold line → `False`.
- `load_mlip_cfgs` / `get_mlip_selected_structures`: against a small hand-written fixture at `tests/static/mlip/selected.cfg` containing two `BEGIN_CFG`/`END_CFG` blocks — one with `Feature MV_grade`, one without — covering cell, positions, types→species mapping, forces, energy, and stress parsing.

## Self-review notes

- Scope is a single new module plus one `__init__.py` edit and one test file — no decomposition needed.
- Verified no naming collisions with existing `compatibility` modules or top-level exports.
- Verified `PlusStress:` column order in real MLIP output is `xx yy zz yz xz xy`, which already matches ASE's Voigt order, but the parser maps by header name rather than assuming position, so a differently-ordered header would still parse correctly.

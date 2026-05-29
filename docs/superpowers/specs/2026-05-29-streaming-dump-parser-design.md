# Streaming Dump Parser Design

**Date:** 2026-05-29  
**Status:** Approved

## Goal

Add a memory-efficient streaming iterator `iter_lammps_frames()` that yields one parsed frame at a time from a LAMMPS text dump file, enabling users to process large HPC trajectories without loading the entire file into memory.

## Motivation

The existing `parse_lammps_output()` reads all frames into memory at once. For large simulations (thousands of frames, millions of atoms), this is a hard memory wall. A streaming generator solves this and makes the package more attractive to HPC users — a common audience for LAMMPS tools.

## Architecture

Two layers, both following the existing separation between `output_raw.py` (low-level file I/O) and `output.py` (unit conversion, coordinate transforms, public API).

### `output_raw.py` — `_iter_raw_frames(file_name, start, stop, step)`

A private generator that reads `dump.out` line by line. For each frame:
1. Reads the `ITEM: TIMESTEP`, `ITEM: NUMBER OF ATOMS`, `ITEM: BOX BOUNDS`, and `ITEM: ATOMS` blocks.
2. Parses the atoms block into a pandas DataFrame (same approach as the existing `parse_raw_dump_from_text`).
3. Yields a dict with the same keys as `DumpData`: `step`, `natoms`, `cells`, `indices`, `forces`, `velocities`, `positions`, `unwrapped_positions`, `computes`, etc.
4. Applies `start`/`stop`/`step` slicing by frame index — frames outside the range are read but not yielded (the file must still be scanned linearly).

The existing `parse_raw_dump_from_text` is refactored to call `_iter_raw_frames` internally and accumulate into lists, preserving identical behavior.

### `output.py` — `iter_lammps_frames(...)` (public)

A public generator that wraps `_iter_raw_frames` and applies per-frame:
- Coordinate rotation (inverse of LAMMPS triclinic rotation via `UnfoldingPrism`)
- Index remapping via `remap_indices_funct`
- Unit conversion via `UnitConverter`

Yields a `LammpsFrame` dataclass per frame.

## Public API

```python
from lammpsparser import iter_lammps_frames

for frame in iter_lammps_frames(
    working_directory="./output",
    structure=atoms,
    potential_elements=["Ni", "Al"],
    units="metal",
    dump_out_file_name="dump.out",   # default
    start=500,                        # skip warmup frames
    stop=None,                        # read to end
    step=1,                           # every frame
    fields=None,                      # all columns; pass a set to restrict
):
    print(frame.step, frame.positions.shape)
```

### `LammpsFrame` dataclass fields

| Field | Type | Notes |
|---|---|---|
| `step` | `int` | Simulation timestep |
| `cell` | `np.ndarray` (3×3) | Unit-converted cell matrix |
| `positions` | `np.ndarray` (N×3) | Wrapped Cartesian positions |
| `forces` | `np.ndarray` (N×3) | Unit-converted forces |
| `indices` | `np.ndarray` (N,) | Structure indices (remapped) |
| `velocities` | `np.ndarray` \| `None` | Present only if dump contains `vx vy vz` |
| `computes` | `dict` | Per-atom compute results keyed by compute ID |

### Scope exclusions

- H5MD (`dump.h5`) is out of scope — h5py already supports lazy dataset access natively.
- The log file (`log.lammps`) is not streamed — thermo data is small and always fits in memory.

## Error handling

- `FileNotFoundError` if `dump_out_file_name` does not exist.
- `ValueError` with the offending step number if a frame is malformed (truncated file or unexpected EOF mid-frame).
- No silent partial results.

## Testing

1. **Equivalence test:** Stream all frames from a multi-frame fixture, accumulate into dicts, assert result matches `parse_lammps_output()` output exactly.
2. **Memory test:** Generate a large synthetic dump (many atoms × many frames) using `tracemalloc`, assert peak memory during streaming is a small fraction of what full load would require.
3. **Slicing test:** Verify `start`, `stop`, `step` parameters return the correct subset of frames.
4. **Robustness test:** Truncated dump file raises `ValueError` with the correct step number.

## Files changed

| File | Change |
|---|---|
| `src/lammpsparser/output_raw.py` | Add `_iter_raw_frames`; refactor `parse_raw_dump_from_text` to use it |
| `src/lammpsparser/output.py` | Add `LammpsFrame` dataclass and `iter_lammps_frames` generator |
| `src/lammpsparser/__init__.py` | Export `iter_lammps_frames` |
| `tests/` | Add test module for streaming behavior |
| `notebooks/example.ipynb` | Add streaming example cell |

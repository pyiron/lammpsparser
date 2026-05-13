from ase.atoms import Atoms


def structure_from_parsed_output(
    initial_structure: Atoms,
    parsed_output: dict,
    *,
    wrap: bool = False,
    index: int = -1,
) -> Atoms:
    """Construct an `Atoms` object from parsed output data.

    Args:
        initial_structure: The initial atomic structure to use as a template.
        parsed_output: Parsed output containing atomic positions, cell, and indices.
        wrap: Whether to wrap the atomic positions to the simulation cell (default is False).
            Keeping the unwrapped positions is more beneficial if structures are passed between
            different LAMMPS simulations in one workflow to ensure continuity.
        index: The index of the frame to use from the parsed output (default is -1, i.e., the last frame).

    Returns:
        An `Atoms` object with updated positions and cell.

    Example:
        >>> new_atoms = structure_from_parsed_output(atoms, lammps_output)

    """
    # Take a copy of the initial structure as template and update the relevant properties
    atoms_copy = initial_structure.copy()
    atoms_copy.set_array("indices", parsed_output["generic"]["indices"][index])
    atoms_copy.set_positions(parsed_output["generic"]["positions"][index])
    atoms_copy.set_velocities(parsed_output["generic"]["velocities"][index])
    atoms_copy.set_cell(parsed_output["generic"]["cells"][index])
    atoms_copy.set_pbc(True)
    if wrap:
        atoms_copy.wrap()

    return atoms_copy

import numpy as np
from ase.atoms import Atoms


def _get_fixed_atom_boolean_vector(structure: Atoms) -> np.ndarray:
    """
    Convert ASE constraints to a per-atom, per-direction boolean array.

    Translates ASE :class:`~ase.constraints.FixAtoms` and
    :class:`~ase.constraints.FixedPlane` constraints on ``structure`` into a
    boolean array where ``True`` indicates that the corresponding degree of
    freedom is frozen.

    Args:
        structure (ase.atoms.Atoms): Structure whose ``constraints`` attribute
            is inspected.

    Returns:
        numpy.ndarray: Boolean array of shape ``(N, 3)`` where ``True`` means
        the atom's motion in that Cartesian direction is fixed.

    Raises:
        ValueError: If a constraint type other than ``FixAtoms`` or
            ``FixedPlane`` is encountered, or if the ``FixedPlane`` direction
            is not one of the supported axis combinations.
    """
    fixed_atom_vector = np.array([[False, False, False]] * len(structure))
    for c in structure.constraints:
        c_dict = c.todict()
        if c_dict["name"] == "FixAtoms":
            fixed_atom_vector[c_dict["kwargs"]["indices"]] = [True, True, True]
        elif c_dict["name"] == "FixedPlane":
            if all(np.isin(c_dict["kwargs"]["direction"], [0, 1, 1 / np.sqrt(2)])):
                if "indices" in c_dict["kwargs"]:
                    fixed_atom_vector[c_dict["kwargs"]["indices"]] = np.array(
                        c_dict["kwargs"]["direction"]
                    ).astype(bool)
                elif "a" in c_dict["kwargs"]:
                    fixed_atom_vector[c_dict["kwargs"]["a"]] = np.array(
                        c_dict["kwargs"]["direction"]
                    ).astype(bool)
            else:
                raise ValueError(
                    "Currently the directions are limited to [1, 0, 0], [1, 1, 0], [1, 1, 1] and its permutations."
                )
        else:
            raise ValueError("Only FixAtoms and FixedPlane are currently supported. ")
    return fixed_atom_vector


def set_selective_dynamics(structure: Atoms, calc_md: bool = False) -> dict[str, str]:
    """
    Translate ASE constraints into LAMMPS ``group`` / ``fix setforce`` commands.

    Groups atoms by their frozen degrees of freedom and generates the
    corresponding LAMMPS commands to zero the force (and optionally the
    velocity) on those atoms.  Supports constraints on individual Cartesian
    directions (x, y, z) as well as the combined xy, yz, and zx pairs.

    The LAMMPS equivalent of ASE's ``FixAtoms`` is ``fix setforce 0.0 0.0 0.0``
    and ``FixedPlane`` along an axis is ``fix setforce 0.0 NULL NULL`` (or the
    appropriate permutation).

    Args:
        structure (ase.atoms.Atoms): Structure whose ``constraints`` are
            mapped to LAMMPS commands.
        calc_md (bool): If ``True``, also add ``velocity … set`` commands to
            initialise constrained atoms with zero velocity (required for MD
            to prevent initial kinetic energy in frozen directions).

    Returns:
        dict[str, str]: Ordered mapping of LAMMPS command keyword to the rest
        of the command line.  For example::

            {
                "group constraintxyz": "id 1 2 3",
                "fix constraintxyz": "constraintxyz setforce 0.0 0.0 0.0",
                "velocity constraintxyz": "set 0.0 0.0 0.0",  # only when calc_md=True
            }

        Returns an empty dict if the structure has no constraints.
    """
    control_dict: dict[str, str] = {}
    if len(structure.constraints) > 0:
        sel_dyn = _get_fixed_atom_boolean_vector(structure=structure)
        # Enter loop only if constraints present
        if len(np.argwhere(np.any(sel_dyn, axis=1)).flatten()) != 0:
            all_indices = np.arange(len(structure), dtype=int)
            constraint_xyz = np.argwhere(np.all(sel_dyn, axis=1)).flatten()
            not_constrained_xyz = np.setdiff1d(all_indices, constraint_xyz)
            # LAMMPS starts counting from 1
            constraint_xyz += 1
            ind_x = np.argwhere(sel_dyn[not_constrained_xyz, 0]).flatten()
            ind_y = np.argwhere(sel_dyn[not_constrained_xyz, 1]).flatten()
            ind_z = np.argwhere(sel_dyn[not_constrained_xyz, 2]).flatten()
            constraint_xy = not_constrained_xyz[np.intersect1d(ind_x, ind_y)] + 1
            constraint_yz = not_constrained_xyz[np.intersect1d(ind_y, ind_z)] + 1
            constraint_zx = not_constrained_xyz[np.intersect1d(ind_z, ind_x)] + 1
            constraint_x = (
                not_constrained_xyz[np.setdiff1d(np.setdiff1d(ind_x, ind_y), ind_z)] + 1
            )
            constraint_y = (
                not_constrained_xyz[np.setdiff1d(np.setdiff1d(ind_y, ind_z), ind_x)] + 1
            )
            constraint_z = (
                not_constrained_xyz[np.setdiff1d(np.setdiff1d(ind_z, ind_x), ind_y)] + 1
            )
            control_dict = {}
            if len(constraint_xyz) > 0:
                control_dict["group constraintxyz"] = "id " + " ".join(
                    [str(ind) for ind in constraint_xyz]
                )
                control_dict["fix constraintxyz"] = "constraintxyz setforce 0.0 0.0 0.0"
                if calc_md:
                    control_dict["velocity constraintxyz"] = "set 0.0 0.0 0.0"
            if len(constraint_xy) > 0:
                control_dict["group constraintxy"] = "id " + " ".join(
                    [str(ind) for ind in constraint_xy]
                )
                control_dict["fix constraintxy"] = "constraintxy setforce 0.0 0.0 NULL"
                if calc_md:
                    control_dict["velocity constraintxy"] = "set 0.0 0.0 NULL"
            if len(constraint_yz) > 0:
                control_dict["group constraintyz"] = "id " + " ".join(
                    [str(ind) for ind in constraint_yz]
                )
                control_dict["fix constraintyz"] = "constraintyz setforce NULL 0.0 0.0"
                if calc_md:
                    control_dict["velocity constraintyz"] = "set NULL 0.0 0.0"
            if len(constraint_zx) > 0:
                control_dict["group constraintxz"] = "id " + " ".join(
                    [str(ind) for ind in constraint_zx]
                )
                control_dict["fix constraintxz"] = "constraintxz setforce 0.0 NULL 0.0"
                if calc_md:
                    control_dict["velocity constraintxz"] = "set 0.0 NULL 0.0"
            if len(constraint_x) > 0:
                control_dict["group constraintx"] = "id " + " ".join(
                    [str(ind) for ind in constraint_x]
                )
                control_dict["fix constraintx"] = "constraintx setforce 0.0 NULL NULL"
                if calc_md:
                    control_dict["velocity constraintx"] = "set 0.0 NULL NULL"
            if len(constraint_y) > 0:
                control_dict["group constrainty"] = "id " + " ".join(
                    [str(ind) for ind in constraint_y]
                )
                control_dict["fix constrainty"] = "constrainty setforce NULL 0.0 NULL"
                if calc_md:
                    control_dict["velocity constrainty"] = "set NULL 0.0 NULL"
            if len(constraint_z) > 0:
                control_dict["group constraintz"] = "id " + " ".join(
                    [str(ind) for ind in constraint_z]
                )
                control_dict["fix constraintz"] = "constraintz setforce NULL NULL 0.0"
                if calc_md:
                    control_dict["velocity constraintz"] = "set NULL NULL 0.0"
    return control_dict

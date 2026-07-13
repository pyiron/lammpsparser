import os


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
    file_path = os.path.abspath(os.path.join(working_directory, file_name))
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

import os
import shutil
import subprocess
from typing import Optional

import pandas
from ase.atoms import Atoms

from lammpsparser.compatibility.calculate import (
    calc_md,
    calc_minimize,
    calc_static,
)
from lammpsparser.compatibility.constraints import set_selective_dynamics
from lammpsparser.output import parse_lammps_output
from lammpsparser.potential import get_potential_by_name
from lammpsparser.structure import write_lammps_datafile


def lammps_file_interface_function(
    working_directory: str,
    structure: Atoms,
    potential: str,
    calc_mode: str = "static",
    calc_kwargs: Optional[dict] = None,
    units: str = "metal",
    lmp_command: Optional[str] = None,
    resource_path: Optional[str] = None,
    input_control_file: Optional[dict] = None,
    write_restart_file: bool = False,
    read_restart_file: bool = False,
    restart_file: str = "restart.out",
):
    """
    A single function to execute a LAMMPS calculation based on the LAMMPS job implemented in pyiron

    Examples:

    >>> import os
    >>> from ase.build import bulk
    >>> from pyiron_atomistics.lammps.lammps import lammps_function
    >>>
    >>> shell_output, parsed_output, job_crashed = lammps_function(
    ...     working_directory=os.path.abspath("lmp_working_directory"),
    ...     structure=bulk("Al", cubic=True),
    ...     potential='2009--Mendelev-M-I--Al-Mg--LAMMPS--ipr1',
    ...     calc_mode="md",
    ...     calc_kwargs={"temperature": 500.0, "pressure": 0.0, "n_ionic_steps": 1000, "n_print": 100},
    ...     cutoff_radius=None,
    ...     units="metal",
    ...     bonds_kwargs={},
    ...      enable_h5md=False,
    ... )

    Args:
        working_directory (str): directory in which the LAMMPS calculation is executed
        structure (Atoms): ase.atoms.Atoms - atomistic structure
        potential (str): Name of the LAMMPS potential based on the NIST database and the OpenKIM database
        calc_mode (str): select mode of calculation ["static", "md", "minimize", "vcsgc"]
        calc_kwargs (dict): key-word arguments for the calculate function, the input parameters depend on the calc_mode:
          "static": No parameters
          "md": "temperature", "pressure", "n_ionic_steps", "time_step", "n_print", "temperature_damping_timescale",
                "pressure_damping_timescale", "seed", "tloop", "initial_temperature", "langevin", "delta_temp",
                "delta_press", job_name", "rotation_matrix"
          "minimize": "ionic_energy_tolerance", "ionic_force_tolerance", "max_iter", "pressure", "n_print", "style",
                      "rotation_matrix"
          "vcsgc": "mu", "ordered_element_list", "target_concentration", "kappa", "mc_step_interval", "swap_fraction",
                   "temperature_mc", "window_size", "window_moves", "temperature", "pressure", "n_ionic_steps",
                   "time_step", "n_print", "temperature_damping_timescale", "pressure_damping_timescale", "seed",
                   "initial_temperature", "langevin", "job_name", "rotation_matrix"
        cutoff_radius (float): cut-off radius for the interatomic potential
        units (str): Units for LAMMPS
        bonds_kwargs (dict): key-word arguments to create atomistic bonds:
          "species", "element_list", "cutoff_list", "max_bond_list", "bond_type_list", "angle_type_list",
        server_kwargs (dict): key-word arguments to create server object - the available parameters are:
          "user", "host", "run_mode", "queue", "qid", "cores", "threads", "new_h5", "structure_id", "run_time",
          "memory_limit", "accept_crash", "additional_arguments", "gpus", "conda_environment_name",
          "conda_environment_path"
        enable_h5md (bool): activate h5md mode for LAMMPS
        write_restart_file (bool): enable writing the LAMMPS restart file
        read_restart_file (bool): enable loading the LAMMPS restart file
        restart_file (str): file name of the LAMMPS restart file to copy
        executable_version (str): LAMMPS version to for the execution
        executable_path (str): path to the LAMMPS executable
        input_control_file (str|list|dict): Option to modify the LAMMPS input file directly

    Returns:
        str, dict, bool: Tuple consisting of the shell output (str), the parsed output (dict) and a boolean flag if
                         the execution raised an accepted error.
    """
    if calc_kwargs is None:
        calc_kwargs = {}
    else:
        calc_kwargs = calc_kwargs.copy()

    if lmp_command is None:
        lmp_command = os.getenv("ASE_LAMMPSRUN_COMMAND", "mpiexec -n 1 --oversubscribe lmp_mpi") + " -in lmp.in"

    os.makedirs(working_directory, exist_ok=True)
    potential_lst, potential_replace, species = _get_potential(
        potential=potential, resource_path=resource_path
    )

    lmp_str_lst = []
    atom_type = "atomic"
    for l in lammps_file_initialization(
        structure=structure,
        units=units,
        read_restart_file=read_restart_file,
        restart_file=restart_file,
    ):
        if l.strip().startswith("units") and "units" in potential_replace:
            lmp_str_lst.append(potential_replace["units"])
        elif l.strip().startswith("atom_style") and "atom_style" in potential_replace:
            lmp_str_lst.append(potential_replace["atom_style"])
            atom_type = potential_replace["atom_style"].split()[-1]
        elif l.strip().startswith("dimension") and "dimension" in potential_replace:
            lmp_str_lst.append(potential_replace["dimension"])
        else:
            lmp_str_lst.append(l)

    lmp_str_lst += potential_lst
    lmp_str_lst += ["variable dumptime equal {} ".format(calc_kwargs.get("n_print", 1))]
    lmp_str_lst += [
        "dump 1 all custom ${dumptime} dump.out id type xsu ysu zsu fx fy fz vx vy vz",
        'dump_modify 1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"',
    ]

    if calc_mode == "static":
        lmp_str_lst += [
            k + " " + v
            for k, v in set_selective_dynamics(
                structure=structure, calc_md=False
            ).items()
        ]
        lmp_str_tmp_lst = calc_static()
        lmp_str_lst = _modify_input_dict(
            input_control_file=input_control_file,
            lmp_str_lst=lmp_str_lst + lmp_str_tmp_lst[:-1],
        )
        lmp_str_lst.append(lmp_str_tmp_lst[-1])
    elif calc_mode == "md":
        lmp_str_lst += [
            k + " " + v
            for k, v in set_selective_dynamics(
                structure=structure, calc_md=True
            ).items()
        ]
        if "n_ionic_steps" in calc_kwargs.keys():
            n_ionic_steps = int(calc_kwargs.pop("n_ionic_steps"))
        else:
            n_ionic_steps = 1
        if read_restart_file:
            calc_kwargs["initial_temperature"] = 0.0
        calc_kwargs["units"] = units
        lmp_str_lst += calc_md(**calc_kwargs)
        lmp_str_lst = _modify_input_dict(
            input_control_file=input_control_file,
            lmp_str_lst=lmp_str_lst,
        )
        if read_restart_file:
            lmp_str_lst += ["reset_timestep 0"]
        lmp_str_lst += ["run {} ".format(n_ionic_steps)]
    elif calc_mode == "minimize":
        calc_kwargs["units"] = units
        lmp_str_lst += [
            k + " " + v
            for k, v in set_selective_dynamics(
                structure=structure, calc_md=False
            ).items()
        ]
        lmp_str_tmp_lst, structure = calc_minimize(structure=structure, **calc_kwargs)
        lmp_str_lst = _modify_input_dict(
            input_control_file=input_control_file,
            lmp_str_lst=lmp_str_lst + lmp_str_tmp_lst[:-1],
        )
        lmp_str_lst.append(lmp_str_tmp_lst[-1])
    else:
        raise ValueError(
            f"calc_mode must be one of: static, md or minimize, not {calc_mode}"
        )
    if read_restart_file:
        shutil.copyfile(
            os.path.abspath(restart_file),
            os.path.join(working_directory, os.path.basename(restart_file)),
        )
    if write_restart_file:
        lmp_str_lst.append(f"write_restart {os.path.basename(restart_file)}")

    with open(os.path.join(working_directory, "lmp.in"), "w") as f:
        f.writelines([l + "\n" for l in lmp_str_lst])

    write_lammps_datafile(
        structure=structure,
        potential_elements=species,
        bond_dict=None,
        units=units,
        file_name="lammps.data",
        working_directory=working_directory,
        atom_type=atom_type,
    )

    shell = subprocess.check_output(
        lmp_command,
        cwd=working_directory,
        shell=True,
        universal_newlines=True,
        env=os.environ.copy(),
    )
    output = parse_lammps_output(
        working_directory=working_directory,
        structure=structure,
        potential_elements=species,
        units=units,
        prism=None,
        dump_h5_file_name="dump.h5",
        dump_out_file_name="dump.out",
        log_lammps_file_name="log.lammps",
    )
    return shell, output, False


def lammps_file_initialization(
    structure, dimension=3, units="metal", read_restart_file=False, restart_file=None
):
    init_commands = ["units " + units]
    boundary = " ".join(["p" if coord else "f" for coord in structure.pbc])
    if read_restart_file:
        init_commands.append(f"read_restart {os.path.basename(restart_file)}")
    else:
        init_commands += [
            "dimension " + str(dimension),
            "boundary " + boundary + "",
            "atom_style atomic",
            "read_data lammps.data",
        ]
    return init_commands


def _modify_input_dict(
    input_control_file: Optional[dict] = None,
    lmp_str_lst: list[str] = [],
):
    if input_control_file is not None:
        lmp_tmp_lst, keys_used = [], []
        for l in lmp_str_lst:
            ls = l.split()
            if len(ls) >= 1:  # Remove empty lines
                key = ls[0]
                if key in input_control_file.keys():
                    lmp_tmp_lst.append(key + " " + input_control_file[key])
                    keys_used.append(key)
                else:
                    lmp_tmp_lst.append(l)
        for k, v in input_control_file.items():
            if k not in keys_used:
                lmp_tmp_lst.append(k + " " + v)

        return lmp_tmp_lst
    else:
        return lmp_str_lst


def _get_potential(potential, resource_path: Optional[str] = None):
    if isinstance(potential, str):
        potential_dataframe = get_potential_by_name(
            potential_name=potential, resource_path=resource_path
        )
    elif isinstance(potential, pandas.DataFrame):
        potential_dataframe = potential.iloc[0]
    elif isinstance(potential, pandas.Series):
        potential_dataframe = potential
    else:
        raise TypeError()

    potential_replace = {}
    potential_lst = []
    for l in potential_dataframe["Config"]:
        if l.startswith("units"):
            potential_replace["units"] = l
        elif l.startswith("atom_style"):
            potential_replace["atom_style"] = l
        elif l.startswith("dimension"):
            potential_replace["dimension"] = l
        else:
            potential_lst.append(l)

    return potential_lst, potential_replace, potential_dataframe["Species"]

import warnings
from typing import Any, cast

import numpy as np
from ase.atoms import Atoms

from lammpsparser.structure import UnfoldingPrism
from lammpsparser.units import LAMMPS_UNIT_CONVERSIONS


def calc_md(
    temperature=None,
    pressure=None,
    time_step=1.0,
    n_print=100,
    temperature_damping_timescale=100.0,
    pressure_damping_timescale=1000.0,
    seed=80996,
    tloop=None,
    initial_temperature=None,
    langevin=False,
    delta_temp=None,
    delta_press=None,
    rotation_matrix=None,
    units="metal",
):
    """
    Set an MD calculation within LAMMPS. Nosé Hoover is used by default.

    Args:
        temperature (None/float/list): Target temperature value(-s). If set to None, an NVE
            calculation is performed. It is required when the pressure is set or langevin is set
            It can be a list of temperature values, containing the initial target temperature and
            the final target temperature (in between the target value is varied linearly).
        pressure (None/float/numpy.ndarray/list): Target pressure. If set to None, an NVE or an
            NVT calculation is performed. If set to a scalar, the shear of the cell and the
            ratio of the x, y, and z components is kept constant, while an isotropic, hydrostatic
            pressure is applied. A list of up to length 6 can be given to specify xx, yy, zz, xy,
            xz, and yz components of the pressure tensor, respectively. These values can mix
            floats and `None` to allow only certain degrees of cell freedom to change. (Default
            is None, run isochorically.)
        n_ionic_steps (int): Number of ionic steps
        time_step (float): Step size in fs between two steps.
        n_print (int):  Print frequency
        temperature_damping_timescale (float): The time associated with the thermostat adjusting
            the temperature.  (In fs. After rescaling to appropriate time units, is equivalent to
            Lammps' `Tdamp`.)
        pressure_damping_timescale (float): The time associated with the barostat adjusting the
            temperature.  (In fs. After rescaling to appropriate time units, is equivalent to
            Lammps' `Pdamp`.)
        seed (int):  Seed for the random number generation used for the intiial velocity creation
            and langevin dynamics - otherwise ignored. If not specified, the seed is created via
            job name
        tloop:
        initial_temperature (None/float):  Initial temperature according to which the initial
            velocity field is created. If None, the initial temperature will be twice the target
            temperature (which would go immediately down to the target temperature as described
            in equipartition theorem). If 0, the velocity field is not initialized (in which case
            the initial velocity given in structure will be used). If any other number is given,
            this value is going to be used for the initial temperature.
        langevin (bool): (True or False) Activate Langevin dynamics
        delta_temp (float): Thermostat timescale, but in your Lammps time units, whatever those
            are. (DEPRECATED.)
        delta_press (float): Barostat timescale, but in your Lammps time units, whatever those
            are. (DEPRECATED.)
        job_name (str): Job name of the job to generate a unique random seed.
        rotation_matrix (numpy.ndarray): The rotation matrix from the pyiron to Lammps coordinate
            frame.
    """
    if units not in LAMMPS_UNIT_CONVERSIONS.keys():
        raise NotImplementedError
    time_units = LAMMPS_UNIT_CONVERSIONS[units]["time"]
    temperature_units = LAMMPS_UNIT_CONVERSIONS[units]["temperature"]

    # Transform time
    if time_step is not None:
        time_step *= time_units

    # Transform thermostat strength (time)
    if delta_temp is not None:
        warnings.warn(
            "WARNING: `delta_temp` is deprecated, please use `temperature_damping_timescale`."
        )
        temperature_damping_timescale = delta_temp
    else:
        temperature_damping_timescale *= time_units

    # Transform barostat strength (time)
    if delta_press is not None:
        warnings.warn(
            "WARNING: `delta_press` is deprecated, please use `pressure_damping_timescale`."
        )
        pressure_damping_timescale = delta_press
    else:
        pressure_damping_timescale *= time_units

    # Transform temperature
    if temperature is not None:
        temperature = np.array([temperature], dtype=float).flatten()
        if len(temperature) == 1:
            temperature = np.array(2 * temperature.tolist())
        elif len(temperature) != 2:
            raise ValueError(
                "At most two temperatures can be provided "
                "(for a linearly ramping target temperature), "
                "but got {}".format(len(temperature))
            )
        temperature *= temperature_units

    # Apply initial overheating (default uses the theorem of equipartition of energy between KE and PE)
    if initial_temperature is None and temperature is not None:
        initial_temperature = 2 * temperature[0]

    if seed <= 0:
        raise ValueError("Seed must be a positive integer larger than 0")

    thermo_str = ""
    # Set thermodynamic ensemble
    if pressure is not None:  # NPT
        if temperature is None or temperature.min() <= 0:
            raise ValueError(
                "Target temperature for fix nvt/npt/nph cannot be 0 or negative"
            )

        # force_skewed = False
        pressure = _pressure_to_lammps(
            pressure=pressure, rotation_matrix=rotation_matrix, units=units
        )

        if np.isscalar(pressure):
            pressure_string = " iso {0} {0} {1}".format(
                pressure, pressure_damping_timescale
            )
        else:
            pressure_string = ""
            for ii, (coord, value) in enumerate(
                zip(["x", "y", "z", "xy", "xz", "yz"], pressure)
            ):
                if value is not None:
                    pressure_string += " {0} {1} {1} {2}".format(
                        coord, value, pressure_damping_timescale
                    )
                    # if ii > 2:
                    #     force_skewed = True

        if langevin:  # NPT(Langevin)
            fix_ensemble_str = "fix ensemble all nph" + pressure_string
            thermo_str = "fix langevin all langevin {0} {1} {2} {3} zero yes".format(
                str(temperature[0]),
                str(temperature[1]),
                str(temperature_damping_timescale),
                str(seed),
            )
        else:  # NPT(Nose-Hoover)
            fix_ensemble_str = "fix ensemble all npt temp {0} {1} {2}".format(
                str(temperature[0]),
                str(temperature[1]),
                str(temperature_damping_timescale),
            )
            fix_ensemble_str += pressure_string
    elif temperature is not None:  # NVT
        if temperature.min() <= 0:
            raise ValueError(
                "Target temperature for fix nvt/npt/nph cannot be 0 or negative"
            )

        if langevin:  # NVT(Langevin)
            fix_ensemble_str = "fix ensemble all nve"
            thermo_str = "fix langevin all langevin {0} {1} {2} {3} zero yes".format(
                str(temperature[0]),
                str(temperature[1]),
                str(temperature_damping_timescale),
                str(seed),
            )
        else:  # NVT(Nose-Hoover)
            fix_ensemble_str = "fix ensemble all nvt temp {0} {1} {2}".format(
                str(temperature[0]),
                str(temperature[1]),
                str(temperature_damping_timescale),
            )
    else:  # NVE
        if langevin:
            warnings.warn("Temperature not set; Langevin ignored.")
        fix_ensemble_str = "fix ensemble all nve"

    if tloop is not None:
        fix_ensemble_str += " tloop " + str(tloop)

    line_lst = [fix_ensemble_str]
    if thermo_str != "":
        line_lst.append(thermo_str)

    line_lst.append("variable thermotime equal {} ".format(n_print))
    line_lst.append("timestep {}".format(time_step))
    if initial_temperature is not None and initial_temperature > 0:
        line_lst.append(
            _set_initial_velocity(
                temperature=initial_temperature,
                seed=seed,
                gaussian=True,
                append_value=False,
                zero_lin_momentum=True,
                zero_rot_momentum=True,
            )
        )
    line_lst += _get_thermo()
    return line_lst


def calc_minimize(
    structure: Atoms,
    ionic_energy_tolerance=0.0,
    ionic_force_tolerance=1e-4,
    max_iter=100000,
    pressure=None,
    n_print=100,
    style="cg",
    rotation_matrix=None,
    units="metal",
):
    """
    Sets parameters required for minimization.

    Args:
        ionic_energy_tolerance (float): If the magnitude of difference between energies of two consecutive steps is
            lower than or equal to `ionic_energy_tolerance`, the minimisation terminates. (Default is 0.0 eV.)
        ionic_force_tolerance (float): If the magnitude of the global force vector at a step is lower than or equal
            to `ionic_force_tolerance`, the minimisation terminates. (Default is 1e-4 eV/angstrom.)
        e_tol (float): Same as ionic_energy_tolerance (deprecated)
        f_tol (float): Same as ionic_force_tolerance (deprecated)
        max_iter (int): Maximum number of minimisation steps to carry out. If the minimisation converges before
            `max_iter` steps, terminate at the converged step. If the minimisation does not converge up to
            `max_iter` steps, terminate at the `max_iter` step. (Default is 100000.)
        pressure (None/float/numpy.ndarray/list): Target pressure. If set to None, an isochoric (constant V)
            calculation is performed. If set to a scalar, the shear of the cell and the ratio of the x, y, and
            z components is kept constant, while an isotropic, hydrostatic pressure is applied. A list of up to
            length 6 can be given to specify xx, yy, zz, xy, xz, and yz components of the pressure tensor,
            respectively. These values can mix floats and `None` to allow only certain degrees of cell freedom
            to change. (Default is None, run isochorically.)
        n_print (int): Write (dump or print) to the output file every n steps (Default: 100)
        style ('cg'/'sd'/other values from Lammps docs): The style of the numeric minimization, either conjugate
            gradient, steepest descent, or other keys permissible from the Lammps docs on 'min_style'. (Default
            is 'cg' -- conjugate gradient.)
        rotation_matrix (numpy.ndarray): The rotation matrix from the pyiron to Lammps coordinate frame.
    """
    # This docstring is a source for the calc_minimize method in pyiron_atomistics.lammps.base.LammpsBase.calc_minimize and
    # pyiron_atomistics.lammps.interactive.LammpsInteractive.calc_minimize -- Please ensure that changes to signature or
    # defaults stay consistent!

    max_evaluations = 100 * max_iter
    if n_print > max_iter:
        warnings.warn("n_print larger than max_iter, adjusting to n_print=max_iter")
        n_print = max_iter

    if units not in LAMMPS_UNIT_CONVERSIONS.keys():
        raise NotImplementedError
    energy_units = LAMMPS_UNIT_CONVERSIONS[units]["energy"]
    force_units = LAMMPS_UNIT_CONVERSIONS[units]["force"]

    ionic_energy_tolerance *= energy_units
    ionic_force_tolerance *= force_units

    line_lst = ["variable thermotime equal {} ".format(n_print)]
    line_lst += _get_thermo()
    if pressure is not None:
        if rotation_matrix is None:
            if structure is None:
                raise ValueError(
                    "No rotation matrix given while trying to convert pressure. "
                    "This is most likely due to no structure being defined."
                )
            else:
                rotation_matrix, structure = _get_rotation_matrix(
                    structure=structure, pressure=pressure
                )
        # force_skewed = False
        pressure = _pressure_to_lammps(
            pressure=pressure, rotation_matrix=rotation_matrix, units=units
        )
        if np.isscalar(pressure):
            str_press = " iso {}".format(float(cast(Any, pressure)))
        else:
            str_press = ""
            for ii, (press, str_axis) in enumerate(
                zip(pressure, ["x", "y", "z", "xy", "xz", "yz"])
            ):
                if press is not None:
                    str_press += " {} {}".format(str_axis, press)
                    # if ii > 2:
                    #     force_skewed = True
            if len(str_press) > 1:
                str_press += " couple none"
        line_lst += [
            "fix ensemble all box/relax" + str_press,
        ]
    line_lst += [
        "min_style " + style,
        "minimize "
        + str(ionic_energy_tolerance)
        + " "
        + str(ionic_force_tolerance)
        + " "
        + str(int(max_iter))
        + " "
        + str(int(max_evaluations)),
    ]
    return line_lst, structure


def calc_static():
    return ["variable thermotime equal 1"] + _get_thermo() + ["run 0"]


def _get_thermo():
    return [
        "thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol",
        "thermo_modify format float %20.15g",
        "thermo ${thermotime}",
    ]


def _is_isotropic_hydrostatic(pressure):
    axial_all_alike = None not in pressure[:3] and np.allclose(
        pressure[:3], pressure[0]
    )
    shear_all_none = all(p is None for p in pressure[3:])
    shear_all_zero = None not in pressure[3:] and np.allclose(pressure[3:], 0)
    return axial_all_alike and (shear_all_none or shear_all_zero)


def _set_initial_velocity(
    temperature,
    seed=80996,
    gaussian=False,
    append_value=False,
    zero_lin_momentum=True,
    zero_rot_momentum=True,
):
    """
    Create initial velocities via velocity all create. More information can be found on LAMMPS website:
    https://lammps.sandia.gov/doc/velocity.html

    Args:
        temperature: (int or float)
        seed: (int) Seed for the initial random number generator
        gaussian: (True/False) Create velocity according to the Gaussian distribution (otherwise uniform)
        append_value: (True/False) Add the velocity values to the current velocities (probably not functional now)
        zero_lin_momentum: (True/False) Cancel the total linear momentum
        zero_rot_momentum: (True/False) Cancel the total angular momentum
    """

    arg = ""
    if gaussian:
        arg = " dist gaussian"
    if append_value:
        arg += " sum yes"
    if not zero_lin_momentum:
        arg += " mom no"
    if not zero_rot_momentum:
        arg += " rot no"
    return "velocity all create " + str(temperature) + " " + str(seed) + arg


def _pressure_to_lammps(pressure, rotation_matrix, units="metal"):
    """
    Convert a singular value, list of values, or combination of values to the appropriate six elements for Lammps
    pxx, pyy, pzz, pxy, pxz, and pyz pressure tensor representation.

    Lammps handles complex cells in a particular way, namely by using an upper triangular cell. This means we may
    need to convert our pressure tensor to a new coordinate frame. We also handle that transformation here.

    In case of a single pressure value, it is again returned as a single pressure value, to be used with the "iso"
    option (i.e., coupled deformation in x, y, and z).

    Finally, we also ensure that the units are converted from pyiron's GPa to whatever Lammps needs.

    Args:
        pressure (float/list/tuple/numpy.ndarray): The pressure(s) to convert.
        rotation_matrix (numpy.ndarray): The 3x3 matrix rotating from the pyiron to Lammps coordinate frame.

    Returns:
        (list): pxx, pyy, pzz, pxy, pxz, and pyz to be passed to Lammps.

        or

        (float): a single, isotropic pressure to be used with the "iso" option
    """

    # in case no rotation matrix is given, assume identity
    if rotation_matrix is None:
        rotation_matrix = np.eye(3)

    # If pressure is a scalar, only unit conversion is needed.
    if np.isscalar(pressure):
        return float(pressure) * LAMMPS_UNIT_CONVERSIONS[units]["pressure"]

    # Normalize pressure to a list of 6 entries of either float or NoneType type.
    if len(pressure) > 6:
        raise ValueError(
            "Pressure can have a maximum of 6 values, (x, y, z, xy, xz, and yz), but got "
            + "{}".format(len(pressure))
        )
    pressure = [float(p) if p is not None else None for p in pressure]
    pressure += (6 - len(pressure)) * [None]

    if all(p is None for p in pressure):
        raise ValueError("Pressure cannot have a length but all be None")

    # If necessary, rotate the pressure tensor to the Lammps coordinate frame.
    # Isotropic, hydrostatic pressures are rotation invariant.
    if not np.isclose(
        np.matrix.trace(rotation_matrix), 3
    ) and not _is_isotropic_hydrostatic(pressure):
        if any(p is None for p in pressure):
            raise ValueError(
                "Cells which are not orthorhombic or an upper-triangular cell are incompatible with Lammps "
                "constant pressure calculations unless the entire pressure tensor is defined. "
                "The reason is that Lammps demands such cells be represented with an "
                "upper-triangular unit cell, thus a rotation between Lammps and pyiron coordinate "
                "frames is required; it is not possible to rotate the pressure tensor if any of "
                "its components is None."
            )
        pxx, pyy, pzz, pxy, pxz, pyz = pressure
        pressure_tensor = np.array([[pxx, pxy, pxz], [pxy, pyy, pyz], [pxz, pyz, pzz]])
        lammps_pressure_tensor = rotation_matrix.T @ pressure_tensor @ rotation_matrix
        pressure = list(lammps_pressure_tensor[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]])

    return [
        (p * LAMMPS_UNIT_CONVERSIONS[units]["pressure"] if p is not None else p)
        for p in pressure
    ]


def _get_rotation_matrix(structure, pressure):
    """

    Args:
        pressure:

    Returns:

    """
    if structure is not None:
        prism = UnfoldingPrism(structure.cell)

        structure = _modify_structure_to_allow_requested_deformation(
            pressure=pressure, structure=structure, prism=prism
        )
        rotation_matrix = prism.R
    else:
        warnings.warn("No structure set, can not validate the simulation cell!")
        rotation_matrix = None
    return rotation_matrix, structure


def _modify_structure_to_allow_requested_deformation(structure, pressure, prism=None):
    """
    Lammps will not allow xy/xz/yz cell deformations in minimization or MD for non-triclinic cells. In case the
    requested pressure for a calculation has these non-diagonal entries, we need to make sure it will run. One way
    to do this is by invoking the lammps `change_box` command, but it is easier to just force our box to to be
    triclinic by adding a very small cell perturbation (in the case where it isn't triclinic already).

    Args:
        pressure (float/int/list/numpy.ndarray/tuple): Between three and six pressures for the x, y, z, xy, xz, and
            yz directions, in that order, or a single value.
    """
    if hasattr(pressure, "__len__"):
        non_diagonal_pressures = np.any([p is not None for p in pressure[3:]])

        if prism is None:
            prism = UnfoldingPrism(structure.cell)

        if non_diagonal_pressures:
            try:
                if not prism.is_skewed():
                    skew_structure = structure.copy()
                    skew_structure.cell[0, 1] += 2 * prism.acc
                    return skew_structure
            except AttributeError:
                warnings.warn(
                    "WARNING: Setting a calculation type which uses pressure before setting the structure risks "
                    + "constraining your cell shape evolution if non-diagonal pressures are used but the structure "
                    + "is not triclinic from the start of the calculation."
                )
    return structure

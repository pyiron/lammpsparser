import lammpsparser._version
from lammpsparser.compatibility.calculate import calc_md, calc_minimize, calc_static
from lammpsparser.compatibility.file import (
    lammps_file_initialization,
    lammps_file_interface_function,
)
from lammpsparser.output import (
    iter_lammps_frames,
    LammpsFrame,
    parse_lammps_output as parse_lammps_output_files,
)
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
    "iter_lammps_frames",
    "LammpsFrame",
    "lammps_file_initialization",
    "lammps_file_interface_function",
    "parse_lammps_output_files",
    "validate_potential_dataframe",
    "write_lammps_structure",
]

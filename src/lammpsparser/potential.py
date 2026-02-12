import os
from pathlib import Path
from typing import Optional, Union

import pandas
from ase.atoms import Atoms

potential_installation = """
Potential installation guide:

1. Check whether iprpy-data is installed. If not, install it using:

`conda install -c conda-forge iprpy-data`

2. Check whether the resource path is set via:

```python
import os
print(os.environ["CONDA_PREFIX"])
```

3. If the resource path is set, you can call the potential using:

```python
from atomistics.calculators import get_potential_by_name


get_potential_by_name(
    potential_name=my_potential,
    resource_path=os.path.join(os.environ["CONDA_PREFIX"], "share", "iprpy"),
)
```

"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd


@dataclass
class Potential:
    """Unified potential representation."""

    year: str
    year_suffix: str
    authors: str
    elements: Set[str]
    repo_type: str
    ipr: Optional[int]
    original: str
    df_index: Optional[int] = None

    @property
    def sort_key(self):
        """Key for sorting - prefer LAMMPS, then higher ipr."""
        return (0 if self.repo_type == "LAMMPS" else 1, -(self.ipr if self.ipr else 0))

    @property
    def family_id(self) -> str:
        """Return author_year[suffix] identifier."""
        year_full = self.year + self.year_suffix if self.year_suffix else self.year
        return f"{self.authors}_{year_full}"


class PotentialDeduplicator:
    """
    Deduplicate interatomic potentials from DataFrame.

    Rules:
    1. Potentials from same author+year+suffix are duplicates
    2. Within LAMMPS: prefer higher ipr
    3. Across repos: prefer LAMMPS over OpenKIM
    4. Only keep potentials containing ALL target_elements
    """

    def __init__(
        self,
        target_elements: Union[str, List[str], Set[str]] = "Ni",
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        target_elements : str, list of str, or set of str
            Element(s) to filter for. If multiple elements provided,
            potentials must contain ALL of them.
            Examples: 'Ni', ['Ni', 'Al'], {'Ni', 'Al', 'Cu'}
        verbose : bool
            Print deduplication details
        """
        # Normalize to set
        if isinstance(target_elements, str):
            self.target_elements = {target_elements}
        elif isinstance(target_elements, (list, tuple)):
            self.target_elements = set(target_elements)
        elif isinstance(target_elements, set):
            self.target_elements = target_elements
        else:
            raise ValueError(
                f"target_elements must be str, list, or set, got {type(target_elements)}"
            )

        self.verbose = verbose
        self.last_duplicates_map = {}
        self.last_stats = {}

    @property
    def target_elements_str(self) -> str:
        """Human-readable string of target elements."""
        if len(self.target_elements) == 1:
            return list(self.target_elements)[0]
        else:
            return "{" + ", ".join(sorted(self.target_elements)) + "}"

    @staticmethod
    def normalize_author(author_str: str) -> str:
        """Extract and normalize primary author surname."""
        parts = re.split(r"[-_]", author_str)
        main = parts[0]

        # Handle camelCase in OpenKIM
        if len(re.findall(r"[A-Z]", main)) > 1:
            camel_parts = re.split(r"(?=[A-Z])", main)
            camel_parts = [p for p in camel_parts if p]
            main = camel_parts[0] if camel_parts else main

        return re.sub(r"[^a-z]", "", main.lower())

    @staticmethod
    def parse_potential_metadata(name: str) -> Optional[Dict]:
        """Parse potential name for metadata (year, author, repo, ipr)."""

        # Try LAMMPS format
        lammps_pattern = (
            r"(\d{4})--([^-]+(?:-[^-]+)*)--([^-]+(?:-[^-]+)*)--LAMMPS--ipr(\d+)"
        )
        match = re.match(lammps_pattern, name)
        if match:
            year, authors, _, ipr = match.groups()
            return {
                "year": year,
                "year_suffix": "",
                "authors": PotentialDeduplicator.normalize_author(authors),
                "repo_type": "LAMMPS",
                "ipr": int(ipr),
            }

        # Try OpenKIM format
        year_match = re.search(r"_(\d{4})([^_]*)", name)
        mo_match = re.search(r"__(MO_|SM_)", name)

        if year_match and mo_match:
            year = year_match.group(1)
            year_suffix = year_match.group(2)

            parts = name.split("_")
            year_idx = None
            for i, part in enumerate(parts):
                if part.startswith(year):
                    year_idx = i
                    break

            if year_idx and year_idx > 0:
                authors = parts[year_idx - 1]
            else:
                authors = ""

            return {
                "year": year,
                "year_suffix": year_suffix,
                "authors": (
                    PotentialDeduplicator.normalize_author(authors) if authors else ""
                ),
                "repo_type": "OpenKIM",
                "ipr": None,
            }

        return None

    def contains_target_elements(self, elements: Set[str]) -> bool:
        """Check if elements set contains ALL target elements."""
        return self.target_elements.issubset(elements)

    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate potentials from DataFrame.

        Parameters
        ----------
        df : DataFrame
            Must have 'Name' and 'Species' columns

        Returns
        -------
        deduplicated_df : DataFrame
            Deduplicated potentials containing all target elements
        """

        if "Name" not in df.columns or "Species" not in df.columns:
            raise ValueError("DataFrame must have 'Name' and 'Species' columns")

        # Parse all potentials
        potentials = []
        unparsed_indices = []
        filtered_out_indices = []

        for idx, row in df.iterrows():
            name = row["Name"]
            species = row["Species"]

            # Convert species to set
            if isinstance(species, list):
                elements = set(species)
            elif isinstance(species, set):
                elements = species
            else:
                elements = set()

            # Check if ALL target elements are present
            if not self.contains_target_elements(elements):
                filtered_out_indices.append(idx)
                continue

            # Parse metadata
            metadata = self.parse_potential_metadata(name)

            if metadata:
                pot = Potential(
                    year=metadata["year"],
                    year_suffix=metadata["year_suffix"],
                    authors=metadata["authors"],
                    elements=elements,
                    repo_type=metadata["repo_type"],
                    ipr=metadata["ipr"],
                    original=name,
                    df_index=idx,
                )
                potentials.append(pot)
            else:
                unparsed_indices.append(idx)

        # Store stats
        self.last_stats = {
            "total": len(df),
            "filtered_out": len(filtered_out_indices),
            "unparsed": len(unparsed_indices),
            "valid": len(potentials),
        }

        if self.verbose:
            print(f"Total potentials: {self.last_stats['total']}")
            print(f"Target elements: {self.target_elements_str}")
            print(
                f"Filtered out (missing target elements): {self.last_stats['filtered_out']}"
            )
            print(f"Unparsed: {self.last_stats['unparsed']}")
            print(f"Valid for deduplication: {self.last_stats['valid']}")

        # Group by (year+suffix, author)
        groups = defaultdict(list)
        for pot in potentials:
            year_full = pot.year + pot.year_suffix if pot.year_suffix else pot.year
            key = (year_full, pot.authors)
            groups[key].append(pot)

        # Keep only the best from each group
        kept_indices = []
        self.last_duplicates_map = {}

        for (year_full, author), group in sorted(groups.items()):
            if len(group) == 1:
                kept_indices.append(group[0].df_index)
                continue

            # Sort by preference: LAMMPS first, then highest ipr
            group.sort(key=lambda p: p.sort_key)

            best = group[0]
            rest = group[1:]

            kept_indices.append(best.df_index)
            self.last_duplicates_map[best.original] = [p.original for p in rest]

            if self.verbose:
                print(f"\nGroup: {year_full} - {author}")
                print(f"  Kept: {best.original}")
                for dup in rest:
                    print(f"  Removed: {dup.original}")

        # Add back unparsed items
        kept_indices.extend(unparsed_indices)

        # Update stats
        self.last_stats["kept"] = len(kept_indices)
        self.last_stats["removed_duplicates"] = sum(
            len(v) for v in self.last_duplicates_map.values()
        )

        if self.verbose:
            print(f"\nFinal count: {self.last_stats['kept']}")
            print(f"Duplicates removed: {self.last_stats['removed_duplicates']}")

        # Return deduplicated DataFrame
        return df.loc[kept_indices].copy()

    def get_duplicates(self) -> Dict[str, List[str]]:
        """Return the duplicates map from last deduplication."""
        return self.last_duplicates_map.copy()

    def get_stats(self) -> Dict[str, int]:
        """Return statistics from last deduplication."""
        return self.last_stats.copy()

    def get_family_id(self, potential_name: str) -> Optional[str]:
        """
        Get the family label (author_year[suffix]) for a potential.

        Returns normalized label like 'foiles_1986' or 'adams_1989Universal6'.
        """
        metadata = self.parse_potential_metadata(potential_name)
        if metadata:
            year_full = (
                metadata["year"] + metadata["year_suffix"]
                if metadata["year_suffix"]
                else metadata["year"]
            )
            return f"{metadata['authors']}_{year_full}"
        return None

    def analyze_families(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze potential families in the DataFrame.

        Returns a summary DataFrame with family counts and repo types.
        Only includes potentials with all target elements.
        """
        families = defaultdict(lambda: {"count": 0, "repos": set(), "names": []})

        for _, row in df.iterrows():
            name = row["Name"]
            species = row["Species"]

            # Check target elements
            elements = set(species) if isinstance(species, list) else species
            if not self.contains_target_elements(elements):
                continue

            family_id = self.get_family_id(name)
            if family_id:
                metadata = self.parse_potential_metadata(name)
                families[family_id]["count"] += 1
                families[family_id]["repos"].add(metadata["repo_type"])
                families[family_id]["names"].append(name)

        # Convert to DataFrame
        summary = []
        for family_id, info in sorted(families.items()):
            summary.append(
                {
                    "family": family_id,
                    "count": info["count"],
                    "repos": ", ".join(sorted(info["repos"])),
                    "has_duplicates": info["count"] > 1,
                }
            )

        return pd.DataFrame(summary)

    def filter_by_elements(
        self,
        df: pd.DataFrame,
        target_elements: Optional[Union[str, List[str], Set[str]]] = None,
    ) -> pd.DataFrame:
        """
        Filter DataFrame to only potentials containing specified elements.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame with 'Species' column
        target_elements : str, list, set, or None
            Elements to filter for. If None, uses self.target_elements

        Returns
        -------
        filtered_df : DataFrame
            Filtered to potentials with all target elements
        """
        if target_elements is not None:
            # Temporarily change target elements
            if isinstance(target_elements, str):
                target_set = {target_elements}
            elif isinstance(target_elements, (list, tuple)):
                target_set = set(target_elements)
            else:
                target_set = target_elements
        else:
            target_set = self.target_elements

        filtered_indices = []
        for idx, row in df.iterrows():
            species = row["Species"]
            elements = set(species) if isinstance(species, list) else species
            if target_set.issubset(elements):
                filtered_indices.append(idx)

        return df.loc[filtered_indices].copy()


class PotentialAbstract:
    """
    The PotentialAbstract class loads a list of available potentials and sorts them. Afterwards the potentials can be
    accessed through:
        PotentialAbstract.<Element>.<Element> or PotentialAbstract.find_potentials_set({<Element>, <Element>}

    Args:
        potential_df:
        default_df:
        selected_atoms:
    """

    def __init__(
        self,
        potential_df: pandas.DataFrame,
        default_df: pandas.DataFrame = None,
        selected_atoms: list[str] = None,
    ):
        self._potential_df = potential_df
        self._default_df = default_df
        if selected_atoms is not None:
            self._selected_atoms = selected_atoms
        else:
            self._selected_atoms = []

    def find(self, element: Union[set[str], list[str], str]) -> pandas.DataFrame:
        """
        Find the potentials

        Args:
            element (set, str): element or set of elements for which you want the possible LAMMPS potentials

        Returns:
            list: of possible potentials for the element or the combination of elements

        """
        if isinstance(element, list):
            element = set(element)
        elif isinstance(element, str):
            element = {element}
        elif not isinstance(element, set):
            raise TypeError("Only, str, list and set supported!")
        return self._potential_df[
            [
                bool(set(element).issubset(species))
                for species in self._potential_df["Species"].values
            ]
        ]

    def find_by_name(self, potential_name: str) -> pandas.DataFrame:
        mask = self._potential_df["Name"] == potential_name
        if not mask.any():
            raise ValueError(f"Potential '{potential_name}' not found in database.")
        return self._potential_df[mask]

    def list(self) -> pandas.DataFrame:
        """
        List the available potentials

        Returns:
            list: of possible potentials for the element or the combination of elements
        """
        return self._potential_df

    def __getattr__(self, item):
        return self[item]

    def __getitem__(self, item):
        potential_df = self.find(element=item)
        selected_atoms = self._selected_atoms + [item]
        return PotentialAbstract(
            potential_df=potential_df,
            default_df=self._default_df,
            selected_atoms=selected_atoms,
        )

    def __str__(self) -> str:
        return str(self.list())

    @staticmethod
    def _get_potential_df(file_name_lst, resource_path):
        """

        Args:
            file_name_lst (set):
            resource_path (str):

        Returns:
            pandas.DataFrame:
        """
        for path, _folder_lst, file_lst in os.walk(resource_path):
            for periodic_table_file_name in file_name_lst:
                if (
                    periodic_table_file_name in file_lst
                    and periodic_table_file_name.endswith(".csv")
                ):
                    return pandas.read_csv(
                        os.path.join(path, periodic_table_file_name),
                        index_col=0,
                        converters={
                            "Species": (
                                lambda x: x.replace("'", "").strip("[]").split(", ")
                            ),
                            "Config": (
                                lambda x: (
                                    x.replace("'", "")
                                    .replace("\\n", "\n")
                                    .strip("[]")
                                    .split(", ")
                                )
                            ),
                            "Filename": (
                                lambda x: x.replace("'", "").strip("[]").split(", ")
                            ),
                        },
                    )
        raise ValueError(
            "Was not able to locate the potential files." + potential_installation
        )


class LammpsPotentialFile(PotentialAbstract):
    """
    The Potential class is derived from the PotentialAbstract class, but instead of loading the potentials from a list,
    the potentials are loaded from a file.

    Args:
        potential_df:
        default_df:
        selected_atoms:
    """

    def __init__(
        self,
        potential_df=None,
        default_df=None,
        selected_atoms=None,
        resource_path=None,
    ):
        if potential_df is None:
            potential_df = self._get_potential_df(
                file_name_lst={"potentials_lammps.csv"},
                resource_path=resource_path,
            )
        super().__init__(
            potential_df=potential_df,
            default_df=default_df,
            selected_atoms=selected_atoms,
        )
        self._resource_path = resource_path

    def default(self):
        if self._default_df is not None:
            atoms_str = "_".join(sorted(self._selected_atoms))
            return self._default_df[
                (self._default_df["Name"] == self._default_df.loc[atoms_str].values[0])
            ]
        return None

    def find_default(
        self, element: Union[set[str], list[str], str]
    ) -> pandas.DataFrame:
        """
        Find the potentials

        Args:
            element (set, str): element or set of elements for which you want the possible LAMMPS potentials
            path (bool): choose whether to return the full path to the potential or just the potential name

        Returns:
            list: of possible potentials for the element or the combination of elements

        """
        if isinstance(element, list):
            element = set(element)
        elif isinstance(element, str):
            element = {element}
        elif not isinstance(element, set):
            raise TypeError("Only, str, list and set supported!")
        element_lst = list(element)
        if self._default_df is not None:
            merged_lst = list(set(self._selected_atoms + element_lst))
            atoms_str = "_".join(sorted(merged_lst))
            return self._default_df[
                (self._default_df["Name"] == self._default_df.loc[atoms_str].values[0])
            ]
        return None

    def __getitem__(self, item):
        potential_df = self.find(element=item)
        selected_atoms = self._selected_atoms + [item]
        return LammpsPotentialFile(
            potential_df=potential_df,
            default_df=self._default_df,
            selected_atoms=selected_atoms,
            resource_path=self._resource_path,
        )


class PotentialAvailable:
    def __init__(self, list_of_potentials):
        self._list_of_potentials = {
            "pot_" + v.replace("-", "_").replace(".", "_"): v
            for v in list_of_potentials
        }

    def __getattr__(self, name):
        if name in self._list_of_potentials:
            return self._list_of_potentials[name]
        else:
            raise AttributeError

    def __dir__(self):
        return list(self._list_of_potentials.keys())

    def __repr__(self):
        return str(dir(self))


def find_potential_file_base(path, resource_path_lst, rel_path):
    if path is not None:
        for resource_path in resource_path_lst:
            path_direct = os.path.join(resource_path, path)
            path_indirect = os.path.join(resource_path, rel_path, path)
            if os.path.exists(path_direct):
                return path_direct
            elif os.path.exists(path_indirect):
                return path_indirect
    raise ValueError(
        "Either the filename or the functional has to be defined.",
        path,
        resource_path_lst,
    )


def view_potentials(structure: Atoms, resource_path: str) -> pandas.DataFrame:
    """
    List all interatomic potentials for the given atomistic structure including all potential parameters.

    To quickly get only the names of the potentials you can use `list_potentials()` instead.

    Args:
        structure (Atoms): The structure for which to get potentials.
        resource_path (str): Path to the "lammps/potentials_lammps.csv" file

    Returns:
        pandas.Dataframe: Dataframe including all potential parameters.
    """
    list_of_elements = set(structure.get_chemical_symbols())
    raw_df = LammpsPotentialFile(resource_path=resource_path).find(list_of_elements)

    dedup = PotentialDeduplicator(target_elements=list_of_elements, verbose=True)
    clean_df = dedup.deduplicate(raw_df)

    return clean_df


def convert_path_to_abs_posix(path: str) -> str:
    """
    Convert path to an absolute POSIX path

    Args:
        path (str): input path.

    Returns:
        str: absolute path in POSIX format
    """
    return (
        Path(path.strip())
        .expanduser()
        .resolve()
        .absolute()
        .as_posix()
        .replace("\\", "/")
    )


def update_potential_paths(
    df_pot: pandas.DataFrame, resource_path: str
) -> pandas.DataFrame:
    config_lst = []
    for row in df_pot.itertuples():
        potential_file_lst = row.Filename
        potential_file_path_lst = [
            os.path.join(resource_path, f) for f in potential_file_lst
        ]
        potential_dict = {os.path.basename(f): f for f in potential_file_path_lst}
        potential_commands = []
        for line in row.Config:
            line = line.replace("\n", "")
            for key, value in potential_dict.items():
                line = line.replace(key, value)
            potential_commands.append(line)
        config_lst.append(potential_commands)
    df_pot["Config"] = config_lst
    return df_pot


def get_resource_path_from_conda(
    env_variables: tuple[str] = ("CONDA_PREFIX", "CONDA_DIR"),
) -> str:
    env = os.environ
    for conda_var in env_variables:
        if conda_var in env:
            resource_path = os.path.join(env[conda_var], "share", "iprpy")
            if os.path.exists(resource_path):
                return resource_path
    raise ValueError("No resource_path found" + potential_installation)


def get_potential_dataframe(structure: Atoms, resource_path=None):
    if resource_path is None:
        resource_path = get_resource_path_from_conda()
    return update_potential_paths(
        df_pot=view_potentials(structure=structure, resource_path=resource_path),
        resource_path=resource_path,
    )


def get_potential_by_name(potential_name: str, resource_path: Optional[str] = None):
    if resource_path is None:
        resource_path = get_resource_path_from_conda()
    df = LammpsPotentialFile(resource_path=resource_path).list()
    return update_potential_paths(
        df_pot=df[df.Name == potential_name], resource_path=resource_path
    ).iloc[0]


def validate_potential_dataframe(
    potential_dataframe: pandas.DataFrame,
) -> pandas.DataFrame:
    if isinstance(potential_dataframe, pandas.Series):
        return potential_dataframe
    elif isinstance(potential_dataframe, pandas.DataFrame):
        if len(potential_dataframe) == 1:
            return potential_dataframe.iloc[0]
        elif len(potential_dataframe) == 0:
            raise ValueError(
                "The potential_dataframe is an empty pandas.DataFrame:",
                potential_dataframe,
            )
        else:
            raise ValueError(
                "The potential_dataframe contains more than one interatomic potential, please select one:",
                potential_dataframe,
            )
    else:
        raise TypeError(
            "The potential_dataframe should be a pandas.DataFrame or pandas.Series, but instead it is of type:",
            type(potential_dataframe),
        )

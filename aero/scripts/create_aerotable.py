"""
Purpose of this script is to load CSV files, interpolate to common values
of the independent variables, and write to an HDF5 file.
"""

from dataclasses import dataclass
import pathlib
from typing import List

import h5py
import matplotlib.pyplot as plt
import numpy as np

coef_name_symb = {
        "ca": r"$C_A$",
        "cn": r"$C_N$",
        "clm": r"$C_{lm}$",
        }

@dataclass
class AeroTable:
    """Simple dataclass to store/manipulate/visualize an aero table.
    """

    mach: np.ndarray
    alpha_tot: np.ndarray
    coef: np.ndarray
    name: str

    def plot(self):
        """Method for plotting an aero table.
        """

        fig = plt.figure()
        for i, alpha_t in enumerate(self.alpha_tot):
            alpha_t_str = r"$\alpha_T=" + f"{alpha_t:4.1f}" + r"^\circ$"
            plt.plot(self.mach, self.coef[i], label=alpha_t_str)

        plt.xlabel(r"$M_\infty$")
        plt.ylabel(coef_name_symb[self.name])

        plt.gca().minorticks_on()

        plt.legend()

        return fig

    def to_h5(self, filepath: str):
        """Writes the aero table to an HDF5 file.

        Args:
            filepath (str): path to hdf5 file
        """

        with h5py.File(filepath, "a") as h5_file:
            group = h5_file.create_group(self.name)
            group.create_dataset("mach", data=self.mach)
            group.create_dataset("alpha_tot", data=self.alpha_tot)
            group.create_dataset("coef", data=self.coef)


def create_common_vec(vectors: List[np.ndarray]) -> np.ndarray:
    """Creates a common indep var vector from a list of vectors.

    Args:
        vectors (List[np.ndarray]): list of input indep var vectors

    Returns:
        np.ndarray: an array which contains the unique set of indep vars
    """
    combined = np.concatenate(vectors)
    return np.unique(combined)


def extract_alpha_tot(filename: str) -> float:
    """Extracts alpha_total from the given filename.

    Args:
        filename (str): CSV filename which contains alpha_total

    Returns:
        float: alpha_total in degrees
    """
    return float(filename.strip(".csv").split("_")[-1].replace("p", "."))

def extract_coef_name(filename: str) -> str:
    """Extracts coefficient name from the file name.

    Args:
        filename (str): CSV filename which contains coefficient name

    Returns:
        str: coefficient name (e.g., ca, cn, clm)
    """
    return filename.split("/")[-1].split("_")[0]


def generate_table(files: List[str]) -> AeroTable:
    """Generates an aero table from a list of CSV files.

    Args:
        files (List[str]): list of CSV files

    Returns:
        AeroTable: aerotable object
    """

    # Load data.
    data = [np.genfromtxt(f, skip_header=1, delimiter=", ") for f in files]

    # Extract alpha_tot from the filename.
    alpha_tot = np.array(list(map(extract_alpha_tot, files)))

    # Extract coef name from filename
    coef_name = extract_coef_name(files[0])

    # Find common mach vector
    mach_vectors = [d[:, 0] for d in data]
    mach = create_common_vec(mach_vectors)

    # Interpolate each array to the common mach.
    coef_common = [np.interp(mach, d[:,0], d[:,1]) for d in data]

    # Concatenate coefficient array into 2D array.
    coef = np.stack(coef_common, axis=0)

    return AeroTable(mach, alpha_tot, coef, coef_name)


# Build path to data.
script_dir = pathlib.Path(__file__).parent.absolute()
data_dir = script_dir.parent / "data"

# Find all CSV files.
csv_files = list(data_dir.glob("c*.csv"))
ca_files  = sorted([str(f) for f in csv_files if f.name.startswith("ca")])
cn_files  = sorted([str(f) for f in csv_files if f.name.startswith("cn")])
clm_files = sorted([str(f) for f in csv_files if f.name.startswith("clm")])

# Build aero tables.
ca_table = generate_table(ca_files)
cn_table = generate_table(cn_files)
clm_table = generate_table(clm_files)
tables = [ca_table, cn_table, clm_table]

# Write.
h5_file_path = data_dir / "muses-c.h5"
h5_file_path.unlink(missing_ok=True)
for table in tables:
    table.to_h5(str(h5_file_path))

# Plot.
for table in tables:
    table.plot()

plt.show()

import os
from pathlib import Path

import numpy as np


def element_mapping():
    mapping = {
        "C": 6,
        "H": 1,
        "B": 5,
        "Br": 35,
        "Cl": 17,
        "D": 0,
        "F": 9,
        "I": 53,
        "N": 7,
        "O": 8,
        "P": 15,
        "S": 16,
        "Se": 34,
        "Si": 14,
    }
    reverse_mapping = dict([reversed(pair) for pair in mapping.items()])
    mapping.update(reverse_mapping)
    return mapping


def parse_xyz(f: str):
    """
    :param f: str. Path to .xyz file.
    :return: List[tuple]. List of atomic coordinates. Tuple indices: (atomic_num, x, y, z)
    """
    with open(f, "r") as file:
        lines = file.readlines()[2:]
    coords = []
    for l in lines:
        element, x, y, z = l.split()
        if not element.isdigit():
            element = element_mapping()[element]
        else:
            element = int(element)
        coords.append((element, float(x), float(y), float(z)))
    return coords


def xyz_to_ndarray(path):
    """Single .xyz file to cartesian, atomic_nums arrays"""
    coordinates = parse_xyz(path)
    coordinate_array = np.array(coordinates)
    cartesians = coordinate_array[:, 1:]
    atomic_nums = coordinate_array[:, :1].astype("int").reshape((-1,))
    return cartesians, atomic_nums


def ndarrays_to_xyz(c, z, path, message: "str" = None):
    """

    :param c: cartesian array of shape (points, 3)
    :param z: atomic_nums arrray of shape (points,)
    :param path: path to .xyz file
    :param message: str to add to message portion of .xyz. Defaults to None
    :return:
    """
    os.makedirs(Path(path).parent, exist_ok=True)
    message = message or ""
    first_dummy_atom = np.where(z == 0)[0][0]
    coordinates = np.concatenate([z.reshape((-1, 1)), c], axis=-1)  # (atoms, 4)
    text = [
        "    ".join([element_mapping()[c[0]], str(c[1]), str(c[2]), str(c[3]),])
        for c in coordinates[:first_dummy_atom]
    ]
    with open(path, "w") as file:
        file.write(str(len(z[:first_dummy_atom])) + "\n")
        file.write(f"{message}\n")
        file.write("\n".join(text))
        file.write("\n")

import sys
from pathlib import Path

import h5py
import numpy as np


def split_xyz_file(path):
    with open(path) as file:
        lines = file.readlines()
    separated_cartesian_lines = consume_lines(lines)
    cartesians_collection = []
    atomic_nums_collection = []
    for cartesian_lines in separated_cartesian_lines:
        coordinates = convert_xyz_lines_to_list_of_numbers(cartesian_lines[2:])
        cartesians, atomic_nums = coordinate_to_array(coordinates)
        cartesians_collection.append(cartesians)
        atomic_nums_collection.append(atomic_nums)
    return cartesians_collection, atomic_nums_collection


def consume_lines(lines):
    def c(l: list):
        if len(l) > 0:
            i = int(l[0])
            separated_cartesians.append(l[0 : i + 2])
            l[0 : i + 2] = []
            c(l)

    separated_cartesians = []
    sys.setrecursionlimit(10000)
    c(lines)
    sys.setrecursionlimit(1500)
    return separated_cartesians


def convert_xyz_lines_to_list_of_numbers(lines):
    coords = []
    for l in lines:
        element, x, y, z = l.split()
        if not element.isdigit():
            element = element_mapping()[element]
        else:
            element = int(element)
        coords.append((element, float(x), float(y), float(z)))
    return coords


def coordinate_to_array(coordinates):
    coordinate_array = np.array(coordinates)
    cartesians = coordinate_array[:, 1:]
    atomic_nums = coordinate_array[:, :1].astype("int").reshape((-1, 1))
    return cartesians, atomic_nums


def pad_array(arr, atom_padding, value=np.nan):
    return np.pad(
        arr,
        ((0, atom_padding - arr.shape[0]), (0, 0)),
        mode="constant",
        constant_values=value,
    )


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


def parse():
    names = ["p_train", "r_train", "ts_train"]
    paths = [Path(f"./{n}.xyz") for n in names]

    arrays = {}
    for path in paths:
        cartesians_list, atomic_nums_list = split_xyz_file(path)
        for i, (c, a) in enumerate(zip(cartesians_list, atomic_nums_list)):
            cartesians_list[i] = pad_array(c, 21)
            atomic_nums_list[i] = pad_array(a, 21, value=0)
        arrays[path.stem] = [
            np.array(arr) for arr in (cartesians_list, atomic_nums_list)
        ]

    with h5py.File("./isomerization_dataset.hd5f", "w") as file:
        for name, (cartesians, atomic_nums) in arrays.items():
            file.create_dataset(f"{name}/cartesians", data=cartesians)
            file.create_dataset(
                f"{name}/atomic_nums",
                data=np.nan_to_num(np.squeeze(atomic_nums, axis=-1), nan=0),
            )


if __name__ == "__main__":
    parse()

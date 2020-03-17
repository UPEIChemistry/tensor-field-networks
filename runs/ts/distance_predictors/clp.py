import argparse
import pickle

import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description=(
        'Script for obtaining TS critical lengths from input of reactant/product lengths'
    ))
    parser.add_argument('nucleophile')
    parser.add_argument('lg')
    parser.add_argument('reactant_distances', nargs=2)
    parser.add_argument('product_distances', nargs=2)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    vector = np.array([
        args.nucleophile,
        args.lg,
        args.reactant_distances[0],
        args.reactant_distances[1],
        args.product_distances[0],
        args.product_distances[1]
    ]).reshape([1, -1])

    with open('model0.pickle', 'rb') as file:
        model0 = pickle.load(file)
    with open('model1.pickle', 'rb') as file:
        model1 = pickle.load(file)
    print('TS Nucleophile to Center Distance: {}'.format(model0.predict(vector)))
    print('TS Center to LG Distance: {}'.format(model1.predict(vector)))

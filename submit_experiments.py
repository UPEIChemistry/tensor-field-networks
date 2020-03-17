import argparse
import getpass
import socket
import time
from warnings import warn
from pathlib import Path
from typing import List
import os

TIME = '24:00:00'
MEM = '16G'
NUM_GPUS = 1


def get_args():
    parser = argparse.ArgumentParser(description='Script for running experiments on tater.')
    parser.add_argument('experiments', nargs='+',
                        help='Path to one or more experiment file(s) or the directories '
                             'containing them')
    parser.add_argument('--local', default=False, action='store_true',
                        help='Flag to tell script NOT to add SLURM commands. NOT IMPLEMENTED.')
    parser.add_argument('--time', default=TIME,
                        help='Wall-time for each SLURM job. Defaults to {}'.format(TIME))
    parser.add_argument('--mem', default=MEM,
                        help='Memory required for each SLURM job. Defaults to {}'.format(MEM))
    parser.add_argument('--num_gpus', default=NUM_GPUS,
                        help=('Number of gpus for each SLURM job. Defaults to {}'.format(NUM_GPUS)))
    return parser.parse_args()


def main(experiments: List[Path], **kwargs):

    paths = sanitize_paths([e.resolve() for e in experiments])

    if not kwargs.pop('local', False):
        run_slurm_job(paths, **kwargs)
    else:
        run_slurm_job(paths, **kwargs)


def sanitize_paths(experiments):
    paths = []
    for exp in experiments:
        if exp.is_dir():
            paths.extend([
                Path(root) / Path(file) for root, dirs, files in os.walk(exp.__str__())
                for file in files if 'exp' in file and '.py' in file
            ])
        elif 'exp' in exp.stem.lower() and exp.suffix == '.py':
            paths.append(exp)
        else:
            warn('incompatible experiment path: `{}` skipping...')
            continue
    if not paths:
        raise ValueError('No compatible experiment paths passed, please ensure script name '
                         'contains "experiment", or is a path to a directory which contains a '
                         'script with "experiment" in it\'s name.')
    return paths


def run_slurm_job(paths, **kwargs):
    import subprocess
    slurm_params_lines = [
        '#!/bin/bash\n',
        '#SBATCH --export=ALL\n',
        '#SBATCH --time={}\n'.format(kwargs.pop('time', TIME)),
        '#SBATCH --mem={}\n'.format(kwargs.pop('mem', MEM)),
        '#SBATCH --gres=gpu:{}\n'.format(str(kwargs.pop('num_gpus', NUM_GPUS)))
    ]
    for i, path in enumerate(paths):
        if path.is_dir():
            d = path
        else:
            d = path.parent
        submission_path = d/'{}.sh'.format(path.stem)
        with open(submission_path, 'w') as file:
            file.write(''.join(line for line in slurm_params_lines))
            file.write('#SBATCH --chdir={}\n'.format(path.parent.__str__()))
            file.write('#SBATCH --output=output.out\n\n')
            if socket.gethostname() == 'tater':
                file.write('source /mnt/fast-data/common/miniconda3/bin/activate mlenv\n')
            else:
                file.write('module load python/3.7\n')
                file.write('. ~/dev/environments/mlenv/bin/activate\n')
            file.write('python {}\n'.format(path))
        subprocess.run(
            ['sbatch', str(submission_path)],
            check=True
        )
        time.sleep(kwargs.pop('sleep_time', 3))
        if i == len(paths):
            break
    subprocess.run(['squeue', '-u', getpass.getuser()])


if __name__ == '__main__':
    args = get_args()
    main(
        [Path(p) for p in args.experiments],
        local=args.local,
        time=args.time,
        mem=args.mem,
        num_gpus=args.num_gpus
    )

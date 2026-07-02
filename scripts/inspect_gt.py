"""
Inspect ground truth annotations: how many nodes, edges, divisions per sample.
This tells us the true expected cell density and annotation sparsity.
Usage: python scripts/inspect_gt.py --data_dir data/train/train
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.geff_reader import load_geff
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    geff_dirs = sorted(data_dir.glob('*.geff'))

    print(f'{"Sample":>24} {"nodes":>7} {"edges":>7} {"divs":>6} {"t_min":>6} {"t_max":>6} {"nodes/t":>8}')
    print('-' * 75)

    for geff_path in geff_dirs:
        G = load_geff(str(geff_path))
        nodes = G.number_of_nodes()
        edges = G.number_of_edges()
        divs = sum(1 for n in G.nodes if G.out_degree(n) >= 2)

        t_vals = [d['t'] for _, d in G.nodes(data=True)]
        t_min, t_max = min(t_vals), max(t_vals)
        t_counts = Counter(t_vals)
        avg_nodes_per_t = nodes / len(t_counts)

        print(f'{geff_path.stem:>24} {nodes:>7} {edges:>7} {divs:>6} {t_min:>6} {t_max:>6} {avg_nodes_per_t:>8.1f}')


if __name__ == '__main__':
    main()

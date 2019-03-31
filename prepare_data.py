"""
usage:
    prepare_data.py [options] <inputfile>

options:
    -n <n_stores>, --n_stores=<n_stores>
        number of stores to include (all data if unspecified)
    -o <filename>, --outfile=<filename>
        name of output file [default: prep.csv]
"""
from docopt import docopt
import pandas as pd
from deepfc import data

args = docopt(__doc__)

reduced = data.prepare_data(
    pd.read_csv(args['<inputfile>']),
    int(args['--n_stores']) if args['--n_stores'] else None)
print('Done')
print(reduced.head())
print('Writing output file...')
reduced.to_csv(args['--outfile'], index=False)
print('Done')

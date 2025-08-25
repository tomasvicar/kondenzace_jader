

from glob import glob
import os

data_path = r"D:\kondenzace_jader\data\Chromatin Architecture Analyses Python\Mikroskop horní\zz_Original data files"
results_path = data_path + '_results'

to_remove = [
    '*features.json',
    '*fourier.png',
    '*tpd.png',
    '*radial.png',
    '*fractal.png'
]


for rem in to_remove:
    fnames = glob(results_path + '/**/' + rem, recursive=True)
    for fname in fnames:
        os.remove(fname)

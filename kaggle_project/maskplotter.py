import pillow
import numpy as np
import pandas as pd

def load_filepath(self, filepath):
        X = np.load(filepath)
        y_raw = str(self.csv.loc[filepath.split(os.sep)[-1][:-4], 1])
        if len(y_raw.split()) != 1:
            Y = np.expand_dims(rle2mask(y_raw, *self.dim).T, axis=2)
        else:
            Y = np.zeros((*self.dim, 1))
        return Y

mask = load_file
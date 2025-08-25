from glob import glob
import json
import os
import pandas as pd

data_path = r"D:\kondenzace_jader\data\Chromatin Architecture Analyses Python\Mikroskop horní\zz_Original data files  reduced"
results_path = data_path + '_results'


fnames = glob(results_path + '/**/*_features.json', recursive=True)



data_all = []
for fname_ind, fname in enumerate(fnames):

    data = dict()

    with open(fname, 'r') as f:
        features = json.load(f)

    # {'fractal_dim': 1.6730451998087634,
    # 'fourier_hff': 0.7945160743381868,
    # 'tpd': 0.1046119235095613,
    # 'radial_in_out_fraction': 1.2691911834100145,
    # 'radial_max_position': 0.045}

    # save_name_fractal = fname.replace('_features.json', '_fractal.png')
    # if os.path.exists(save_name_fractal):
    #     data['fractal_dim'] = features['fractal_dim']



    # save_name_hff = fname.replace('_features.json', '_fourier.png')
    # if os.path.exists(save_name_hff):
    #     data['fourier_hff'] = features['fourier_hff']

    # save_name_tpd = fname.replace('_features.json', '_tpd.png')
    # if os.path.exists(save_name_tpd):
    #     data['tpd'] = features['tpd']


    # save_name_radial = fname.replace('_features.json', '_radial.png')
    # if os.path.exists(save_name_radial):
    #     data['radial_in_out_fraction'] = features['radial_in_out_fraction']
    #     data['radial_max_position'] = features['radial_max_position']

    for key, value in features.items():
        data[key] = value


    data['file_name'] = fname

    data_all.append(data)

df = pd.DataFrame(data_all)
df.to_excel(results_path + '/results_reduced2.xlsx', index=False)





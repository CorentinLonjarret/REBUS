import time
import pandas as pd
import os
import yaml
import datetime
from importlib.machinery import SourceFileLoader
import corpus

# ---- 1 - Chargement des données ---- #
dict_datasets_path = {
    'Epinions': [os.path.join('..', '01-Data', 'Epinions.txt'), 5, 5, [1, 2], [3]],
    'foursq': [os.path.join('..', '01-Data', 'foursq.txt'), 5, 5, [2, 4, 6], [2, 5, 10, 15]],
    'adressa_one_week': [os.path.join('..', '01-Data', 'adressa_one_week.csv'), 5, 5, [2, 4, 6], [2, 5, 10, 15]],
    'visiativ-prep': [os.path.join('..', '01-Data', 'visiativ-prep.txt'), 5, 5, [1, 2, 4, 6], [3, 6, 10]],
    'ratings_Automotive': [os.path.join('..', '01-Data', 'ratings_Automotive.txt'), 5, 5, [2, 4], [2, 5, 8]],
    'ratings_Office_Products': [os.path.join('..', '01-Data', 'ratings_Office_Products.csv'), 5, 5, [2, 4], [2, 5, 8]],
    'ratings_Video_Games': [os.path.join('..', '01-Data', 'ratings_Video_Games.csv'), 5, 5, [2, 4, 6], [2, 5, 10]],
    'ML1M-atmost-5': [os.path.join('..', '01-Data', 'ML1M-atmost-5.txt'), 0, 0, [1, 2], [3]],
    'ML1M-atmost-10': [os.path.join('..', '01-Data', 'ML1M-atmost-10.txt'), 0, 0, [2, 4], [3, 8]],
    'ML1M-atmost-20': [os.path.join('..', '01-Data', 'ML1M-atmost-20.txt'), 0, 0, [2, 4, 6], [3, 8]],
    'ML1M-atmost-30': [os.path.join('..', '01-Data', 'ML1M-atmost-30.txt'), 0, 0, [2, 4, 6], [3, 6, 10]],
    'ML1M-atmost-50': [os.path.join('..', '01-Data', 'ML1M-atmost-50.txt'), 0, 0, [2, 4, 6], [3, 6, 10]]
}

df_datasets_path = pd.DataFrame.from_dict(dict_datasets_path, orient='index')
df_datasets_path.reset_index(level=0, inplace=True)
df_datasets_path.columns = ['dataset', 'path', 'userMin', 'itemMin', 'minCount', 'L']
df_datasets_path = df_datasets_path.sort_values(by=['dataset'], ascending=[1])
df_datasets_path.columns
df_datasets_path['corpus'] = None
for dataset in df_datasets_path['dataset']:
    print(dataset)
    corp = corpus.Corpus()
    corp.loadData(pathFile=df_datasets_path.loc[df_datasets_path['dataset'] == dataset, 'path'].iloc[0], userMin=df_datasets_path.loc[df_datasets_path['dataset'] == dataset, 'userMin'].iloc[0], itemMin=df_datasets_path.loc[df_datasets_path['dataset']
                                                                                                                                                                                                                               == dataset, 'itemMin'].iloc[0], suffix_file_data=df_datasets_path.loc[df_datasets_path['dataset'] == dataset, 'dataset'].iloc[0], name_file_data=df_datasets_path.loc[df_datasets_path['dataset'] == dataset, 'dataset'].iloc[0])
    df_datasets_path.loc[df_datasets_path['dataset'] == dataset, 'corpus'] = corp
    del corp

for dataset in df_datasets_path['dataset']:
    print("Dataset : ", dataset,
          " nUser :", df_datasets_path.loc[df_datasets_path['dataset'] == dataset, 'corpus'].iloc[0].nUsers,
          " nitems :", df_datasets_path.loc[df_datasets_path['dataset'] == dataset, 'corpus'].iloc[0].nItems)

for dataset in df_datasets_path['dataset']:
    print(dataset)
    corp = df_datasets_path.loc[df_datasets_path['dataset'] == dataset, 'corpus'].iloc[0]
    corp.splitDataTrainValTest()
    print(corp.pos_per_user[0])
    corp.create_fsub(df_datasets_path.loc[df_datasets_path['dataset'] == dataset, 'dataset'].iloc[0],
                     tabMinCount=df_datasets_path.loc[df_datasets_path['dataset'] == dataset, 'minCount'].iloc[0],
                     tabL=df_datasets_path.loc[df_datasets_path['dataset'] == dataset, 'L'].iloc[0])

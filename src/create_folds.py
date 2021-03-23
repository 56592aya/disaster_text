import pandas as pd
import numpy as np
import os
from sklearn import model_selection
import config

def create_train_folds():
    
    df_train = pd.read_csv(config.TRAIN_FILE)
    
    #creating the kfold and default all to -1 
    df_train['kfold'] = -1

    df_train.sample(frac=1).reset_index(drop=True)
    y_train = df_train['target']
    

    kf = model_selection.StratifiedKFold(n_splits = 5)

    for k, (train_id, validation_id) in enumerate(kf.split(X=df_train, y=y_train)):
        df_train.loc[validation_id, 'kfold'] = k
    
    df_train.to_csv(os.path.join(config.INPUT_DIR, 'distaster_kfolds.csv'), index=False)

if __name__ == "__main__":
    create_train_folds()
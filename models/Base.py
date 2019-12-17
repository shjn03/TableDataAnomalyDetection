#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 22:28:26 2019

@author: shingo
"""
import optuna
from functools import partial
import pandas as pd
import os
import joblib
from sklearn.metrics import log_loss, mean_squared_error

class BaseModel:
    """
    異常検知モデル定義の抽象クラス
    全てのモデルに共通する部分を記述
    """
    
    def __init__(self):
        self.model = None
        self.BASE_DIR = "BASE_MODEL"

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_train):
        return self.model.predict(x_train)

    def optimize(self, data, n_trials=100, n_jobs=4,
                 direction="minimize", save_path=None):
        '''
        パラメータを最適化してモデルを上書きする
        '''
        obj_f = partial(self.objective, data)
        self.study = optuna.create_study(pruner=optuna.pruners.MedianPruner(
                n_warmup_steps=50), direction=direction)
        self.study.optimize(obj_f, n_trials=n_trials, n_jobs=n_jobs)
        if save_path is None:
            save_path = self.BASE_DIR
        self.study.trials_dataframe().to_csv(os.path.join(self.BASE_DIR,
                                                          "study_result.csv"))
        self.model.set_params(**self.study.best_params)
        hyper_params = self.model.get_params()
        hyp_df = pd.DataFrame.from_dict(hyper_params, orient="index",
                                        columns=["value"])
        hyp_df.to_csv(os.path.join(save_path, "best_params.csv"))

    def save(self, path, name):
        '''
        モデルの保存（sklearn用）
        joblibが使えないAPI場合は別途オーバーライドすること
        '''
        assert self.model is not None,"save method called before fitted.plase train fit model first."
        joblib.dump(self.model, os.path.join(path, name), compress=True)
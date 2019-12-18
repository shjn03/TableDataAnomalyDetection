#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 23:37:34 2019

@author: shingo
"""

from .Base import BaseModel
import os
from pyod.models.auto_encoder import AutoEncoder
from sklearn.preprocessing import StandardScaler

MODEL_NAME = "AutoEncoder"
class AEModel(BaseModel):
    def __init__(self, base_dir=None):
        super().__init__()
        self.BASE_DIR = os.path.join(base_dir, MODEL_NAME)
        os.makedirs(self.BASE_DIR, exist_ok=True)
        self.model = AutoEncoder()
        self.scaler = StandardScaler()

    def fit(self, x_train):
        if x_train.shape[1] < 64 and x_train.shape[1] >32:
            self.model.hidden_neurons = [32, 16, 32]
        elif x_train.shape[1] < 32 and x_train.shape[1] >16:
            self.model.hidden_neurons = [16, 4, 16]
        else:
            self.model.hidden_neurons = [2]
        scaled_x_train = self.scaler.fit_transform(x_train)
        self.model.fit(scaled_x_train)

    def predict(self, x_test):
        scaled_x_test = self.scaler.fit_transform(x_test)
        pred = self.model.predict(scaled_x_test)
        return pred

    def decision_function(self, x_test):
        scaled_x_test = self.scaler.fit_transform(x_test)
        pred = self.model.decision_function(scaled_x_test)
        return pred
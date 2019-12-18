#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 22:32:52 2019

@author: shingo
"""
from .Base import BaseModel
import os
from sklearn.neighbors import LocalOutlierFactor

MODEL_NAME = "LocalOutlierFactor"


class LOFModel(BaseModel):
    def __init__(self, base_dir=None):
        super().__init__()
        self.BASE_DIR = os.path.join(base_dir, MODEL_NAME)
        os.makedirs(self.BASE_DIR, exist_ok=True)
        self.model = LocalOutlierFactor(novelty=True)

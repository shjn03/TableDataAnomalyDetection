#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 23:17:07 2019

@author: shingo
"""
from .Base import BaseModel
import os
from pyod.models.pca import PCA

MODEL_NAME="PCA"
class PCAModel(BaseModel):
    def __init__(self, base_dir=None):
        super().__init__()
        self.BASE_DIR = os.path.join(base_dir, MODEL_NAME)
        os.makedirs(self.BASE_DIR, exist_ok=True)
        self.model = PCA()
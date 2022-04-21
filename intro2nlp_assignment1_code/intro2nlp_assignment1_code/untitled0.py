# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:39:12 2022

@author: lefko
"""
import os
import pandas as pd
import numpy as np
import matplotlib as plt

dir1 = 'D:\\arxeia\\AI_VU\\Data Mining\\ass1'
os.chdir(dir1)

df = pd.read_csv('ODI-2022.csv', sep=';')
#304 entries, 17attr

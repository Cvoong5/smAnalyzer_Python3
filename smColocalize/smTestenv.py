#smSupport.py
import numpy as np
import pandas as pd
import os
import smSupport as smS
import smCore as smC
import smColocalize as sm2
import matplotlib.pyplot as plt
import scipy

#Read files

directory = '/Users/calvin/Library/CloudStorage/OneDrive-UCB-O365/7 Single_molecule_data/smMovies/Calvin/2023/20231207/Experiment1'
os.chdir(directory)
file1 = 'HMGB1001.nd2'

movie2 = smC.read_movie(file1)


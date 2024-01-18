import sma
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate


file = "HMGB1001.nd2"
directory = "/Users/calvin/Desktop/smF_analyzer/20240111/"
os.chdir(directory)

movie, summed_image = sma.imread(file)
bg_sub, background = sma.TemporalMedianFilter(summed_image, radius = 3)
coordinates = sma.detect(bg_sub, threshold = 500)
gaussfit, fit_coordinate = sma.gauss2d_Fit(bg_sub, coordinates)
time_series = sma.extract(movie, fit_coordinate) #== frame, spot_intensity, background

#df = pd.read_csv("spot_3_52.csv")
#data = []
#for i in range(1, len(df)):
#    frame = i
#    intensity = df.loc[i, "spot_intensity"] - df.loc[i - 1, "spot_intensity"]
#    background = df.loc[i, "background"] - df.loc[i - 1, "background"]
#    dictionary = {"frame": frame, "spot_intensity": intensity, "background": background}
#    data.append(dictionary) 
#df2 = pd.DataFrame(data)
#fig, ax = plt.subplots(1,2)
#background_sub = [df2.loc[i, "spot_intensity"] - df.loc[i, "background"] for i in range(len(df2))]
#ax[0].plot(df.loc[:, "frame"], df.loc[:, "spot_intensity"])
#ax[1].plot(df2.loc[:,"frame"], df2.loc[:, "spot_intensity"])
#plt.show()


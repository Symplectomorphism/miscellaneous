import numpy as np
import pandas as pd


data = pd.read_csv("./data/vector_data.csv")

names = [f"TIB_LAT_CFN", "TIB_MED_CFN", "PAT_CFN", "TIB_LAT_CoF", "TIB_MED_CoF", "PAT_CoF"]

vahid_data = np.zeros((6, 12, 241, 3))
amanda_data = np.zeros((6, 12, 241, 3))
manuel_data = np.zeros((6, 12, 241, 3))

for k in range(6):
    for j in range(12):
        for i in range(3):
            vahid_data[k, j, :, i] = data[data["s"] == f"S{j+1}"][names[k] + f"{i+1}" + "u_V"]
            amanda_data[k, j, :, i] = data[data["s"] == f"S{j+1}"][names[k] + f"{i+1}" + "u_A"]
            manuel_data[k, j, :, i] = data[data["s"] == f"S{j+1}"][names[k] + f"{i+1}" + "u_M"]
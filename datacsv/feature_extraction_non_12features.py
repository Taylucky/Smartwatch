import pandas as pd
import numpy as np
import os

name_list=os.listdir(r'D:\月月\UVA\2023Fall\SP ML\Projects\Final\datacsv\non')
# Read the original CSV file
features= pd.DataFrame(None,columns=['mean_x', 'mean_y', 'mean_z', 'std_dev_x', 'std_dev_y', 'std_dev_z',
                                     'median_x', 'median_y', 'median_z', 'rms_x', 'rms_y', 'rms_z' ])
for j in name_list:
    data = pd.read_csv(r'D:\月月\UVA\2023Fall\SP ML\Projects\Final\datacsv\non'+'\\'+ j,header=None)
    # Take the first 1000 rows
    data = data.head(5000)
    # Split the data into groups of 100 rows each
    data_groups = [data.iloc[i:i + 100] for i in range(0, len(data), 100)]
    #print(np.array(data_groups).shape)
# Calculate mean and standard deviation for each group
    result = []
    for group in data_groups:
        # print(np.array(group).shape)
        means = group.iloc[:, 1:4].mean(axis=0)
        # print(np.array(group.iloc[:, 1:4]).shape)
        # print(np.array(means).shape)
        # print("mean1:",means)
        std_devs = group.iloc[:, 1:4].std()
        # print("std1:", std_devs)
        medians = group.iloc[:, 1:4].median()
        rms = np.sqrt((group.iloc[:, 1:4]**2).mean())
        result.append(pd.concat([means, std_devs, medians, rms]))
        # print("result:", result)

    # print("means:",means)
    print(np.array(result).shape)
    print(result[0].values)
    # Create a DataFrame for the results
    result1 = []
    for i in range(0,50):
        result1.append(result[i].values)

    #print(np.array(result1).shape)
    #print(result1)

    features_data = pd.DataFrame(result1, columns=['mean_x', 'mean_y', 'mean_z', 'std_dev_x', 'std_dev_y', 'std_dev_z',
                                                   'median_x', 'median_y', 'median_z','rms_x', 'rms_y', 'rms_z'])
    print("features_shape:",features_data.shape)
    features=features._append(features_data)
    print("shape:",features.shape)
    #print("ok!",features)
# Save the features to a new CSV file
features.to_csv('features_non_12.csv', index=False)# 2sec_12features_nothandwash
import pandas as pd
import numpy as np
import os

name_list=os.listdir(r'D:\月月\UVA\2023Fall\SP ML\Projects\Final\datacsv\counterclockwise')
# Read the original CSV file
features= pd.DataFrame(None,columns=['mean_x', 'mean_y', 'mean_z', 'std_dev_x', 'std_dev_y', 'std_dev_z'])
for i in name_list:
    data = pd.read_csv(r'D:\月月\UVA\2023Fall\SP ML\Projects\Final\datacsv\counterclockwise'+'\\'+ i,header=None)
    # Take the first 1000 rows
    data = data.head(500)
    # Split the data into groups of 100 rows each
    data_groups = [data.iloc[i:i + 100] for i in range(0, len(data), 100)]
    print(np.array(data_groups).shape)
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
        result.append(pd.concat([means, std_devs]))
        # print("result:", result)

    # print("means:",means)
    print(np.array(result).shape)
    print(result[0].values)
    # Create a DataFrame for the results
    result1 = []
    for i in range(0,5):
        result1.append(result[i].values)

    #print(np.array(result1).shape)
    #print(result1)

    features_data = pd.DataFrame(result1, columns=['mean_x', 'mean_y', 'mean_z', 'std_dev_x', 'std_dev_y', 'std_dev_z'])
    features=features._append(features_data)
    print("ok!",features)
# Save the features to a new CSV file
features.to_csv('features_counterclockwise_6.csv', index=False)
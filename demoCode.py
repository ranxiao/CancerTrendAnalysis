import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import ta_l1_tf as ta

# # load
# with open('sampleDataResults/featureDict.pickle', 'rb') as f:
#     featureDict = pickle.load(f)
#

# get all files in ./sampleData
import os
from os import listdir
from os.path import isfile, join

# get all files in ./sampleData
mypath = './sampleData'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# create folder to store results
if not os.path.exists('sampleDataResults'):
    os.makedirs('sampleDataResults')

# create a feature dictionary to store file name and feature table
featureDict = []
# log the files that failed to optimize
files_failedOptimization = []
# log the files that have less than 3 data points
files_lessThan3DataPoints = []
# log the files that are flat, i.e., all values are the same
files_flat = []

# loop through all files
for file in files:
    # print the index of the file and the total number of files
    print('file index: '+str(files.index(file)+1)+' out of '+str(len(files)))
    # read the file
    pd_exp = pd.read_csv(mypath+'/'+file)
    # get rows with the first concept ID in the file (can change to analysis other concept IDs)
    # Convert the date strings to datetime objects and Convert to the number of days since epoch
    pd_oneVar = (pd_exp.query("measurement_concept_id == measurement_concept_id[0]")
                 .sort_values('measurement_date')
                 .assign(measurement_date=lambda x: (pd.to_datetime(x['measurement_date']) - pd.Timestamp('1970-01-01')) // pd.Timedelta('1D')))

    # remove rows with value_as_number = NaN
    pd_oneVar = pd_oneVar[pd_oneVar['value_as_number'].notna()]

    # remove duplicate rows with the same measurement_date and keep the first row
    pd_oneVar = pd_oneVar.drop_duplicates(subset=['measurement_date'], keep='first')

    # prepare the data for trend analysis
    y = pd_oneVar['value_as_number'].values
    t = pd_oneVar['measurement_date'].values
    # re-reference time t to time 0 and absolute value in days
    t = np.abs(t - t[0])

    if len(y) >2:
        # declare the trend features as an empty dataframe
        trendFeatures = pd.DataFrame(columns=['slope', 'duration', 'total_dur', 'start_v', 'end_v', 'sign', 'lmd'])
        try:
            # get the max_scale for the y
            max_scale = ta.l1tf_lambdamax(y, t)
            if max_scale == 0:
                files_flat.append(file)
                continue
            min_scale = 1e-1
            if max_scale < min_scale:
                min_scale = max_scale/10

            num_points = 3
            # get the linearly spaced lambdas
            lambdas = np.linspace(min_scale, max_scale, num_points)
            # loop through all lambdas
            for lambda_ in lambdas:
                # get the trend features
                features,trends = ta.getTrendFeatures(y, lambda_, t, 0)

                # plot out x and trends with t
                plt.figure()
                plt.plot(t, y, 'k.', label='signal')
                plt.plot(t, trends, 'r-', label='l1tf')
                # save the last figure with filename and lambda
                plt.savefig('sampleDataResults/trend_'+file+'_'+str(lambda_)+'.png')
                plt.close()

                # append the trend features to trendFeatures
                trendFeatures = pd.concat([trendFeatures, features], ignore_index=True)

                # generate trend features table (not timing features are concerted into proportions of total duration, so unit does not matter)
                featureTable = ta.genFeatureTable(trendFeatures)

        except: # if trend analsyis cannot converge, then use the slope based on the first and last point, lmd is labeled as -1
            delta_t = np.diff(t) # use the original time difference so that the slope reflects its original time unit
            temp_slope = (y[-1] - y[0])/sum(delta_t)
            temp_dur = sum(delta_t)
            slope_totalDuration = sum(delta_t)
            temp_start_v = y[0]
            temp_end_v = y[-1]
            temp_sign = 1 if temp_slope > 0 else -1
            features = [temp_slope, temp_dur, slope_totalDuration, temp_start_v, temp_end_v, temp_sign,-1]
            # assign features to trendFeatures
            trendFeatures.loc[0] = features

            # save the file name to files_failedOptimization
            files_failedOptimization.append(file)

        # append the file name and feature table to featureDict
        featureDict.append({'file':file, 'featureTable':featureTable})
    else:
        # save the file name to files_lessThan3DataPoints
        files_lessThan3DataPoints.append(file)

# save the feature dictionary as a pickle file
with open('sampleDataResults/featureDict.pickle', 'wb') as f:
    pickle.dump(featureDict, f)
# save the files_failedOptimization as a csv file
pd.DataFrame(files_failedOptimization).to_csv('sampleDataResults/files_failedOptimization.csv')
# save the files_lessThan3DataPoints as a csv file
pd.DataFrame(files_lessThan3DataPoints).to_csv('sampleDataResults/files_lessThan3DataPoints.csv')
# save the files_flat as a csv file
pd.DataFrame(files_flat).to_csv('sampleDataResults/files_flat.csv')


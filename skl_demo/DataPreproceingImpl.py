import numpy as np
import pandas as pd


def normalization(data, feature_range=(0, 1)):
    min, max = data.min(axis=0), data.max(axis=0)
    range_min, range_max = feature_range[0], feature_range[1]
    data_normalization = (data - min) / (max - min)
    # print data_normalization
    return data_normalization * (range_max - range_min) + range_min, (min, max)


def inverse_normalization(normalization_data, min, max, feature_range=(0, 1)):
    range_min, range_max = feature_range[0], feature_range[1]
    data = (normalization_data - range_min) / (range_max - range_min)
    # print data
    return data * (max - min) + min


data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
data = np.array(data)

normalization_data, (min, max) = normalization(data, feature_range=(5, 10))
print normalization_data, min, max
data = inverse_normalization(normalization_data, min, max, feature_range=[5, 10])
print data

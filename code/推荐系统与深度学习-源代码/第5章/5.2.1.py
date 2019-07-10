import numpy as np
from sklearn import preprocessing

x = np.array([[ 1., -1.,  2.],
              [ 2.,  0.,  0.],
              [ 0.,  1., -1.]])
x_scaled = preprocessing.scale(x)
print(x_scaled)


##############################################
import numpy as np
from sklearn import preprocessing

x = np.array([[ 1., -1.,  2.],
              [ 2.,  0.,  0.],
              [ 0.,  1., -1.]])
x_max_min_scaled = preprocessing.MinMaxScaler().fit_transform(x)
print(x_max_min_scaled)


##############################################
import numpy as np
from sklearn import preprocessing

x = np.array([[ 1., -1.,  2.],
              [ 2.,  0.,  0.],
              [ 0.,  1., -1.]])
x_normalize =preprocessing.normalize(x, norm='l2')
print(x_normalize)


##############################################
##One-Hot
import numpy as np
from sklearn import preprocessing

one_hot_enc= preprocessing.OneHotEncoder()
one_hot_enc.fit([[1 ,1, 2], [0, 1, 0], [0, 2, 1], [1, 0, 3]])
after_one_hot = one_hot_enc.transform([[0, 1, 3]]).toarray()
print(after_one_hot)


##############################################
##hash-track
def hashing_vectorizer(s, N):
    x = [0 for i in xrange(N)]
    for f in s.split():
        h = hash(f)
        x[h % N] += 1
    return x
print hashing_vectorizer('make a hash feature', 3)


##############################################
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
    load_robot_execution_failures
download_robot_execution_failures()
timeseries, y = load_robot_execution_failures()

from tsfresh import extract_features
extracted_features = extract_features(timeseries, column_id="id", column_sort="time")

from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

impute(extracted_features)
features_filtered = select_features(extracted_features, y)
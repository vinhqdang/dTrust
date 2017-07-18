import argparse

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def parse_args():
    '''
    Parses the dTrust arguments.
    '''
    parser = argparse.ArgumentParser(description="dTrust algorithm for rating prediction in social recommendation.")

    parser.add_argument('--input_filename', nargs='?', default='epinions2_full.csv',
                        help='Input rating/trust file')

    return parser.parse_args()



def main (args):
    # load dataset
    dataframe = pandas.read_csv("args.input_filename")
    dataset = dataframe.values
    # split into input (X) and output (Y) variables
    X = dataset[:,[0,1,4]]
    Y = dataset[:,3]

    # define base model
    def baseline_model():
        # create model
        model = Sequential()
        model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)
    # evaluate model with standardized dataset
    estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)

    kfold = KFold(n_splits=10, random_state=seed)
    results = cross_val_score(estimator, X, Y, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

if __name__ == "__main__":
    args = parse_args()
    main(args)




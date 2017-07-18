import argparse

import numpy
import pandas
from keras.models import Sequential
# from keras.layers import Dense
from keras.layers.core import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
from keras.callbacks import TensorBoard, EarlyStopping
import datetime

def parse_args():
    '''
    Parses the dTrust arguments.
    '''
    parser = argparse.ArgumentParser(description="dTrust algorithm for rating prediction in social recommendation.")

    parser.add_argument('--input_filename', nargs='?', default='epinions2_full.csv',
                        help='Input rating/trust file')
    parser.add_argument('--layer_size', nargs='?', default='128,64,64',
                        help = "Size of each neural layer")
    parser.add_argument('--dropout', nargs='?', default='0.5,0.5,0.5',
                        help = "Dropout of each layer. Need to be the same length with layer_size.")
    parser.add_argument('--activation', nargs='?', default='relu',
                        help = "Activation function. We will use a same function for the network.")
    parser.add_argument('--epochs', type = int, nargs='?', default=50,
                        help = "Number of epochs.")
    parser.add_argument('--batch_size', type = int, nargs='?', default=32,
                        help = "Batch size")
    parser.add_argument('--verbose', type = int, nargs='?', default=2,
                        help = "Verbose level. From 0 to 2.")
    parser.add_argument('--min_delta', type = float, nargs='?', default=0.1,
                        help = "Min delta using in early stopping. If the model cannot improve more than min_delta, there is no improvement.")
    parser.add_argument('--patience', type = float, nargs='?', default=5,
                        help = "For early stopping. Is number of epochs wait for the model to improve.")
    return parser.parse_args()



def main (args):
    # load dataset
    print ("Read data")
    dataframe = pandas.read_csv(args.input_filename)
    dataset = dataframe.values

    # divide train and test
    print ("Divide data")
    trust = dataframe.loc[dataframe['Type'] == 'Rating']
    rating = dataframe.loc[dataframe['Type'] == 'Trust']
    
    train_rating = rating.sample (frac = 0.8)
    test_rating = rating.drop (train_rating.index)

    train = pandas.concat ([train_rating, trust])

    train = train.values
    test = test_rating.values

    # split into input (X) and output (Y) variables
    X_train = train[:,[0,1,4]]
    Y_train = train[:,3]

    X_test = test[:,[0,1,4]]
    Y_test = test[:,3]

    print ("Build model")
    # New sequential network structure.
    model = Sequential()

    layers = map (int, args.layer_size.split(','))
    dropouts = map (float, args.dropout.split(','))

    if (len(layers) < len(dropouts)):
        print ("Too many droput values. Cut off.")
    elif (len(layers) > len(dropouts)):
        print ("Too many layer without dropout. Auto fill out.")
        k = len(layers) - len (dropouts)
        for i in range(k):
            dropouts.append (dropouts[-1])

    for i in range(len(layers)):
        if i == 0:
            model.add(Dense(layers[i], input_dim=3, activation = args.activation))
            model.add(Dropout(dropouts[i]))
            model.add(Activation("linear"))
        else:
            model.add(Dense(layers[i], activation = args.activation))
            model.add(Dropout(dropouts[i]))
            model.add(Activation("linear"))
    model.add(Dense(1))
     
    # Model is derived and compiled using mean square error as loss
    # function, accuracy as metric and gradient descent optimizer.
    model.compile(loss='mse', optimizer='adam', metrics=["mae","mse"])

    # tensorboard
    tensorboard = TensorBoard(log_dir='./logs_' + datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S"), histogram_freq=0,
                          write_graph=True, write_images=False)
    earlystopping = EarlyStopping(monitor = 'mse', min_delta = args.min_delta, patience = args.patience, verbose = args.verbose)
     
    print ("Training")
    # Training model with train data. Fixed random seed:
    numpy.random.seed(3)
    model.fit(X_train, Y_train, epochs = args.epochs, batch_size = args.batch_size, verbose=args.verbose, callbacks=[tensorboard,earlystopping])

    print ("Predict")
    predicted = model.predict(X_test)
 
    # Plot in blue color the predicted adata and in green color the
    # actual data to verify visually the accuracy of the model.
    pyplot.plot(predicted, color="blue")
    pyplot.plot(Y_test, color="green")
    pyplot.show()

if __name__ == "__main__":
    args = parse_args()
    main(args)




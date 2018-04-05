import numpy as np
import argparse
import csv
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

def parse_argument():
    """
    Code for parsing arguments
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('--train', nargs=1, required=True)
    parser.add_argument('--test', nargs=1, required=True)
    parser.add_argument('--numTrees', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args


def adaboost(X, y, num_iter):
    """Given an numpy matrix X, a array y and num_iter return trees and weights 
   
    Input: X, y, num_iter
    Outputs: array of trees from DecisionTreeClassifier
             trees_weights array of floats
    Assumes y is in {-1, 1}^n
    """
    trees = []
    trees_weights = []
    N = len(y)
    curr_weights = [1/float(len(y)) for i in range(N)]
    for i in range(num_iter):
        alpha = 1
        tree = DecisionTreeClassifier(max_depth=1)
        tree.fit(X,y, sample_weight=curr_weights)
        y_pred = tree.predict(X)
        to_change = [i for i in range(N) if y_pred[i]!=y[i]]
        error = np.sum([curr_weights[i] for i in to_change])/float(np.sum(curr_weights))
        if to_change:
            alpha = np.log((1 - error)/(error))
        curr_weights = [curr_weights[j]*np.exp(alpha) if j in to_change else curr_weights[j] for j in range(N)]
        trees.append(tree)
        trees_weights.append(alpha)
    return trees, trees_weights


def adaboost_predict(X, trees, trees_weights):
    """Given X, trees and weights predict Y
    assume Y in {-1, 1}^n
    """
    y_pred = [0 for i in range(len(X))]
    for i in range(len(trees_weights)):
        y_pred+=trees_weights[i]*trees[i].predict(X)
    
    Yhat = [np.sign(y_pred[i])*1 for i in range(len(y_pred))]        
    return Yhat


def parse_spambase_data(filename):
    """ Given a filename return X and Y numpy arrays
    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row.
    """
        # your code here
    data = np.loadtxt(filename, delimiter=',')
    Y = [data[i][-1] for i in range(len(data))]
    X = [data[i][0:-1] for i in range(len(data))]
    return np.array(X), np.array(Y)

def new_label(Y):
    """ Transforms a vector od 0s and 1s in -1s and 1s.
    """
    return [-1. if y == 0. else 1. for y in Y]

def old_label(Y):
    return [0. if y == -1. else 1. for y in Y]

def accuracy(y, pred):
    return np.sum(y == pred) / float(len(y)) 

def main():
    """
    This code is called from the command line via
    
    python adaboost.py --train [path to filename] --test [path to filename] --numTrees 
    """
    args = parse_argument()
    train_file = args['train'][0]
    test_file = args['test'][0]
    num_trees = int(args['numTrees'][0])
    print train_file, test_file, num_trees
    X_train,Y_train = parse_spambase_data(train_file)
    X_test, Y_test = parse_spambase_data(test_file)
    Y_train = new_label(Y_train)

    trees,weights = adaboost(X_train, Y_train, num_trees)
    Yhat = adaboost_predict(X_train, trees, weights)
    Yhat_test = adaboost_predict(X_test, trees, weights)
    Yhat = np.array(old_label(Yhat))
    Yhat_test = np.array(old_label(Yhat_test))
    Y_train = np.array(old_label(Y_train))

    ## here print accuracy and write predictions to a file
    acc_test = accuracy(Y_test, Yhat_test)
    acc = accuracy(Y_train, Yhat)
    print("Train Accuracy %.4f" % acc)
    print("Test Accuracy %.4f" % acc_test)
    
    data = np.loadtxt(test_file, delimiter=',')
    mat = np.matrix(data)
    y = np.transpose(np.matrix(Yhat_test))
    all_data = np.append(mat, y, 1)
    np.savetxt('predictions.txt', all_data, delimiter=',', fmt='%.1f')

if __name__ == '__main__':
    main()

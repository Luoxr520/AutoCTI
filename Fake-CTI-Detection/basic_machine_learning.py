'''基础机器学习模块'''
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time


def main(args):
    """
    Run desired method
    """
    print(args)
    input_name = args.dataDir + args.input

    data = readdData(input_name)

    indices = np.arange(2000)
    # print(len(y))
    x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(data['text'], data['label'], indices, test_size=0.2,
                                                        shuffle=True, random_state=42)
    
    vectorizer = CountVectorizer()
    tfidf = TfidfTransformer()

    x_train = vectorizer.fit_transform(x_train)
    x_train = tfidf.fit_transform(x_train)

    
    x_test = vectorizer.transform(x_test)
    x_test = tfidf.transform(x_test)

    features = args.features
    if features == 1:
        x_train, x_test= runSomeFeatures(data, x_train, x_test, indices_train, indices_test)
    elif features == 2:
        x_train, x_test= runAllFeatures(data, x_train, x_test, indices_train, indices_test)

        
    modelName = args.model
    
    optimise = args.optimise

    if optimise:
        model = runOptimise(x_train, y_train, modelName)
    else:
        model = train(x_train, y_train, modelName)
    
    test(x_test, y_test, model, modelName)




def readdData(input_name):
    """
    Read the data set from the provided file
    """
    print("###Read Dataset###")
    df = pd.read_excel(input_name, engine="openpyxl", sheet_name="Sheet1", header=0)
    input_rows = df.shape[0]
    print("Read input file. Rows: " + str(input_rows))

    # combine topic and content columns
    df['text'] = df['topic_processed'] + ' ' + df['content_processed']
    
    # convert labels to 1 and 0
    df['label'] = df['label'].map({'Real': 1, 'Fake': 0})

    # drop excess columns
    df = df.drop(columns = ['topic', 'content','topic_processed', 'content_processed'])

    return df


def dispConfusionMatrix(y_test, predicted):
    """
    Display confusion matrix for the model in command line and ploy
    """
    cm = confusion_matrix(y_test, predicted)
    print('TP: ' + str(cm[0][0]) + '\t\tFP: ' + str(cm[0][1]))
    print('FN: ' + str(cm[1][0]) + '\t\tTN: ' + str(cm[1][1]))

    # plot the confusion matrix
    classes = ['Real', 'Fake']
    plt.clf()
    sn.heatmap(cm, annot=True, cmap="Blues", xticklabels=classes, yticklabels=classes, fmt='g')
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()


def dispReport(ModelName, DatasetName, y_true, y_pred):
    """
    Display performance report for the model
    """
    print('\n*** ' + ModelName + ' Model - ' + DatasetName + ' Dataset ***')
    print(classification_report(y_true, y_pred))
    dispConfusionMatrix(y_true, y_pred)


def scaleFeatures(data, columns):
    """
    Scale the features with min max scaling
    """
    scaled = pd.DataFrame()
    
    # run scaling for each column
    for column in columns:
        # if there is no difference in minimum and maximum 
        if (data[column].max() - data[column].min()) == 0:
            scaled[column] = data[column] - data[column].min()
        else:
            scaled[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
        scaled[column] = np.nan_to_num(scaled[column])

    return scaled


def appendFeatures(x, indices, scaled, columns):
    """
    Append the desired features to the passed array
    """
    x = np.hstack((x.toarray(), np.array(scaled.loc[indices, columns])))

    return x

def getModel(modelName):
    """
    Get the desired model for training
    """
    if modelName == 'SVC':
        model = SVC(kernel='linear')
    elif modelName == "Logistic Regression":
        model = LogisticRegression(C=10)
    elif modelName == "Random Forest":
        model = RandomForestClassifier(max_depth=30, max_features='auto', min_samples_leaf=2, random_state=42)
    else:
        model = PassiveAggressiveClassifier(C=0.01)
    return model

def runAllFeatures(data, x_train, x_test, indices_train, indices_test):
    '''
    Get the features in the dataframe and append to train and test data
    '''
    # get the existing column names other than text and layer
    data = data.drop(columns = ['text', 'label'])
    columns = data.columns

    # scale the data
    scaled = scaleFeatures(data, columns)

    # append features to x_train and x_test
    x_train = appendFeatures(x_train, indices_train, scaled, columns)
    x_test = appendFeatures(x_test, indices_test, scaled, columns)

    return x_train, x_test

def runSomeFeatures(data, x_train, x_test, indices_train, indices_test):
    '''
    Get the features in the dataframe and append to train and test data
    '''
    # created columns
    columns = ['topic_words_count', 'content_words_count', 'total_words_count',
               'total_sentence_count', 'total_paragraph_count', 'mean_words_per_sentence',
               'mean_words_per_paragraph', 'mean_sent_per_paragraph', 'stdev_words_per_sentence',
               'stdev_words_per_paragraph', 'stdev_sent_per_paragraph', 'flesch_reading_ease',
               'flesch_kincaid_readability', 'gunning_fog_readability', 'automatic_readability_index']

    # scale the data
    scaled = scaleFeatures(data, columns)

    # append features to x_train and x_test
    x_train = appendFeatures(x_train, indices_train, scaled, columns)
    x_test = appendFeatures(x_test, indices_test, scaled, columns)

    return x_train, x_test

def optimiseSVC():
    '''
    Create grid search for optimisation of SVC model
    '''
    param_grid = {
        'C': [0.1, 1, 10, 100],  # Values of C to explore
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']  # Different kernel functions to explore
    }

    svc = SVC()
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=2)
    
    return grid_search

def optimiseLR():
    '''
    Create grid search for optimisation of Linear Regression model
    '''
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Values of C to explore
        'penalty': ['l1', 'l2']  # Regularization type
    }

    lr = LogisticRegression()
    grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=2)

    return grid_search

def optimiseRF():
    '''
    Create grid search for optimisation of Random Forest model
    '''
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2']
    }

    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=2)

    return grid_search


def optimisePA():
    '''
    Create grid search for optimisation of Passive Aggressive model
    '''
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],  # Values of C to explore
        'loss': ['hinge', 'squared_hinge']  # Loss functions to explore
    }

    pa = PassiveAggressiveClassifier()
    grid_search = GridSearchCV(estimator=pa, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=2)

    return grid_search

def runOptimise(x_train, y_train, modelName):
    '''
    Optimise selected model selected model
    '''
    # Get desired grid search model
    if modelName == 'SVC':
        grid_search = optimiseSVC()
    elif modelName == "Logistic Regression":
        grid_search = optimiseLR()
    elif modelName == "Random Forest":
        grid_search = optimiseRF()
    else:
        grid_search = optimisePA()

    # Fitting grid search
    grid_search.fit(x_train, y_train)

    # Print the best parameters
    best_params = grid_search.best_params_
    print(f"{modelName} Model Best Parameters: \n{best_params}")

    # Get the best model
    model = grid_search.best_estimator_    
    
    return model


def train(x_train, y_train, modelName):
    '''
    Train selected model
    '''
    # Get desired model
    model = getModel(modelName)
    
    # Fitting the model
    start = time.time()
    model.fit(x_train, y_train)
    end = time.time()
    print(f"Training time: {end-start} seconds")
    
    return model

def test(x_test, y_test, model, modelName):
    '''
    Test the accuracy of the fitted model
    '''
    # Accuracy
    predicted = model.predict(x_test)
    predicted = np.where(predicted>=0.5, 1, 0)
    dispReport(modelName, 'Test', y_test, predicted)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ####################################################################
    # Parse command line
    ####################################################################

    # default dirs
    parser.add_argument('-d', '--dataDir', type=str, default='./dataset/', help='intput Corpus folder')

    # input
    parser.add_argument('-i', '--input', type=str, default='dataset_long_with_features.xlsx', help='output file name')

    # model
    parser.add_argument('-m', '--model', type=str, default='SVC', choices = ['SVC', 'Logistic Regression', 'Random Forest', 'Passive Aggressive'], help='model type')

    # use features
    parser.add_argument('-f', '--features', type=int, default=2, choices = [0, 1, 2], help='what features to include: 0 for None, 1 for stylometry and readability, 2 for all')

    # optimise
    parser.add_argument('-o', '--optimise', type=bool, default=False, help='whether to run to optimise')

    m_args = parser.parse_args()
    main(m_args)

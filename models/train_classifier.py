import sys
import os
import pandas as pd
import numpy as np
import re
import sqlalchemy as db
import argparse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, make_scorer

from functions import (tokenize, evaluate_results, TextLengthExtractor, TextAugmentation)

import joblib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_data(database_filepath):
    """
    Loads the data from .csv files and create a raw but merged DataFrame

    Args:
        database_filepath:  path to sqlite .db database file
    Returns:
        X:                  Features of the dataset (as DataFrame)
        Y:                  Labels of the dataset (as DataFrame)
        category_names:     list containing the category names
    """
    engine = db.create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_data', engine)
    X = pd.DataFrame(df.message)
    Y = df.iloc[:,5:]
    category_names = df.columns[5:]
    return X, Y, category_names


def get_model_params(apply_grid_search, X, Y, category):
    """
    Helper function for build_model() that gets paramters for the
    final pipeline as defined in build_model().
    
    Args:
        apply_grid_search: True, if grid search optimization 
                           should be used to tune the params.
                           If True, a single-class classifier
                           is finetuned using grid search. The
                           optimal parameters gained are then
                           used for all classifiers.
        X:                 features used for grid search
        Y:                 labels used for grid search
        category:          category for which the single-class
                           classifier is defined and tuned
    Returns:
        params:            parameters for model as defined in
                           build_model()
    """
    
    params = dict({'n_estimators': 100, 'min_samples_split': 2,
                    'class_weight': 'balanced'})

    if (apply_grid_search):
        # Define single-class pipeline identical as in build_model()
        pipeline_single_class = Pipeline([
            ('features', FeatureUnion([
                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ])),
                ('text_length', TextLengthExtractor())
            ])),
            ("clf", RandomForestClassifier())
            ])
        # Define grid search parameters. Note that one can set
        # more parameters, but this will increase optimization time.
        parameters = {'clf__n_estimators': [50, 100],
                        'clf__min_samples_split': [2],
                        'clf__class_weight': ['balanced']}

        # We are interested in optimizing for the recall
        scorer = make_scorer(recall_score)

        # Define grid search and fit the model
        model_single_class = GridSearchCV(pipeline_single_class, param_grid=parameters, scoring=scorer)
        model_single_class.fit(X, Y[category])

        # Set optimal parameters
        best_params = model_single_class.best_params_
        params['n_estimators'] = best_params['clf__n_estimators']
        params['min_samples_split'] = best_params['clf__min_samples_split']
        params['class_weight'] = best_params['clf__class_weight']

    return params


def build_model(params):
    """
    Build an end-to-end ML pipeline including feature transformation
    and estimation.
    
    Args:
        params:         parameters set to the model
    Returns:
        pipeline:       end-to-end ML pipeline
    """

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('text_length', TextLengthExtractor())
        ])),
        ("clf", MultiOutputClassifier(
                RandomForestClassifier(n_estimators = params['n_estimators'],
                                        min_samples_split = params['min_samples_split'],
                                        class_weight = params['class_weight'])))
    ])

    return pipeline


def save_model(model, model_filepath):
    """
    Save the model as a .pkl file

    Args:
        model:          model to be saved
        model_filepath: path the model is saved to
    """
    joblib.dump(model, model_filepath)
    return


def main():
    """
    Main function of this program
    """

    help_text = 'Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl\n'

    if (len(sys.argv) == 2 and (sys.argv[1:][0] == '--help' or sys.argv[1:][0] =='-h') 
        or len(sys.argv) >= 3):

        # Add a parser to handle optional parameters
        parser = argparse.ArgumentParser(description='Train or read in a disaster response model. ' + help_text)
        parser.add_argument('database_filepath', type=str, 
            help='Path to .db database file')
        parser.add_argument('model_filepath', type=str, 
            help='Path to .pkl model file')
        parser.add_argument('--read_existing_model', dest='read_existing_model', action='store_true', 
            help='Read in model from model_filepath, if existing.')
        parser.add_argument('--skip_data_augmentation', dest='skip_data_augmentation', action='store_true', 
            help='If not set, trainings data is augmented so that each category has at least 3000 samples.')
        parser.add_argument('--skip_grid_search', dest='skip_grid_search', action='store_true', 
            help='Do not use grid search to accelerate the training.')

        # Access parser arguments
        args = parser.parse_args()
        database_filepath = args.database_filepath
        model_filepath = args.model_filepath
        train_model = not (args.read_existing_model and os.path.isfile(model_filepath))
        apply_grid_search = not args.skip_grid_search
        apply_data_augmentation = not args.skip_data_augmentation

        # Check for correct file endings
        _, file_extension = os.path.splitext(database_filepath)
        if file_extension != '.db':
            exception = Exception("Wrong input file format. Expect .db file")
            raise exception

        _, file_extension = os.path.splitext(model_filepath)
        if file_extension != '.pkl':
            exception = Exception("Wrong output file format. Expect .pkl file")
            raise exception

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        # Split the data and remove unnecessary raws from the training dataset
        # that does not have any labels set. Note that this is totally fine since
        # we train a classifier for each category. Therefore, for a category all
        # other categories serve as class with label=0, so there is still enough data.
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=8820)
        X_train = X_train.message.values
        X_test = X_test.message.values
        X_train = X_train[Y_train.sum(axis = 1) > 0]
        Y_train = Y_train[Y_train.sum(axis = 1) > 0]

        # Read in or build/train the model
        if train_model:
            # Augment the data for training
            if apply_data_augmentation:
                print('Augment data ...')
                augmentation = TextAugmentation(category_names, 3000)
                X_train, Y_train = augmentation.augment(X_train, Y_train)

            # Get (optimal) parameters
            if apply_grid_search:
                print('Find optimal model parameters. This may take a while ...')
            params = get_model_params(apply_grid_search, X_train, Y_train, 'water')

            # Build and tain the model
            print('Building model ...')
            model = build_model(params)
            print('Training model ...')
            model.fit(X_train, Y_train)
        else:
            # Load existing model
            print('Reading in trained model ...')
            model = joblib.load(model_filepath)
        
        print('Evaluating model ...')
        Y_pred = model.predict(X_test)
        evaluate_results(Y_pred, Y_test, category_names)

        # Only save model if not read in
        if train_model:
            print('Saving model ...\n    MODEL: {}'.format(model_filepath))
            save_model(model, model_filepath)
            print('Trained model saved!')

    else:
        print(help_text)


if __name__ == '__main__':
    main()
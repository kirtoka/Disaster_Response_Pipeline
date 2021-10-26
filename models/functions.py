import pandas as pd
import numpy as np
import re
from numpy.random import default_rng

import nltk
nltk.download(['averaged_perceptron_tagger', 'punkt', 'wordnet', 'stopwords'], quiet=True)
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import word_tokenize

from sklearn.base import BaseEstimator
from sklearn.metrics import (precision_score, recall_score, 
                             accuracy_score, f1_score)

from collections import Counter
from tqdm.auto import tqdm


def tokenize(text):
    """
    Tokenization function used for Bag of Words.
    This includes
        - URL replacement
        - Stopword removal
        - Lemmatization
    
    Args:
        text:           input text
    Returns:
        clean_tokens:   clean tokens    
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex, 'urlplaceholder', text)

    # tokenize text
    #text = re.sub(r"[^A-Za-z0-9]", " ", text.lower())
    # Ignore everything that is not a text, even numbers !""
    text = re.sub(r"[^A-Za-z]", " ", text.lower())
    text = re.sub(r" +", " ", text)
    tokens = text.split(" ")
    
    # Remove stop words and empty stringsstopwords_new = stopwords.words("english")
    stopwords_new = stopwords.words('english')
    stopwords_new = np.append(stopwords_new, ['', 'urlplaceholder'])
    tokens = [w for w in tokens if w not in stopwords_new]
    
    # Remove words that are not a verb, adjective, adverb or noun.
    def penn_to_wn_tags(pos_tag):
        if pos_tag.startswith('J'):
            return wordnet.ADJ
        elif pos_tag.startswith('V'):
            return wordnet.VERB
        elif pos_tag.startswith('N'):
            return wordnet.NOUN
        elif pos_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None    
    tokens_tagged = nltk.pos_tag(tokens)
    words = [w for w, tag in tokens_tagged if penn_to_wn_tags(tag) is not None]
    
    # Apply lemmatizer
    tokens_lemmatized = [WordNetLemmatizer().lemmatize(w, pos="v") for w in words]
    
    # Remove stop words and empty strings once more, there are maybe new once by the
    # lemmatization
    clean_tokens = [w for w in tokens_lemmatized if w not in stopwords_new]

    return clean_tokens


def evaluate_results(Y_pred, Y_gt, categories, print_results = True):
    """
    Evaluate the results (accuracy, precision, recall, f1-score).
    In addition, calculates the mean of all metrices over all categories. 

    Args:
        Y_pred:         predictions
        Y_gt:           ground truth
        categories:     list of all categories
        print_results:  if True, results are printed
    """
    
    # Transform data do pandas data frames
    Y_pred = pd.DataFrame(data = Y_pred)
    Y_gt = pd.DataFrame(data = Y_gt)
    
    if (Y_pred.shape[1] != len(categories)):
        raise Exception('Number of predictions does not fit length of categories.')
    
    # Get metrics for all categories
    metrics = []
    for col in range(Y_gt.shape[1]):
        accuracy = accuracy_score(Y_gt.iloc[:,col], Y_pred.iloc[:,col])
        precision=precision_score(Y_gt.iloc[:,col], Y_pred.iloc[:,col], zero_division=0)
        recall = recall_score(Y_gt.iloc[:,col], Y_pred.iloc[:,col], zero_division=0)
        f1 = f1_score(Y_gt.iloc[:,col], Y_pred.iloc[:,col], zero_division=0)
        metrics_tmp = [accuracy, precision, recall, f1]
        metrics.append(metrics_tmp)
        
    df_metrics = pd.DataFrame(metrics, index = categories, 
                            columns=["Accuracy", "Precision", "Recall", "F1-Score"])
    if print_results:
        print('Evaluation results for all categories:')
        print(df_metrics)
    
    # Get the mean over all categories
    df_metrics_mean = pd.DataFrame(df_metrics.mean(), columns = ['overall mean'])
    if print_results:
        print('\nMean evaluation metrics over all categories:')
        print(df_metrics_mean)
        
    return df_metrics, df_metrics_mean


class TextLengthExtractor(BaseEstimator):
    """
    Text lenght extractor that can be used as additional feature for
    text classifcation.
    """ 

    def sentence_length(self, text):
        """
        Function that returns the length of a text
        
        Args: 
            text:   text for which the number of words are counted
        """
        from nltk import word_tokenize
        #return len(text)
        return len(word_tokenize(text))

    def fit(self, X, y=None):
        return self

    def transform(self, X):        
        """
        Apply sentence_length() to all values in X
        NOTE: wrap X into a pandas Series to allow the lambda-functionality with apply
        
        Args: 
            X:    object where sentence_length() is applied on
        Returns:
            Pandas DataFrame
        """
        X_tagged = pd.Series(X).apply(self.sentence_length)

        return pd.DataFrame(X_tagged)


class TextAugmentation():
    """
    Class to augment text data. For each category that has not
    a certain number of samples, new sentences are generated
    based on tokens that appear from existing samples. Tokens that
    appear more often are more likely to appear in generated sentences
    as others.
    """

    def __init__(self, categories_to_augment, num_samples_total):
        """
        Initialization of class.

        Args:
          self:                   calling class
          categories_to_augment:  list of categories to be augmented
          num_samples_total:      requested number of total samples per category
        """
        self.X = []
        self.Y = []
        self.num_samples_total = num_samples_total
        self.categories_to_augment = categories_to_augment
    
    def augment_category(self, X, Y, category):
        """
        Generate augmented data for a single category.

        Args:
          self:           calling class
          X:              features used for augmentation
          Y:              labels of features
          category:       category that shall be augmented
        Returns:
          X_aug, Y_aug:   augmentations based on inputs
        """
        category_idx = [i for i, x in enumerate(Y.columns) if x == category][0]
        
        X_aug = []
        Y_aug = pd.DataFrame(columns = self.categories_to_augment)
        # Check if augmentation is required        
        num_augmentation = self.num_samples_total - X[Y[category] == True].shape[0]
        if (num_augmentation <= 0):
            # Nothing to do...
            return X_aug, Y_aug
        
        # Get some values that have the correct labels, however, avoid
        # that too much other labels as related are included
        X_category = np.array([])
        num_labels = 0
        while (X_category.shape[0] < 50 and num_labels < 5):
            X_category = X[(Y[category] == True) & (Y.sum(axis=1) <= num_labels)]
            num_labels = num_labels + 1

        if (X_category.shape[0] < 50):
            # Not enougth data to augment
            return X_aug, Y_aug

        # Get all tokens from available text samples
        all_tokens = []
        for text in X_category:
            all_tokens.extend(tokenize(text))
        all_tokens = np.sort(all_tokens)

        # Select top 50%, but not more than 124
        all_tokens_sorted = [word for word, _ in Counter(all_tokens).most_common()]
        all_tokens_counts = [counts for _, counts in Counter(all_tokens).most_common()]
        cumsums = np.array(all_tokens_counts).cumsum()
        tmp = cumsums[cumsums < len(all_tokens)*0.50]
        tokens_selected = all_tokens_sorted[0:len(tmp)][0:100]

        # Build a sentence that takes words from the most common parts
        # using exponential distributed random selection of words
        rng = default_rng()
        
        # Pre-allocate new data
        X_aug = np.empty([num_augmentation], dtype=object)
        Y_aug = pd.DataFrame(data = np.zeros([num_augmentation, len(Y.columns)], dtype=int),
                             columns = Y.columns)
        
        TextLength = 10
        ls = np.linspace(0.7, 7.5, len(tokens_selected))[::-1]
        for i in tqdm(range(0, num_augmentation)):
            vals = rng.exponential(1.5, len(tokens_selected))
            vals = np.sort(vals)[::-1]
            
            token_ids = [sum(val <= ls) for val in vals]
            new_text = [tokens_selected[item] for item in token_ids[0:TextLength]]
            new_sent  = ' '.join(new_text)
            
            X_aug[i] = new_sent
            Y_aug.iloc[i, category_idx] = 1
                
        Y_aug = Y_aug.astype('int')
        return X_aug, Y_aug
    
    def augment(self, X, Y):
        """
        Generate augmented data for all categories.

        Args:
          self:           calling class
          X:              features used for augmentation
          Y:              labels of features
        Returns:
          X_tmp, Y_tmp:   augmented data (including X/Y)
        """
        X_tmp = X
        Y_tmp = Y
        for category in tqdm(self.categories_to_augment):
            X_aug, Y_aug = self.augment_category(X, Y, category)
            X_tmp = np.append(X_tmp, X_aug)
            Y_tmp = pd.concat([Y_tmp, Y_aug], axis = 0)
            
        Y_tmp = Y_tmp.astype('int')            
        return X_tmp, Y_tmp
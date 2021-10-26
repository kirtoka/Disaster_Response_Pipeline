import sys
import os
import pandas as pd
import sqlalchemy as db

def load_data(messages_filepath, categories_filepath):
    """
    Loads the data from .csv files and create a raw but merged DataFrame

    Args:
        messages_filepath:       input message .csv file
        categories_filepath:     input categories .csv file
    Returns:
        df:                      raw output DataFrame
    """
    
    # Since we do not know if the input files have the format we
    # need to merge the files, put everything in a try-catch block
    try:
        df_messages = pd.read_csv(messages_filepath, dtype=str)
        df_categories = pd.read_csv(categories_filepath, dtype=str)
        if df_categories.shape[0] != df_messages.shape[0]:
            Exception('Number of rows does not fit')

        # Merge DataFrames
        df = pd.merge(df_messages, df_categories, how='inner', on='id')
    except:
        print('Error when reading in input .csv files')
        raise

    return df


def clean_data(df):
    """
    Clean the raw DataFrame to gain the desired format of the categories

    Args:
        df:  raw output DataFrame
    Returns:
        df:  DataFrame that is cleand inplace
    """
    
    # Create new categories DataFrame
    df_categories_new = df.categories.str.split(';', expand = True)

    # Rename columns
    categories_new_names = []
    for item in df_categories_new.iloc[0,:].str.split('-'):
        categories_new_names.append(item[0])
        
    df_categories_new.columns = categories_new_names

    # Replace text with integer value for category.
    # In addition, set all labels to 1 if the category is >0.
    categories_to_integer = lambda col: pd.to_numeric(col.str[-1])
    df_categories_new = df_categories_new.apply(categories_to_integer, axis = 0)
    df_categories_new[df_categories_new > 0] = 1

    # Renew categories in input df
    df.drop(columns=['categories'], inplace=True)
    df_categories_new['id'] = df.id
    df = pd.merge(df, df_categories_new, how='inner', on='id')
    df.drop_duplicates(subset=['id'], inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save DataFrame as a sqlite database

    Args:
        df:                 DataFrame
        database_filename:  filename of .db file
    """

    # If needed, delete already existing .db file
    try:
        os.remove(database_filename)
    except:
        pass

    engine = db.create_engine('sqlite:///' + database_filename, echo=False)
    df.to_sql('disaster_data', con=engine)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        # Check for correct input file endings
        inputs = [messages_filepath, categories_filepath]
        for inp in inputs:
            _, file_extension = os.path.splitext(inp)
            if file_extension != '.csv':
                exception = Exception("Wrong input file format. Expect .csv file")
                raise exception

        # Check for correct output file ending
        _, file_extension = os.path.splitext(database_filepath)
        if file_extension != '.db':
            exception = Exception("Wrong output file format. Expect .db file")
            raise exception

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
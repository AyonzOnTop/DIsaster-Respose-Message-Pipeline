import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loading data from csv file
    
    input:
        messages_filepath: filepath to message data
        categories_filepath: filepath to categories data
    output:
        df: Merged dataset from messages and categories
    '''
    #load data from files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, left_on = "id", right_on = "id", how = "left" )
    return df


def clean_data(df):
    '''
    Input:
        df: Merged dataset from messages and categories
    Output:
        df: Cleaned dataset
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',n = 36, expand = True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)


    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis =1)

    # drop duplicates
    df.drop_duplicates(subset = 'id', inplace = True)

    return df

def save_data(df, database_filename):
    '''
    INPUT 
        df: Dataframe to be saved
        database_filepath - Filepath used for saving the database     
    OUTPUT
        Saves the database
    ''' 
    # save the data to an sql database
    engine = create_engine('sqlite:///' + database_filename)

    df.to_sql(database_filename, engine, index=False)

    return df


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

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
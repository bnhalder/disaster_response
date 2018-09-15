#import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ Load Data from the CSV files
	arguments: names of csv files
        returns: combined dataframe
    """

    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge the messages and categories datasets using the common id
    combined_dataframe = pd.merge(messages, categories, how='left', on=['id'])
    return combined_dataframe


def clean_data(df):
    """ Clean the data
        arguments: combined dataframe
        returns: cleaned dataframe
    """

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x : x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(-1)
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
        # for some rows, 'related' column has value 2, making them 1
        categories[column] = categories[column].apply(lambda x : 0 if x == 0 else 1)

    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    # return cleaned dataframe
    return df


def save_data(df, database_filename):
    """ Save data in sqllite database
        arguments: cleaned dataframe and database file name
        returns: None
    """

    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(name='disaster_messages_table', con=engine, if_exists='replace', index=False, chunksize=100)  


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

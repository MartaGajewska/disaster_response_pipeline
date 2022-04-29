import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Loads files from specified paths and merges them using 'id' columns.
    '''
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    df = messages_df.merge(categories_df, on='id')
    return df


def change_categories_format(categories):
    '''
    Information about all categories is stored in one column
    in format "category_x-1,category_y-0, ...".
    This function transforms this column into a format where 
    each category in a single colum with 0-1 values.
    '''
    # Step 1: create a dataframe of the individual category columns
    categories_wide = categories.str.split(';', expand = True)

    # Step 2: Extract categories names to be used as column names:
    # - select the first row of the categories_wide dataframe
    row = categories_wide.iloc[0,:]
    # - use this row to extract a list of new column names
    category_colnames = [cat[0:-2] for cat in row]
    # - rename the columns of `categories_wide`
    categories_wide.columns = category_colnames

    # Step 3: convert category values to just numbers 0 or 1
    # Iterate through each column and keep only the last character of each string
    # For example, related-0 becomes 0, related-1 becomes 1
    for column in categories_wide:
        # set each value to be the last character of the string
        categories_wide[column] = categories_wide[column].str[-1]
        # convert column from string to numeric
        categories_wide[column] = pd.to_numeric(categories_wide[column])

    # Step 4: store index as a column
    categories_wide['id'] = categories_wide.index

    return categories_wide

def clean_data(df):
    '''
    Cleans df by:
        - converting categories column;
        - removing constant column ('child_alone');
        - removing duplicates.
    '''
    # Change format of categories column:
    # - convert categories to wide format
    categories_wide = change_categories_format(df['categories'])
    # - drop the original categories column from `df`
    # - drop genre column as well 
    df.drop(columns = ['categories'], inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.merge(categories_wide, on = 'id')

    # Remove constant column
    df.drop(columns = 'child_alone')
    
    # Remove duplicated rows: 
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    '''
    Saves cleaned df into sqlite database.
    '''
    # Save the clean dataset into an sqlite database
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages_with_categories', engine, index=False, if_exists = 'replace')
    pass  


def main():
    '''
    Runs data processing pipeline.
    '''
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
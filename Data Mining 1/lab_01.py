# Lab 01 - Data mining
# Luke Fox

import pandas as pd
import pymysql
import getpass
import re

importing the data
csv_file = pd.read_csv('BSCY4.csv')
connection = pymysql.connect(host='localhost', port=3308, user='root', password='ENTER_PASSWORD', db='bscy4')
sql_file = pd.read_sql('SELECT * FROM avocado', con=connection)


def main():
    clean_CSV()
    clean_SQL()
    join_data()


def clean_CSV():

    # count the different date formats
    dates = csv_file['Date']
    formats = {'.{4}-.{2}-.{2}': 0, '.{2}-.{2}-.{4}': 0, '.{2}/.{2}': 0}
    for date in dates:
        for key, val in formats.items():
            pattern = re.compile(key)
            if pattern.match(date):
                formats[key] += 1

    print('yyyy-mm-dd: {0}'
          '\ndd-mm-yy: {1}'
          '\ndd/mm: {2}'.
          format(formats.get('.{4}-.{2}-.{2}'),
                 formats.get('.{2}-.{2}-.{4}'),
                 formats.get('.{2}-.{2}-.{4}'))
          )

    # append year to dates missing year
    for index, row in dates.items():
        if len(row) == 5:
            year = int(csv_file['year'].at[index])
            dates.at[index] = row + '/{0}'.format(year)

    #  cleaning date
    print(dates.unique())
    datetime = pd.to_datetime(dates, errors='coerce')

    # filtering data by 'type' and finding different categories
    types = csv_file['type']
    print(types.value_counts())
    type_errors = types.isnull().sum()
    print('TYPES ERRORS: ', type_errors)

    for index, row in types.items():
        if row == 'Org.':
            types.at[index] = 'organic'

    # filtering data by 'Average Price'
    average_prices = csv_file['AveragePrice']
    print(average_prices.unique())
    average_prices_errors = average_prices.isna().sum()

    # count string-based representations
    count = sum(',' in str(row) for row in average_prices)
    print('STRING REPRESENTATIONS: ', count)

    # cleaning average price
    average_prices = average_prices.str.replace(',', '.').astype(float)
    average_prices_numeric = pd.to_numeric(average_prices, errors='coerce')

    # assign cleaned columns, drop null values
    csv_file['AveragePrice'] = average_prices_numeric
    csv_file['Date'] = datetime
    csv_file.dropna(inplace=True)

    # export the cleaned data
    export_csv = csv_file.to_csv('CLEANED_CSV.csv', index=None, header=True)


def clean_SQL():
    # clean region
    region = sql_file['region']

    for index, row in region.items():
        if ' ' in row:
            region.at[index] = row.replace(' ', '')
        if '-' in row:
            region.at[index] = row.replace('-', '')

    print('CLEANED: ', region.value_counts())

    # clean year
    year = sql_file['year']
    print(year.value_counts())

    for index, row in year.items():
        if len(str(row)) == 2:
            year.at[index] = '20{0}'.format(row)

    # clean type
    type = sql_file['type']
    print(type.value_counts())
    sql_file['type'] = sql_file['type'].str.lower()

    export_csv = sql_file.to_csv('CLEANED_SQL.csv', index=None, header=True)


def join_data():
    global csv_file
    global sql_file

    # clean column names from both files
    csv_file.columns = \
        csv_file.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    sql_file.columns = \
        sql_file.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

    # concatinate and export cleaned data
    joined_files = pd.concat([csv_file, sql_file], axis=1)
    export_csv = sql_file.to_csv('CONCAT.csv', index=None, header=True)


if __name__ == '__main__':
    main()

"""
--> QUESTIONS

CSV FILE:

Question - Cleanse information in the "date". Do all rows follow the same format when it comes to "date"? 
What formats are there and how many entries per each format? (1 point)

Answer - No, not every row follows the same date format, there are 3 different formats and they are 
distributes as follows: yyyy-mm-dd = 8787, dd-mm-yy = 169, dd/mm = 169

Question - Cleanse the data in the field "type". How many genuine categories are present? Do you see 
problems with how the categories represented? How many entries have errors? (1 point)

Answer - There are a total of 3 categories, but of these 3 there is a duplicate category with a different 
naming convention for 'organic'. This duplicate category is named 'Org.' There are no entries with errors.

Question - Cleanse the content of the field "average price". How many genuine missing values are there? 
How many entries have erroneous string-based representation. (1 point)

Answer - There are 20 genuine missing values, where as there is a total of 30 string representations 
(using commas insdead of dots)


SQL FILE:

Question - Cleanse the content of the field "region". What can you say about the regions represented? 
How many different regions there are? Are there problems with this variable, if yes, what are the 
problems and how many? 

Answer - The regions have quite poor/unclear naming naming conventions - for example, there are multiple 
regions in some variables like 'WestTexNewMexico', but then other areas have a vague name that may overlap 
such as 'West'. There are also naming errors. For example, Denver appears 3 times, due to some columns having 
hidden whitespaces. Denver appears 100 times for the correct column, then 50 and 19 for the whitespace columns
respectively. BaltimoreWashington also has an incorrect entry of 'Baltimore-Washington' which appears 80 times

Question - What years are represented? Describe any errors that you see in data. How many rows are affected?

Answer - There are 4 years represented - 2015 up to 2018 inclusive. There are errors though, as some rows contain
the year as 2 digits. The year '17' appears 2862, which should be named '2017'. Also, there are two variants
of 2018, one reading '2018' with 300 rows and another with '18' reading with 346 rows.

Question - Cleanse the content of the field "type". What avocado type are represented? Describe any
errors that you see. How many rows are affected? 

Answer - There are errors with the 'type' field as there exists two variants of the type 'conventional',
one with a lower-case 'c' with 8955 entries and one with an upper-case with 169 entries.


DATA CONSOLIDATION:

Question - Perform Visual Inspection of the results of the two previous imports. Are the two data 
frames suitable for consolidation? What problems do you see? Correct the problems.

Answer - These two frames are not yet suitable for consolidation as the column names from the two frames 
vary. Some have no whitespaces, others do. Some have all lowercase, others have capitalized words. In the
original CSV file, there also exists an 'Unnamed: 0' column with unrelated data. The same column does not
exist in the SQL file.

Question - What method should you use to consolidate the two frames correctly? Perform the consolidation. 

Answer - To consolidate the frames correctly, I used the 'concat' function - as once the column names had 
been fixed, they were both named identically. Because of this, I did not want duplicates of the column names
and instead wanted to only merge the data of these column names. This was where the 'concat' function was 
most suitable to use.

"""

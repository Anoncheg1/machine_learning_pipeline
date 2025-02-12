import pandas as pd
import numpy as np

def split_columns_by_type_numebers_strings(df):
    "Return numeric_columns, categorical_columns."
    return list(df.select_dtypes(include=[np.number]).columns), \
           list(df.select_dtypes(exclude=[np.number]).columns)


def split_columns_by_numeric_conversion(df):
    """1) replace empty string with NaN
       2) convert to number, errors to NaN
       3) if after 2) count(NaN) > 1) count(NaN) -> had errors
    Side effect: May modify df.
       """
    numeric_columns = []
    non_numeric_columns = []

    for column in df.columns:
        # Replace empty strings with NaN
        df[column] = df[column].replace(r'^\s*$', np.nan, regex=True)

        # Attempt to convert to numeric
        converted = pd.to_numeric(df[column], errors='coerce')

        # Check if any non-NaN values were introduced by the conversion
        if converted.isna().equals(df[column].isna()):
            numeric_columns.append(column)
        else:
            non_numeric_columns.append(column)

    return numeric_columns, non_numeric_columns

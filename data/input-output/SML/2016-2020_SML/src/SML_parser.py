import pandas as pd
import numpy as np
from typing import Dict, Optional
import os
COUNTRY_CODES = country_codes = {
    'AUS': 'Australia',
    'AUT': 'Austria',
    'BEL': 'Belgium',
    'CAN': 'Canada',
    'CHL': 'Chile',
    'COL': 'Colombia',
    'CRI': 'Costa Rica',
    'CZE': 'Czechia',
    'DNK': 'Denmark',
    'EST': 'Estonia',
    'FIN': 'Finland',
    'FRA': 'France',
    'DEU': 'Germany',
    'GRC': 'Greece',
    'HUN': 'Hungary',
    'ISL': 'Iceland',
    'IRL': 'Ireland',
    'ISR': 'Israel',
    'ITA': 'Italy',
    'JPN': 'Japan',
    'KOR': 'Korea',
    'LVA': 'Latvia',
    'LTU': 'Lithuania',
    'LUX': 'Luxembourg',
    'MEX': 'Mexico',
    'NLD': 'Netherlands',
    'NZL': 'New Zealand',
    'NOR': 'Norway',
    'POL': 'Poland',
    'PRT': 'Portugal',
    'SVK': 'Slovakia',
    'SVN': 'Slovenia',
    'ESP': 'Spain',
    'SWE': 'Sweden',
    'CHE': 'Switzerland',
    'TUR': 'Turkiye',
    'GBR': 'United Kingdom',
    'USA': 'United States',
    'ARG': 'Argentina',
    'BGD': 'Bangladesh',
    'BLR': 'Belarus',
    'BRA': 'Brazil',
    'BRN': 'Brunei Darussalam',
    'BGR': 'Bulgaria',
    'KHM': 'Cambodia',
    'CMR': 'Cameroon',
    'CHN': "China (People's Republic of)",
    'CIV': "Côte d'Ivoire",
    'HRV': 'Croatia',
    'CYP': 'Cyprus',
    'EGY': 'Egypt',
    'HKG': 'Hong Kong, China',
    'IND': 'India',
    'IDN': 'Indonesia',
    'JOR': 'Jordan',
    'KAZ': 'Kazakhstan',
    'LAO': "Lao (People's Democratic Rep.)",
    'MYS': 'Malaysia',
    'MLT': 'Malta',
    'MAR': 'Morocco',
    'MMR': 'Myanmar',
    'NGA': 'Nigeria',
    'PAK': 'Pakistan',
    'PER': 'Peru',
    'PHL': 'Philippines',
    'ROU': 'Romania',
    'RUS': 'Russian Federation',
    'SAU': 'Saudi Arabia',
    'SEN': 'Senegal',
    'SGP': 'Singapore',
    'ZAF': 'South Africa',
    'TWN': 'Chinese Taipei',
    'THA': 'Thailand',
    'TUN': 'Tunisia',
    'UKR': 'Ukraine',
    'VNM': 'Viet Nam',
    'ROW': 'Rest of the World'
}

class OECDICIOData:
    def __init__(self, file_path: str, country_codes: dict[str, str] = COUNTRY_CODES):
        """
        Initializes the OECDICIOData object by loading data from a CSV file and filling NA values with 0.

        Parameters:
        file_path (str): Path to the CSV file containing the input-output data.
        """
        self.country_codes = country_codes
        self.data = pd.read_csv(file_path, index_col=0)
        self.data = self.data.fillna(0)
        # print(f"sample of csv: {self.data.head()}")
        
        self.intermediate_use = self.extract_intermediate_use()
        self.country_intermediate_use_dict = self.extract_intermediate_use_all_countries()
        # self.country_intermediate_use_dict = {k: self.shift_first_column_to_index(v) for k, v in self.country_intermediate_use_dict.items()}
        # self.intermediate_use = self.shift_first_column_to_index(self.intermediate_use)
        self.country_data = self.extract_all_countries()
        # self.country_data = {k: self.shift_first_column_to_index(v) for k, v in self.country_data.items()}
    def shift_first_column_to_index(self, data: pd.DataFrame): 
        """
        Shifts the first column to the index of the DataFrame.
        """
        data.set_index(self.data.columns[0], inplace=True)
        data.index.name = None
        return data
    def find_first_column_with_label(self, label: str) -> Optional[int]:
        """
        Finds the first column that contains the specified label in the DataFrame using NumPy.

        Parameters:
        label (str): Label to search for in column names.

        Returns:
        Optional[int]: Index of the first column containing the label, or None if no match is found.
        """
        columns = np.array(self.data.columns).astype(str)  # Ensure all column names are strings
        matches = np.vectorize(lambda x: label in x)(columns)
        first_match_index = np.argmax(matches)

        if matches[first_match_index]:
            return first_match_index
        return None

    def find_first_row_with_label(self, label: str) -> Optional[int]:
        """
        Finds the first row that contains the specified label in the DataFrame using NumPy.

        Parameters:
        label (str): Label to search for in row values.

        Returns:
        Optional[int]: Index of the first row containing the label, or None if no match is found.
        """
        rows = np.array(self.data.index).astype(str)  # Ensure all index values are strings
        matches = np.vectorize(lambda x: label in x)(rows)
        first_match_index = np.argmax(matches)

        if matches[first_match_index]:
            return first_match_index
        return None

    def crop_dataframe(self, df: pd.DataFrame, col_label: str, row_label: str) -> pd.DataFrame:
        """
        Crops the DataFrame to remove all data before the first occurrence of specified column and row labels using NumPy.

        Parameters:
        df (pd.DataFrame): Input DataFrame to crop.
        col_label (str): Label to search for in column names.
        row_label (str): Label to search for in row values.

        Returns:
        pd.DataFrame: Cropped DataFrame.
        """
        # print(f"labels: {col_label}, {row_label}")
        col_index = self.find_first_column_with_label(col_label)
        row_index = self.find_first_row_with_label(row_label)
        # print(f"indices: col: {col_index}, row: {row_index}")
        if col_index is None or row_index is None:
            raise ValueError("Specified labels not found in DataFrame.")

        cropped_df = df.iloc[:row_index, :col_index]
        return cropped_df

    def extract_intermediate_use(self, row_label: str = 'TLS', col_label: str = 'HFCE') -> pd.DataFrame:
        """
        Extracts the intermediate use data from the input DataFrame.

        Parameters:
        row_label(str): Label to search for in column names. Defaults to 'TLS'.
        col_label (str): Label to search for in row values. Defaults to 'HFCE'.
       
        
        Returns:
        pd.DataFrame: DataFrame containing only the intermediate use data.
        """
        return self.crop_dataframe(self.data, col_label, row_label)

    def extract_intermediate_use_all_countries(self) -> Dict[str, pd.DataFrame]:   
        """
        Extracts the intermediate use data for all countries based on the first three letters of row and column labels.

        Returns:
        Dict[str, pd.DataFrame]: Dictionary containing DataFrames for each country.
        """
        country_codes = self.country_codes.keys()
        country_intermediate_use_dict = {}
        if self.intermediate_use is None:
            self.intermediate_use = self.extract_intermediate_use()
        for code in country_codes:
            country_intermediate_use_dict[code] = self.extract_by_country(code, self.intermediate_use)
        return country_intermediate_use_dict
    
    def extract_by_country(self, country_code: str, data: pd.DataFrame= None) -> pd.DataFrame:
        """
        Extracts data for a specific country based on the first three letters of row and column labels.

        Parameters:
        country_code (str): The three-letter country code to filter data by.
        data (pd.DataFrame): DataFrame to extract data from. Defaults to the main data attribute.

        Returns:
        pd.DataFrame: DataFrame containing data for the specified country.
        """
        if data is None:
            data = self.data
        country_rows = [index for index in data.index.astype(str) if index.startswith(country_code)]
        country_columns = [col for col in data.columns.astype(str) if col.startswith(country_code)]
        country_data = data.loc[country_rows, country_columns]
        return country_data

    def extract_all_countries(self) -> Dict[str, pd.DataFrame]:
        """
        Extracts data for all countries based on the first three letters of row and column labels.

        Returns:
        Dict[str, pd.DataFrame]: Dictionary containing DataFrames for each country.
        """
        country_codes = self.country_codes.keys() # Get list of country codes
        country_data_dict = {}

        for code in country_codes:
            country_data_dict[code] = self.extract_by_country(code)

        return country_data_dict

    def save_intermediate_use(self, file_path: str):
        """
        Saves the intermediate use data to a CSV file.

        Parameters:
        file_path (str): Path to the output CSV file.
        """
        self.intermediate_use.to_csv(file_path)

    def save_country_data(self, directory_path: str):
        """
        Saves each country's data to separate CSV files in the specified directory.

        Parameters:
        directory_path (str): Path to the directory where the CSV files will be saved.
        """
        for country_code, df in self.country_data.items():
            df.to_csv(f"{directory_path}/{country_code}_data.csv")
    def save_intermediate_use_country(self, directory_path: str):
        """
        Saves each country's intermediate use data to separate CSV files in the specified directory.

        Parameters:
        directory_path (str): Path to the directory where the CSV files will be saved.
        """
        for country_code, df in self.country_intermediate_use_dict.items():
            df.to_csv(f"{directory_path}/{country_code}_intermediate_use.csv")
# Example usage
# if __name__ == "__main__":
#     file_path = '/users/mf23yyt/Thesis/network_reconstruction/data/input-output/SML/2016-2020_SML/2020_SML.csv'
#     oecd_data = OECDICIOData(file_path)
#     intermediate_data = oecd_data.extract_intermediate_use()
#     country_data = oecd_data.country_data
#     oecd_data.save_intermediate_use('/path/to/save/intermediate_use.csv')
#     oecd_data.save_country_data('/path/to/save/country_data')
#     print(country_data.keys())  # List of country codes
#     print(country_data['ARG'])  # DataFrame for Argentina


# Example usage
if __name__ == "__main__":
    start_year = 2016
    end_year = 2020
    for year in range(start_year, end_year + 1):
        print(f"year: {year}")
        file_path = f'/users/mf23yyt/Thesis/network_reconstruction/data/input-output/SML/{start_year}-{end_year}_SML/{year}_SML.csv'
        print(f"sample of csv: {pd.read_csv(file_path).head()}")
        oecd_data = OECDICIOData(file_path)
        intermediate_data = oecd_data.extract_intermediate_use()
        country_data = oecd_data.country_data
        #make directory {year}_SML
        if not os.path.exists(f'/users/mf23yyt/Thesis/network_reconstruction/data/input-output/SML/{start_year}-{end_year}_SML/preprocessed/{year}_SML'):
            os.makedirs(f'/users/mf23yyt/Thesis/network_reconstruction/data/input-output/SML/{start_year}-{end_year}_SML/preprocessed/{year}_SML')
        if not os.path.exists(f'/users/mf23yyt/Thesis/network_reconstruction/data/input-output/SML/{start_year}-{end_year}_SML/preprocessed/{year}_SML/countries_intermediate_use'):
            os.makedirs(f'/users/mf23yyt/Thesis/network_reconstruction/data/input-output/SML/{start_year}-{end_year}_SML/preprocessed/{year}_SML/countries_intermediate_use')
        if not os.path.exists(f'/users/mf23yyt/Thesis/network_reconstruction/data/input-output/SML/{start_year}-{end_year}_SML/preprocessed/{year}_SML/countries'):
            os.makedirs(f'/users/mf23yyt/Thesis/network_reconstruction/data/input-output/SML/{start_year}-{end_year}_SML/preprocessed/{year}_SML/countries')
        oecd_data.save_intermediate_use(f'/users/mf23yyt/Thesis/network_reconstruction/data/input-output/SML/{start_year}-{end_year}_SML/preprocessed/{year}_SML/intermediate_use.csv')
        oecd_data.save_intermediate_use_country(f'/users/mf23yyt/Thesis/network_reconstruction/data/input-output/SML/{start_year}-{end_year}_SML/preprocessed/{year}_SML/countries_intermediate_use')
        oecd_data.save_country_data(f'/users/mf23yyt/Thesis/network_reconstruction/data/input-output/SML/{start_year}-{end_year}_SML/preprocessed/{year}_SML/countries')
        print(country_data.keys())  # List of country codes
        print(country_data['ARG'])  # DataFrame for Argentina

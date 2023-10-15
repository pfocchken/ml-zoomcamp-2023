import pandas as pd

from src.utils.file_helper import get_file_by_url


# TODO: This need to be wrapped with class - CONSTANTS are variables, etc

CATEGORICAL_COLUMNS = ['make', 'model', 'transmission_type', 'vehicle_style']
ACTIVE_COLUMNS = ["make", 'model', 'year', 'engine_hp', 'engine_cylinders', 'transmission_type', 'vehicle_style', 'highway_mpg', 'city_mpg', "msrp"]
NUMERIC_COLUMNS = ['year', 'engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg']
DATASET_URL = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv"
TARGET_NAME = "above_average"


def prepare_dataset() -> pd.DataFrame:
    # Get dataset to car_dataset file and load it to dataframe
    file_name = get_file_by_url(DATASET_URL, "cars_dataset")

    cars_full_df = pd.read_csv(file_name)

    # Prepare test data

    # TODO: Move this to separate method to src
    cars_full_df.columns = cars_full_df.columns.str.replace(" ", "_").str.lower()

    cars_full_df = cars_full_df[ACTIVE_COLUMNS]

    for na_column, na_sum in cars_full_df.isna().sum().items():
        if na_sum > 0:
            cars_full_df[na_column].fillna(0, inplace=True)

    cars_full_df.rename(columns={"msrp": "price"}, inplace=True)

    return cars_full_df


def prepare_target_column(cars_df: pd.DataFrame, drop_source: bool = False) -> pd.DataFrame:
    source_column = "price"

    mean_price = cars_df[source_column].mean()
    cars_df[TARGET_NAME] = (cars_df[source_column] > mean_price).astype('int')

    if drop_source:
        cars_df.drop(source_column, axis="columns", inplace=True)

    return cars_df




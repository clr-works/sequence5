import os.path
import pandas as pd
import expand_functions as ef
import field_15_functions as f15
import my_module_pretactical_forecasting as mm
from typing import Optional, List, NamedTuple, Tuple, Dict


verbose = True


class AirportPair(NamedTuple):
    adep: str
    ades: str


class AiracRange(NamedTuple):
    first: int
    last: int


class ColumnNames(NamedTuple):
    adep: str
    ades: str
    one: str


def myglo(rte: str, adep: str, airac: int, to_expand):
    l1 = f15.clean_expand(rte, adep, airac, to_expand)
    l2 = f15.convert_to_lat_long(l1, airac, to_expand)
    if l2 == 'ERROR!':
        return None
    return l2 if l2 is None else mm.get_traj(l2.split())


def add_normalized_route(dataset: pd.DataFrame, csv_pts=None, csv_rts=None, first_airac=None, last_airac=None) -> None:
    """Transform the clean message into an expanded set of points."""
    if not len(dataset):
        return

    to_expand = ef.get_global_expand_functions(csv_pts, csv_rts, first_airac, last_airac)

    if True:
        # Make new col of 2D routes (these will have been cleaned and expanded)
        dataset['Exp_2D_Rte'] = dataset.apply(
            lambda x: f15.clean_expand(x.CLEAN_MSG, x.FLT_DEP_AD, x.AIRAC_CYCL, to_expand), axis=1)

        dataset['LatLong_2D_Rte'] = dataset.apply(
            lambda x: f15.convert_to_lat_long(x.Exp_2D_Rte, x.AIRAC_CYCL, to_expand), axis=1)
        # Convert lat-long coords to their decimal equivalents

        dataset['LatLong_2D_Rte_dec'] = dataset['LatLong_2D_Rte'].apply(
            lambda x: None if x == 'ERROR!' else (x if x is None else mm.get_traj(x.split())))
    else:
        dataset['LatLong_2D_Rte_dec'] = dataset.apply(lambda x: myglo(x.CLEAN_MSG,
                                                                      x.FLT_DEP_AD,
                                                                      x.AIRAC_CYCL,
                                                                      to_expand), axis=1)


def get_filtered_dataset(airacs: AiracRange, columns: ColumnNames, input_path: str,
                         minimum_nb_flight: int = 20,
                         airport_pairs=None
                         ) -> Tuple[pd.DataFrame, List[AirportPair]]:
    """Recreate dataset from database exports and filter it by airac and airport pairs.
    Args:
        airacs: included range of airacs we want to consider
        columns: names of columns in imported dataset
        input_path: directory where database export are present
        minimum_nb_flight: minimum number of flights necessary to take the pair in account
        airport_pairs: list of airport pairs
    Returns:
        Dataset with all information for given airac and limited to given airport pairs, and list of airport pairs
    """
    dataset = get_dataset(airacs, input_path)
    airport_pairs_filtered = [f"{x.adep}{x.ades}" for x in airport_pairs]
    print(airport_pairs_filtered)

    dataset['newfield'] = dataset[columns.adep] + dataset[columns.ades]

    if verbose:
        print('Printing head of dataset', dataset['newfield'].head())

    # All datasets with airport filtered pairs
    dataset = dataset[dataset['newfield'].isin(airport_pairs_filtered)]

    if verbose:
        print('Records after first filter', len(dataset['newfield']))

    # Establish minimum number of flights
    counts = dataset[[columns.adep,
                      columns.ades,
                      columns.one]].groupby(by=[columns.adep, columns.ades], as_index=False).count()
    counts = counts.loc[counts[columns.one] >= minimum_nb_flight, [columns.adep, columns.ades]].reset_index()
    dataset = counts.merge(right=dataset, how='inner', on=[columns.adep, columns.ades])

    if verbose:
        print('Records after second filter', len(dataset['newfield']))
        print('# of unique airport pairs after second filter', dataset['newfield'].nunique())

    pairs = []
    for i, pair in counts.iterrows():
        pairs.append(AirportPair(pair[columns.adep], pair[columns.ades]))

    return dataset, pairs


def get_dataset(airacs: AiracRange, input_path: str):
    """Get all data in a dataset. Note that missing airac were ignored, not anymore."""
    frames = [pd.read_csv(os.path.join(input_path, f"{airac}.gz")) for airac in range(airacs.last, airacs.first-1, -1)]
    return pd.concat(frames)


def create_subsets(columns: ColumnNames, dataset: pd.DataFrame,
                   pairs: List[AirportPair]) -> Dict[AirportPair, pd.DataFrame]:
    """Split dataset per airport pair and save each resulting dataframe in a file"""
    dataset.drop(columns=['Unnamed: 0', 'Unnamed: 0_ACFT_TY'], inplace=True)
    subsets = {}
    for pair in pairs:
        subset = dataset[(dataset[columns.adep] == pair.adep) & (dataset[columns.ades] == pair.ades)]
        subsets[pair] = subset
    return subsets


def transform_all(airacs: AiracRange,
                  columns: ColumnNames,
                  minimum_nb_flight: int,
                  input_path: Optional[str] = None,
                  airport_pairs=None,
                  csv_pts=None,
                  csv_rts=None,
                  first_airac=None,
                  last_airac=None) -> Dict[AirportPair, pd.DataFrame]:
    """Read each airac file stored at 'path_raw' and will produce one file per
    relevant airport pair. It will filter by relevant airport pair and will expand the cleaned
    message filed by the AO (message without sid and star) to an expanded 2D route.
    Additionally, it will merge with Predict dataset to get predicted values.
    Returns:
        dataset of flights per airport pair
    """
    input_path = '/storage/nmlab/sequence_5_R/airacs' if input_path is None else input_path
    # airports_filename = 'airport_pairs_min400km_LT50flts.txt' if airports_filename is None else airports_filename

    dataset, pairs = get_filtered_dataset(airacs,
                                          columns,
                                          input_path=input_path,
                                          minimum_nb_flight=minimum_nb_flight,
                                          airport_pairs=airport_pairs)

    add_normalized_route(dataset, csv_pts, csv_rts, first_airac, last_airac)
    return create_subsets(columns, dataset, pairs)
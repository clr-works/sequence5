from transform import AiracRange
from helpers import read_dataframe_from_file
import datetime
import os
import logging
from typing import List, Optional, NamedTuple
from pandarallel import pandarallel  # pip install pandarallel
import json
from steps.io.subsets import SubsetsIO
from steps.io.predictions import PredictionsIO
from steps.io.finals import FinalsIO
from steps.prepare_step import PrepareStep
from steps.training_step import TrainingStep
from steps.testing_step import TestingStep
from steps.predict_data_step import PredictDataStep
from steps.string_comparison_step import StringComparisonStep
from steps.area_comparison_step import AreaComparisonStep
from steps.distance_comparison_step import DistanceComparisonStep
from my_types import AirportPair, DataLakeConfiguration
from datalake import DataLake
from transform import get_pairs

pandarallel.initialize(use_memory_fs = False)


class AirportPairsConfiguration(NamedTuple):
    filename: Optional[str] = None
    airacs: Optional[AiracRange] = None
    minimum_nb_flights: Optional[int] = None
    distance_minimum: Optional[int] = None  # km
    limit: Optional[int] = None


class Pipeline:
    def __init__(self, origin: str, first_airac=None, last_airac=None, datastore: str = None,
                 models_path: str = None, predictions_path: str = None, results_path: str = None):
        """
        Args:
            origin ():
            first_airac ():
            last_airac ():
            datastore ():
            models_path ():
            predictions_path ():
            results_path ():
        """
        self._place = origin
        self._datastore = os.path.join(self._place, 'datastore') if datastore is None else datastore
        self.subsets_path = os.path.join(self._datastore, 'tr')
        self.predict_data = os.path.join(self._datastore, 'PREDICT')
        self.models_properties = os.path.join(self._place, 'results') if results_path is None else results_path
        self.predictions = os.path.join(self._place, 'predictions') if predictions_path is None else predictions_path
        self.models = os.path.join(self._place, 'models') if models_path is None else models_path
        self.final_path = os.path.join(self._place, 'final')
        self.areas_path = os.path.join(self._place, 'areas')
        self.kpi = os.path.join(self._place, 'kpi')

        data = os.path.join(self._place, 'data')
        self.flight_scores = os.path.join(data, 'DistScores')
        self.training_for_filter = os.path.join(self._place, 'training_for_filter')
        self.predictions_for_filter = os.path.join(self._place, 'predictions_for_filter')
        self.airacs_ml_training = []
        self.airacs_ml_testing = []
        self._input_pairs = None
        self._reference_airac = 461
        self._writing = True
        self.datalake_conf = DataLakeConfiguration(os.path.join(self._datastore, 'pts.gz'),
                                                   os.path.join(self._datastore, 'rts.gz'),
                                                   os.path.join(self._datastore, 'airacs'),
                                                   os.path.join(data, 'test_data_07072021.gz'),
                                                   first_airac=first_airac,
                                                   last_airac=last_airac)
        DataLake().set_configuration(self.datalake_conf)
        self.pairs_conf = AirportPairsConfiguration(filename=None,
                                                    airacs=AiracRange(self._reference_airac, self._reference_airac),
                                                    minimum_nb_flights=1,
                                                    distance_minimum=1,
                                                    limit=None)
        self._data = {}
        self._steps = {'prepare PREDICT'    : PredictDataStep(self),
                       'prepare subset'     : PrepareStep(self),
                       'training'           : TrainingStep(self),
                       'testing'            : TestingStep(self),
                       'string comparison'  : StringComparisonStep(self),
                       'area comparison'    : AreaComparisonStep(self),
                       'distance comparison': DistanceComparisonStep(self)}

    def step(self, name):
        """Return existing step instance linked with given name."""
        if name in self._steps:
            return self._steps[name]

    def set_pairs_configuration(self, configuration: AirportPairsConfiguration):
        self.pairs_conf = configuration

    def set_list_pairs(self, pairs: List[AirportPair]):
        self._input_pairs = pairs

    def set_data(self, name, data):
        """Add data as named property in stash."""
        self._data[name] = data

    def get_data(self, name):
        """Return existing data in stash linked with given name."""
        if name in self._data:
            return self._data[name]

    def reset_data(self):
        """Empty stash."""
        self._data = {}

    def set_writing(self, b: bool = True):
        """Set flag to decide to write data."""
        self._writing = b

    @property
    def writing(self) -> bool:
        """Flag to decide to write data generated as files (stash always filled)."""
        return self._writing is True

    def get_pairs(self) -> List[AirportPair]:
        """Return list of airport pairs used as input.
        if not provided then used file containing it (txt or json)"""
        if self._input_pairs is None:
            filename = self.pairs_conf.filename
            if filename is not None:
                filename = os.path.join(self._place, filename)
                if filename.endswith('.txt'):
                    with open(filename) as r:
                        airport_pairs = [line.rstrip() for line in r]
                    airport_pairs = [eval(item) for item in airport_pairs]
                    self._input_pairs = [AirportPair(value[0], value[1]) for value in airport_pairs]
                elif filename.endswith('.json'):
                    with open(filename) as f:
                        data = json.load(f)
                        self._input_pairs = [AirportPair(*x) for x in data["pairs"]]
            else:
                # fetch pairs from datalake
                self._input_pairs = get_pairs(airacs=self.pairs_conf.airacs,
                                              minimum_nb_flights=self.pairs_conf.minimum_nb_flights,
                                              minimum_distance=self.pairs_conf.distance_minimum,
                                              limit=self.pairs_conf.limit)
        return self._input_pairs

    def get_training_airacs(self) -> AiracRange:
        # should move AiracRange everywhere instead of List[int] except cli
        return AiracRange(self.airacs_ml_training[0], self.airacs_ml_training[1])

    def get_testing_airacs(self) -> AiracRange:
        # should move AiracRange everywhere instead of List[int] except cli
        return AiracRange(self.airacs_ml_testing[0], self.airacs_ml_testing[1])

    def set_airacs_training(self, airacs: List[int]):
        self.airacs_ml_training = airacs

    def set_airacs_testing(self, airacs: List[int]):
        self.airacs_ml_testing = airacs

    def set_reference_airac(self, airac: int):
        self._reference_airac = airac

    def missing_finals(self):
        return [p for p in self.get_pairs() if
                not os.path.isfile(os.path.join(self.final_path, f"final_res{p.adep}_{p.ades}.parquet.gz"))]

    def _get_predictions(self, pairs=None, airacs_training=None, airacs_testing=None):
        return PredictionsIO(self).get(pairs, airacs_training, airacs_testing)

    def get_subsets_with_prediction(self, pairs=None, airacs_training=None, airacs_testing=None):
        subsets = self.get_data('subsets')
        io_predictions = PredictionsIO(self)
        if subsets is None:
            io_subsets = SubsetsIO(self)
            pairs = self.get_pairs() if pairs is None else pairs
            airacs_training = self.airacs_ml_training if airacs_training is None else airacs_training
            airacs_testing = self.airacs_ml_testing if airacs_testing is None else airacs_testing
            pairs_s = [p for p in pairs if io_predictions.exists(p, airacs_training, airacs_testing)]
            subsets = {p: read_dataframe_from_file(io_subsets.full_name(p)) for p in pairs_s}
        return subsets

    def get_finals(self):
        return FinalsIO(self).get()

    def execute(self, command, step_name, **kwargs):
        """Execute command on step with given name if exists."""
        step = self.step(step_name)
        if step is None:
            logging.info(f"no step registered under name {step_name}")
            return
        if command == 'reset':
            step.reset()
        elif command == 'deep':
            return step.deep(**kwargs)
        elif command == 'shallow':
            return step.shallow(**kwargs)
        elif command == 'deep_from':
            return step.deep_from(**kwargs)
        elif command == 'shallow_from':
            return step.shallow_from(**kwargs)
        else:
            logging.info(f"no command defined with name {command}")

    def reset_report(self):
        pass

    def deep_report(self):
        self.reset_report()
        return self.shallow_report()

    def shallow_report(self):
        return self.report()

    def shallow_from_report(self,
                            minimum_nb_flight: int,
                            airacs_ml_training: Optional[List[int]] = None,
                            airacs_ml_testing: Optional[List[int]] = None):
        self.execute('shallow_from', 'distance comparison', minimum_nb_flight=minimum_nb_flight,
                     airacs_ml_training=airacs_ml_training,
                     airacs_ml_testing=airacs_ml_testing)
        return self.shallow_report()

    def deep_from_report(self,
                         minimum_nb_flight: int,
                         airacs_ml_training: Optional[List[int]] = None,
                         airacs_ml_testing: Optional[List[int]] = None):
        self.execute('deep_from', 'distance comparison', minimum_nb_flight=minimum_nb_flight,
                     airacs_ml_training=airacs_ml_training,
                     airacs_ml_testing=airacs_ml_testing)
        return self.deep_report()

    def report_input(self) -> List[str]:
        # list of given AD pairs (and algo to find them, maybe) anf airacs
        pairs_as_txt = sorted([f"{p}" for p in self.get_pairs()])
        txt = [f"AD pairs as input: [{', '.join(pairs_as_txt)}]"]
        training_range = self.get_training_airacs()
        training_airacs_txt = f"[{int(training_range.first):03}-{int(training_range.last):03}]"
        txt.append(f"Training airacs: {training_airacs_txt}")
        testing_range = self.get_testing_airacs()
        testing_airacs_txt = f"[{int(testing_range.first):03}-{int(testing_range.last):03}]"
        txt.append(f"Testing airacs:  {testing_airacs_txt}")
        return txt

    def report(self) -> List[str]:
        txt = [f"Experiment Report {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
        txt.extend(self.report_input())
        return txt


if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s %(message)s', filename='pipeline.log', level=logging.DEBUG)
    logging.info('started')
    time = datetime.datetime.now()
    pipeline = Pipeline(origin='/home/cdsw/origin',
              first_airac=459,
                        last_airac=462)
    pipeline.set_reference_airac(461)
    # pipeline.set_pairs_configuration(AirportPairsConfiguration(filename='airport_pairs_top100.json'))
    # pipeline.set_pairs_configuration(AirportPairsConfiguration(airacs=AiracRange(458, 459),
    # minimum_nb_flights=20, distance_minimum=400, limit=10))

    pipeline.set_list_pairs([
        AirportPair("EGLL", "LSGG"), AirportPair("LSGG", "EGLL"), AirportPair("EGLL", "EDDM"),
                                AirportPair("LOWW", "EDDM"),
                                AirportPair("GCLP", "LEMD"), AirportPair("EGLL", "LEMD"), AirportPair("ENTC", "ENGM"),
                                AirportPair("LEBL", "EGKK"),
                                AirportPair("LSZH", "EGLL"), AirportPair("GCXO", "LEMD")
    ])

    pipeline.set_airacs_training([459, 460])
    pipeline.set_airacs_testing([461, 462])
    pipeline.set_writing(True)
    # pipeline.execute('shallow', 'prepare PREDICT', airacs_testing = AiracRange(461, 462))
    #pipeline.execute('deep', 'prepare subset', airacs=AiracRange(459, 462), minimum_nb_flight=20)
    #pipeline.execute('deep', 'training')
    #pipeline.execute('deep', 'testing')
    # pipeline.execute('deep', 'testing')
    # pipeline.execute('deep', 'string comparison')
    pipeline.execute('shallow', 'area comparison', airacs=AiracRange(461, 462))
    # pipeline.execute('deep', 'distance comparison')
    # pipeline.execute('deep_from', 'distance comparison')
    # pipeline.execute('shallow_from', 'distance comparison')
    #logging.info("\n".join(pipeline.deep_from_report(minimum_nb_flight=20)))
    new_time = datetime.datetime.now()
    print(pipeline.get_pairs())
    print("duration", (new_time - time))
    logging.info('finished')
    
    


    print("checking changes in new branch")

    # pre-requisites:
    # pts.gz csv compressed file containing points coordinates for airacs
    # rts.gz csv compressed file containing routes definition for airacs
    # collection of files containing flights, one per airac, <airac number>.gz
    # airport_pairs.txt contains list of airport pairs we would like to deal with
    # eur_airports_expanded.txt contains coordinates of airports
    # TODO pts.gz should contain what eur_airports_expanded.txt provides
Footer

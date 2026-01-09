from dataclasses import dataclass

import logging

import pandas as pd

import trd_interface
import population_control
import aws_interface
import io_utils

logger = logging.getLogger(__name__)


@dataclass
class parameterScreen:
    population_ranges: dict
    username: str
    password: str
    road_course: bool

    def __post_init__(self):
        self._population_control = population_control.Population(
            100 * len(self.population_ranges),
            population_parameters=self.population_ranges,
        )
        self._trd = trd_interface.TrdInterface(
            username=self.username, password=self.password
        )
        self._aws = aws_interface.AwsInterface()
        self.population = self._population_control.create_population()

    def parse_population(self, road_course: bool = False):
        set1_cols = [x for x in self.population.columns if "set1" in x]
        set2_cols = [x for x in self.population.columns if "set2" in x]

        padding_matrix = [1] * 25

        set1_scale = self.population[set1_cols].values.tolist()
        set1_scale = [x + padding_matrix for x in set1_scale]
        set2_scale = self.population[set2_cols].values.tolist()
        set2_scale = [x + padding_matrix for x in set2_scale]

        combined_scaling = [[x, y] for x, y in zip(set1_scale, set2_scale)]

        combined_df = pd.DataFrame(combined_scaling)
        combined_df.columns = ["set1", "set2"]
        print(combined_df)

        if road_course:
            combined_df.columns = [
                "vehicle.tires.lf.scaling",
                "vehicle.tires.lr.scaling",
            ]
            combined_df["vehicle.tires.rf.scaling"] = combined_df[
                "vehicle.tires.lf.scaling"
            ]
            combined_df["vehicle.tires.rr.scaling"] = combined_df[
                "vehicle.tires.lr.scaling"
            ]
        else:
            combined_df.columns = [
                "vehicle.tires.lf.scaling",
                "vehicle.tires.rf.scaling",
            ]
            combined_df["vehicle.tires.lr.scaling"] = combined_df[
                "vehicle.tires.lf.scaling"
            ]
            combined_df["vehicle.tires.rr.scaling"] = combined_df[
                "vehicle.tires.rf.scaling"
            ]

        print(combined_df)
        self.combined_df = combined_df

    def send_scaling(
        self, payload_path: str, sweep_directory: str, download_directory: str
    ):
        print("Getting Payload")
        payload = io_utils.get_json_payload(payload_path)

        print("placing cache")
        cache_uuid = self._trd.place_apex_cache(
            payload=payload, sweep_directory=sweep_directory
        )

        logger.info(f"    Cache: {cache_uuid} | Path: {sweep_directory}")

        print("Submitting Sweep")
        sent_runs = self._trd.submit_sweep(
            unique_id=cache_uuid,
            population=self.combined_df,
            road_course=self.road_course,
        )

        _, output_path = self._aws.wait_on_aws_return(
            cache=cache_uuid,
            s3_directory=sweep_directory,
            download_dir=download_directory,
        )

        return output_path

    def load_sweep_data(self, data_path: str):
        df = pd.read_csv(data_path, delimiter=";", index_col="sweep_id")
        df.columns = df.columns.str.lower()
        df_axis = [int(x.split("_")[0]) - 2 for x in df.index]
        df.index = df_axis
        df = df.sort_index()

        df = df.drop(index=-1, errors="ignore")

        self.sens_results = df

        print(df)

        return df
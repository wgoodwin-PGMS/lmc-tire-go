import os
import json
import logging
import uuid

import pandas as pd
import numpy as np
import orjson

import aws_interface
import trd_interface
import population_control

from logging_config import setup_logging

import parameter_screen


def main():
    logger = logging.getLogger(__name__)
    setup_logging()
    UN = "wgoodwin@legacymotorclub.com"
    PW = "G00dwin!!!"

    importer = aws_interface.AwsInterface()
    trd = trd_interface.TrdInterface(username=UN, password=PW)

    road_course = False

    BUCKET_NAME = "lmc-bayesian-optimizer"

    metric_list = [
        # Driver/car Inputs
        "steering_wheel_angle",
        "throttle_proportion",
        "brake_mastcyl_pressure_front",
        # Saturations
        "avl_axle_sat_combined_front",
        "avl_axle_sat_combined_rear",
        # Errors
        "lateral_gap",
        "heading_error",
        "DemandSpeed_error",
    ]

    print(os.getcwd())

    TRACK_DATA = importer.get_s3_object(
        bucket_name=BUCKET_NAME,
        local_directory="./optimizer",
        s3_file_name="20251031_NCS_R36_PHO_42_P1_R04_154051.csv",
    )

    EXE_PLAN = importer.get_s3_object(
        bucket_name=BUCKET_NAME,
        local_directory="./optimizer",
        s3_file_name="plan_dump.json",
    )
    segments = trd.get_segments(EXE_PLAN)
    print(segments)

    print(TRACK_DATA)
    df = pd.read_csv(TRACK_DATA, skiprows=14)
    df = df.dropna(how="all").iloc[1:].reset_index(drop=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    df_cols = [str(col).lower() for col in df]
    df.columns = df_cols
    print(df)

    ecu_data = {}
    for segment, limits in segments.items():
        start = limits[0]
        end = limits[1]

        if start < end:
            df_filtered = df[df["distance"].between(start, end, inclusive="both")]
            ecu_data[f"{segment}_steering_wheel_angle_min"] = [
                df_filtered["asteering"].min()
            ]
            ecu_data[f"{segment}_steering_wheel_angle_max"] = [
                df_filtered["asteering"].max()
            ]
            ecu_data[f"{segment}_steering_wheel_angle_mean"] = [
                df_filtered["asteering"].mean()
            ]

    ecu_df = pd.DataFrame(ecu_data).round(3)
    print(ecu_df)

    metric_type = ["min", "max", "mean"]
    output_channels = {}
    metric_count = 0
    for metric in metric_list:
        for stat in metric_type:
            output_channels[str(metric_count)] = {
                "channel": metric.lower(),
                "metric": stat.upper(),
                "name": f"{metric}_{stat}".lower(),
            }
            metric_count += 1

    json_sequence, csv_sequence = trd.batch_to_sweep_convert(
        payload_path=EXE_PLAN, metric_dict=output_channels
    )

    print(EXE_PLAN)

    # Get baseline scalings
    base_lf = trd.get_base_scalings(EXE_PLAN, "lf")
    base_lr = trd.get_base_scalings(EXE_PLAN, "lr")
    base_rf = trd.get_base_scalings(EXE_PLAN, "rf")
    base_rr = trd.get_base_scalings(EXE_PLAN, "rr")

    if road_course:
        set1 = np.mean([base_lf, base_rf], axis=0)
        set2 = np.mean([base_lr, base_rr], axis=0)
    else:
        set1 = np.mean([base_lf, base_lr], axis=0)
        set2 = np.mean([base_rf, base_rr], axis=0)

    print(set1)
    print(set2)

    ranges = {
        # Set 1, LS or Front
        "set1_S1": (0.50, 1.93),  # S1
        "set1_S2": (0.61, 1.12),
        "set1_S3": (0.08, 1.48),
        "set1_S4": (set1[3], set1[3] + 1e-6),  # IA Sens
        "set1_S5": (set1[4], set1[4] + 1e-6),  # IA Offset # S5
        "set1_S6": (set1[5], set1[5] + 1e-6),  # P Sens
        "set1_S7": (set1[6], set1[6] + 1e-6),  # P offset
        "set1_S8": (0.73, 1.20),
        "set1_S9": (0.68, 1.37),
        "set1_S10": (0.56, 1.41),  # S10
        "set1_S11": (0.28, 2.40),
        "set1_S12": (set1[11], set1[11] + 1e-6),  # P Sens
        "set1_S13": (-0.83, 2.00),
        "set1_S14": (0.52, 1.08),
        "set1_S15": (0.50, 5.73),  # S15
        "set1_S16": (set1[15], set1[15] + 1e-6),  # IA Sens
        "set1_S17": (set1[16], set1[16] + 1e-6),  # P Sens
        "set1_S18": (0.43, 1.40),
        "set1_S19": (0.44, 1.72),
        "set1_S20": (0.20, 1.90),  # S20
        "set1_S21": (0.01, 3.52),
        "set1_S22": (0.56, 1.13),
        "set1_S23": (0.30, 5.00),
        "set1_S24": (0.70, 5.30),
        "set1_S25": (0.81, 4.70),  # S25
        # Set 1, LS or Front
        "set2_S1": (0.50, 1.93),  # S1
        "set2_S2": (0.61, 1.12),
        "set2_S3": (0.08, 1.48),
        "set2_S4": (set2[3], set2[3] + 1e-6),  # IA Sens
        "set2_S5": (set2[4], set2[4] + 1e-6),  # IA Offset # S5
        "set2_S6": (set2[5], set2[5] + 1e-6),  # P Sens
        "set2_S7": (set2[6], set2[6] + 1e-6),  # P offset
        "set2_S8": (0.73, 1.20),
        "set2_S9": (0.68, 1.37),
        "set2_S10": (0.56, 1.41),  # S10
        "set2_S11": (0.28, 2.40),
        "set2_S12": (set2[11], set2[11] + 1e-6),  # P Sens
        "set2_S13": (-0.83, 2.00),
        "set2_S14": (0.52, 1.08),
        "set2_S15": (0.50, 5.73),  # S15
        "set2_S16": (set2[15], set2[15] + 1e-6),  # IA Sens
        "set2_S17": (set2[16], set2[16] + 1e-6),  # P Sens
        "set2_S18": (0.43, 1.40),
        "set2_S19": (0.44, 1.72),
        "set2_S20": (0.20, 1.90),  # S20
        "set2_S21": (0.01, 3.52),
        "set2_S22": (0.56, 1.13),
        "set2_S23": (0.30, 5.00),
        "set2_S24": (0.70, 5.30),
        "set2_S25": (0.81, 4.70),  # S25
    }

    temp_uuid = str(uuid.uuid4())
    seed = 84

    phase_one = parameter_screen.parameterScreen(
        population_ranges=ranges, username=UN, password=PW, road_course=road_course, seed=seed
    )
    print(phase_one.population)
    phase_one.parse_population(road_course=road_course)
    # csv_path = phase_one.send_scaling(
    #     EXE_PLAN, f"tire_test/{temp_uuid}", f"./optimizer/{temp_uuid}/"
    # )
    csv_path = "optimizer/15022ecc-4250-40dc-9117-3acd8c46732f/sweep_results.csv"
    
    phase_one.load_sweep_data(csv_path)
    phase_one.evaluate_results()
    phase_one.pairwise_dependency_analysis(bins=10)
    phase_one.plot_interaction("set2_S2", "set2_S11", 10)
    phase_one.plot_interaction("set2_S2", "set2_S20", 10)
    phase_one.plot_interaction("set2_S13", "set2_S2", 10)
    


if __name__ == "__main__":
    main()

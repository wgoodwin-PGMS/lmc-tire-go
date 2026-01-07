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
    seed: int

    def __post_init__(self):
        self._population_control = population_control.Population(
            100 * len(self.population_ranges),
            population_parameters=self.population_ranges,
            seed=self.seed,
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

    def evaluate_results(self):
        # Get successful population
        successful_runs = self.sens_results.index.to_list()
        self.successful_pop = self.population.loc[successful_runs]
        
        # Compare sampled vs feasible ranges
        analysis = []
        
        for param in self.population.columns:
            sampled_min = self.population_ranges[param][0]
            sampled_max = self.population_ranges[param][1]
            sampled_range = sampled_max - sampled_min
            
            # Skip pinned parameters (1e-6 range)
            if sampled_range < 1e-5:
                continue
            
            feasible_min = self.successful_pop[param].min()
            feasible_max = self.successful_pop[param].max()
            feasible_range = feasible_max - feasible_min
            
            # Calculate dead zones
            dead_zone_below = feasible_min - sampled_min
            dead_zone_above = sampled_max - feasible_max
            
            utilization = feasible_range / sampled_range
            
            analysis.append({
                'parameter': param,
                'sampled_min': sampled_min,
                'sampled_max': sampled_max,
                'feasible_min': round(feasible_min, 3),
                'feasible_max': round(feasible_max, 3),
                'dead_below': round(dead_zone_below, 3),
                'dead_above': round(dead_zone_above, 3),
                'utilization': round(utilization, 3),
                'n_success': len(self.successful_pop),
            })
        
        self.marginal_analysis = pd.DataFrame(analysis)
        self.marginal_analysis = self.marginal_analysis.sort_values('utilization')
        
        print("\n=== MARGINAL ANALYSIS (sorted by utilization) ===\n")
        print(self.marginal_analysis.to_string(index=False))
        
        # Flag high-impact parameters
        constrained = self.marginal_analysis[self.marginal_analysis['utilization'] < 0.7]
        print(f"\n=== PARAMETERS TO CONSTRAIN ({len(constrained)}) ===\n")
        print(constrained.to_string(index=False))
        
        return self.marginal_analysis

    def pairwise_dependency_analysis(self, bins=5, min_cell_count=3):
        """
        Analyze pairwise failure patterns.
        
        For each pair of active parameters, bin into a grid and calculate
        failure rate per cell. Flag pairs with strong interaction effects.
        """
        # Get active parameters (skip pinned ones)
        active_params = [
            p for p in self.population.columns
            if (self.population_ranges[p][1] - self.population_ranges[p][0]) > 1e-5
        ]
        
        # Label all runs as success/failure
        self.population['success'] = self.population.index.isin(
            self.sens_results.index.to_list()
        ).astype(int)
        
        pair_analysis = []
        interaction_matrices = {}
        
        n_pairs = len(active_params) * (len(active_params) - 1) // 2
        print(f"Analyzing {n_pairs} parameter pairs...")
        
        for i, param1 in enumerate(active_params):
            for param2 in active_params[i+1:]:
                
                # Bin both parameters
                self.population['bin1'] = pd.cut(
                    self.population[param1], 
                    bins=bins, 
                    labels=False
                )
                self.population['bin2'] = pd.cut(
                    self.population[param2], 
                    bins=bins, 
                    labels=False
                )
                
                # Calculate success rate per cell
                grid = self.population.groupby(['bin1', 'bin2']).agg(
                    total=('success', 'count'),
                    successes=('success', 'sum')
                ).reset_index()
                
                grid['success_rate'] = grid['successes'] / grid['total']
                
                # Measure interaction strength
                # High variance in success rate across cells = strong interaction
                cells_with_data = grid[grid['total'] >= min_cell_count]
                
                if len(cells_with_data) < 4:
                    continue
                
                success_rate_variance = cells_with_data['success_rate'].var()
                success_rate_range = (
                    cells_with_data['success_rate'].max() - 
                    cells_with_data['success_rate'].min()
                )
                
                # Check for "death zones" - cells with 0% success rate
                death_zones = len(cells_with_data[cells_with_data['success_rate'] == 0])
                # Check for "safe zones" - cells with high success rate
                safe_zones = len(cells_with_data[cells_with_data['success_rate'] > 0.1])
                
                pair_analysis.append({
                    'param1': param1,
                    'param2': param2,
                    'success_rate_variance': round(success_rate_variance, 4),
                    'success_rate_range': round(success_rate_range, 3),
                    'death_zones': death_zones,
                    'safe_zones': safe_zones,
                    'cells_analyzed': len(cells_with_data),
                })
                
                # Store matrix for visualization of interesting pairs
                if success_rate_range > 0.05:  # Threshold for "interesting"
                    pivot = grid.pivot(
                        index='bin1', 
                        columns='bin2', 
                        values='success_rate'
                    )
                    interaction_matrices[(param1, param2)] = pivot
        
        # Clean up temp columns
        self.population.drop(columns=['success', 'bin1', 'bin2'], inplace=True)
        
        self.pair_analysis = pd.DataFrame(pair_analysis)
        self.pair_analysis = self.pair_analysis.sort_values(
            'success_rate_variance', 
            ascending=False
        )
        self.interaction_matrices = interaction_matrices
        
        print("\n=== PAIRWISE INTERACTIONS (sorted by variance) ===\n")
        print(self.pair_analysis.head(20).to_string(index=False))
        
        return self.pair_analysis, self.interaction_matrices
    
    def plot_interaction(self, param1, param2, bins=5):
        """
        Heatmap of success rate for a parameter pair.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if (param1, param2) in self.interaction_matrices:
            matrix = self.interaction_matrices[(param1, param2)]
        elif (param2, param1) in self.interaction_matrices:
            matrix = self.interaction_matrices[(param2, param1)].T
            param1, param2 = param2, param1
        else:
            print(f"No interaction data for {param1}, {param2}")
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(matrix.values, cmap='RdYlGn', vmin=0, vmax=0.15)
        
        # Labels
        ax.set_xlabel(param2)
        ax.set_ylabel(param1)
        ax.set_title(f'Success Rate: {param1} vs {param2}')
        
        # Bin edge labels
        p1_edges = np.linspace(
            self.population_ranges[param1][0],
            self.population_ranges[param1][1],
            bins + 1
        )
        p2_edges = np.linspace(
            self.population_ranges[param2][0],
            self.population_ranges[param2][1],
            bins + 1
        )
        
        ax.set_yticks(range(bins))
        ax.set_yticklabels([f'{p1_edges[i]:.2f}-{p1_edges[i+1]:.2f}' for i in range(bins)])
        ax.set_xticks(range(bins))
        ax.set_xticklabels([f'{p2_edges[i]:.1f}-{p2_edges[i+1]:.1f}' for i in range(bins)], rotation=45)
        
        # Annotate cells with success rate
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.0%}', ha='center', va='center', fontsize=9)
        
        plt.colorbar(im, label='Success Rate')
        plt.tight_layout()
        plt.savefig(f'interaction_{param1}_{param2}.png', dpi=150)
        plt.show()
        
        return fig
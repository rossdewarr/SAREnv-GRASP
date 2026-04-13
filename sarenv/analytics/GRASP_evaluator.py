# sarenv/analytics/grasp_evaluator.py
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

import sarenv
from sarenv.analytics import metrics
from sarenv.analytics.GRASP_path import generate_grasp_path
from sarenv.utils import geo
from sarenv.utils.logging_setup import get_logger
from sarenv.utils.plot import plot_single_evaluation_results

log = get_logger()



class PathGeneratorConfig:
   

    def __init__(self, num_drones: int, budget: float, **kwargs):
        self.num_drones = num_drones
        self.budget = budget
        self.fov_degrees = kwargs.pop('fov_degrees', 45.0)
        self.altitude_meters = kwargs.pop('altitude_meters', 80.0)
        self.overlap_ratio = kwargs.pop('overlap_ratio', 0)
        self.path_point_spacing_m = kwargs.pop('path_point_spacing_m', 10.0)
        self.transition_distance_m = kwargs.pop('transition_distance_m', 50.0)
        self.pizza_border_gap_m = kwargs.pop('pizza_border_gap_m', 15.0)
        
        # Store any additional parameters
        self.additional_params = kwargs

    def get_params_dict(
        self,
        center_x: float,
        center_y: float,
        max_radius: float,
        probability_map: np.ndarray | None,
        bounds: tuple[float, float, float, float] | None,
    ) -> dict:
        """Generate complete parameter dictionary for path generation."""
        params = {
            'center_x': center_x,
            'center_y': center_y,
            'max_radius': max_radius,
            'probability_map': probability_map,
            'bounds': bounds,
            'num_drones': self.num_drones,
            'budget': self.budget,
            'fov_deg': self.fov_degrees,
            'altitude': self.altitude_meters,
            'overlap': self.overlap_ratio,
            'path_point_spacing_m': self.path_point_spacing_m,
            'transition_distance_m': self.transition_distance_m,
            'border_gap_m': self.pizza_border_gap_m,
        }
        
        params.update(self.additional_params)
        return params


class PathGenerator:


    def __init__(
        self, 
        name: str, 
        func, 
        path_generator_config: PathGeneratorConfig, 
        description: str = ""
    ):

        self.name = name
        self.func = func
        self.description = description
        self.path_generator_config = path_generator_config

    def __call__(
        self,
        center_x: float,
        center_y: float,
        max_radius: float,
        probability_map: np.ndarray | None = None,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> list[LineString]:
        params = self.path_generator_config.get_params_dict(
            center_x=center_x,
            center_y=center_y,
            max_radius=max_radius,
            probability_map=probability_map,
            bounds=bounds,
        )
        
        return self.func(**params)


def get_grasp_path_generator(config: PathGeneratorConfig) -> dict[str, PathGenerator]:
    return {
        "GRASP": PathGenerator(
            name="GRASP",
            func=generate_grasp_path,
            path_generator_config=config,
            description="GRASP-based coverage path"
        ),
    }


class GraspEvaluator:

    def __init__(
        self,
        dataset_directory="sarenv_dataset",
        evaluation_sizes=None,
        num_drones=1,
        num_lost_persons=100,
        budget=100_000,
        **kwargs
    ):
        """
        Initialize the GraspEvaluator.

        Args:
            dataset_directory: Path to the sarenv dataset
            evaluation_sizes: List of dataset sizes to evaluate
            num_drones: Number of drones to simulate
            num_lost_persons: Number of victim locations to generate
            budget: Budget constraint for path generation
            **kwargs: Additional configuration parameters
        """
        self.dataset_directory = dataset_directory
        self.evaluation_sizes = evaluation_sizes or ["small", "medium", "large"]
        self.num_victims = num_lost_persons
        self.num_drones = num_drones
        self.budget = budget
        self.generated_paths = {}
        
        # Create path generator configuration
        self.path_generator_config = PathGeneratorConfig(
            num_drones=self.num_drones, 
            budget=self.budget,
            **kwargs.copy()
        )
        self.path_generators = kwargs.get("path_generators")

        self.loader = sarenv.DatasetLoader(dataset_directory=self.dataset_directory)
        self.environments = {}
        self.results = None
        self.time_series_data = {}

        # Set up path generators
        if self.path_generators is None:
            self.path_generators = get_grasp_path_generator(self.path_generator_config)
        else:
            wrapped_generators = {}
            for name, generator in self.path_generators.items():
                if isinstance(generator, PathGenerator):
                    wrapped_generators[name] = generator
                else:
                    wrapped_generators[name] = PathGenerator(
                        name=name,
                        func=generator,
                        path_generator_config=self.path_generator_config,
                        description=f"Custom generator: {name}"
                    )
            self.path_generators = wrapped_generators

        self.load_datasets()

    def load_datasets(self):
        """Load all specified datasets and generate static victim locations."""
        log.info(f"Loading datasets for sizes: {self.evaluation_sizes}")
        for size in self.evaluation_sizes:
            item = self.loader.load_environment(size)

            if not item:
                log.warning(f"Could not load data for size '{size}'. Skipping.")
                continue

            data_crs = geo.get_utm_epsg(item.center_point[0], item.center_point[1])
            victim_generator = sarenv.LostPersonLocationGenerator(item)
            victim_points = [
                p
                for p in (
                    victim_generator.generate_location()
                    for _ in range(self.num_victims)
                )
                if p
            ]
            victims_gdf = (
                gpd.GeoDataFrame(geometry=victim_points, crs=data_crs)
                if victim_points
                else gpd.GeoDataFrame(columns=["geometry"], crs=data_crs)
            )

            self.environments[size] = {
                "item": item,
                "victims": victims_gdf,
                "crs": data_crs,
            }
        log.info("All datasets loaded and prepared.")

    def run_grasp_evaluation(self) -> tuple[pd.DataFrame, dict]:
        """
        Run GRASP algorithm across all loaded datasets.

        Returns:
            tuple: (metrics_df, time_series_data)
        """
        if not self.environments:
            log.error("No datasets loaded. Please call 'load_datasets()' first.")
            return pd.DataFrame(), {}

        all_results = []
        self.time_series_data = {}

        for size, env_data in self.environments.items():
            item = env_data["item"]
            victims_gdf = env_data["victims"]

            log.info(f"--- Evaluating GRASP on '{size}' dataset ---")

            evaluator = metrics.PathEvaluator(
                item.heatmap,
                item.bounds,
                victims_gdf,
                self.path_generator_config.fov_degrees,
                self.path_generator_config.altitude_meters,
                self.loader._meter_per_bin
            )

            center_proj = (
                gpd.GeoDataFrame(geometry=[Point(item.center_point)], crs="EPSG:4326")
                .to_crs(env_data["crs"])
                .geometry.iloc[0]
            )

            for name, generator in self.path_generators.items():
                log.info(f"Running {name} algorithm on '{size}' dataset...")
                
                generated_paths = generator(
                    center_proj.x,
                    center_proj.y,
                    item.radius_km * 1000,
                    item.heatmap,
                    item.bounds,
                )

                self.generated_paths[(size, name)] = generated_paths

                all_metrics = evaluator.calculate_all_metrics(generated_paths, 0.999)

                victim_metrics = all_metrics['victim_detection_metrics']

                result = {
                    "Dataset": size,
                    "Algorithm": name,
                    "Environment Type": item.environment_type,
                    "Climate": item.environment_climate,
                    "Environment Size": size,
                    "n_agents": self.path_generator_config.num_drones,
                    "Budget (m)": self.path_generator_config.budget,
                    "Likelihood Score": all_metrics['total_likelihood_score'],
                    "Time-Discounted Score": all_metrics['total_time_discounted_score'],
                    "Victims Found (%)": victim_metrics['percentage_found'],
                    "Area Covered (km²)": all_metrics['area_covered'],
                    "Total Path Length (km)": all_metrics['total_path_length'],
                }

                all_results.append(result)

                # Collect time-series data
                if name not in self.time_series_data:
                    self.time_series_data[name] = []

                cumulative_likelihoods = all_metrics['cumulative_likelihoods']
                if cumulative_likelihoods:
                    individual_drone_data = []

                    for drone_idx, cum_lik in enumerate(cumulative_likelihoods):
                        if len(cum_lik) > 0 and drone_idx < len(generated_paths):
                            drone_path = generated_paths[drone_idx]

                            drone_positions = []
                            if not drone_path.is_empty and drone_path.length > 0:
                                interpolation_resolution = int(
                                    np.ceil(self.loader._meter_per_bin / 2)
                                )
                                num_points = int(
                                    np.ceil(drone_path.length / interpolation_resolution)
                                ) + 1
                                distances = np.linspace(0, drone_path.length, num_points)

                                for d in distances:
                                    point = drone_path.interpolate(d)
                                    drone_positions.append((point.x, point.y))

                                while len(drone_positions) < len(cum_lik):
                                    if drone_positions:
                                        drone_positions.append(drone_positions[-1])
                                    else:
                                        drone_positions.append((0, 0))

                                drone_positions = drone_positions[:len(cum_lik)]
                            else:
                                drone_positions = [(0, 0)] * len(cum_lik)

                            individual_drone_data.append({
                                'drone_id': drone_idx,
                                'cumulative_likelihood': cum_lik,
                                'positions': drone_positions
                            })
                else:
                    individual_drone_data = []

                self.time_series_data[name].append({
                    'individual_drone_data': individual_drone_data,
                })

        self.results = pd.DataFrame(all_results)
        log.info("--- GRASP Evaluation Complete ---")
        log.info(f"Results:\n{self.results.to_string()}")
        return self.results, self.time_series_data

    def plot_results(self, results_df: pd.DataFrame = None, output_dir="graphs"):
        """Generate and save plots for evaluation results."""
        if results_df is None:
            results_df = self.results

        plot_single_evaluation_results(results_df, self.evaluation_sizes, output_dir)
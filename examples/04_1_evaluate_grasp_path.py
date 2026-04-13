# examples/04_evaluate_coverage_paths.py
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point 
import sarenv
from sarenv.analytics import metrics
from sarenv.utils import plot
from sarenv.analytics.GRASP_evaluator import GraspEvaluator
log = sarenv.get_logger()

if __name__ == "__main__":
    log.info("--- Initializing the Search and Rescue Toolkit ---")
    data_dir = "sarenv_dataset/1"  # Path to the dataset directory

    # 1. Initialize the evaluator
    evaluator = GraspEvaluator(
        dataset_directory=data_dir,
        evaluation_sizes=[ "medium"],
        num_drones=1,
        num_lost_persons=60,
        budget=60000,
        grasp_iterations=270,       
        enable_local_search=True,
        alpha_min=0.1, alpha_max=0.4, alpha_steps=8,
        use_reactive_alpha=True,
    )
        
    # 2. Run the evaluations
    grasp_results, time_series_data = evaluator.run_grasp_evaluation()

    # 3. Plot the results from the grasp run
    evaluator.plot_results(grasp_results)
 
    # 4. Plot paths on heatmaps for each algorithm and dataset
    log.info("--- Generating Path Heatmap Visualizations ---")

    # Create output directory for heatmap plots
    output_dir = Path()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through each environment and algorithm to generate plots
    for size, env_data in evaluator.environments.items():
        item = env_data["item"]
        victims_gdf = env_data["victims"]

        # Create path evaluator for this environment
        path_evaluator = metrics.PathEvaluator(
            item.heatmap,
            item.bounds,
            victims_gdf,
            evaluator.path_generator_config.fov_degrees,
            evaluator.path_generator_config.altitude_meters,
            evaluator.loader._meter_per_bin,
        )

        # Get center point in projected coordinates
        center_proj = (
            gpd.GeoDataFrame(geometry=[Point(item.center_point)], crs="EPSG:4326")
            .to_crs(env_data["crs"])
            .geometry.iloc[0]
        )

        # Plot paths for each algorithm
        for name, generator in evaluator.path_generators.items():
            log.info(f"Generating heatmap plot for {name} on '{size}' dataset...")

            # Generate paths using the same logic as in the evaluator
            generated_paths = evaluator.generated_paths[(size, name)]

            # Define plot bounds (use heatmap bounds)
            x_min, y_min, x_max, y_max = item.bounds

            # Create output filename
            output_file = output_dir / f"{name}_{size}_heatmap.pdf"

            # Plot the heatmap with paths
            plot.plot_heatmap(
                item=item,
                generated_paths=generated_paths,
                name=name,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                output_file=output_file
            )

            log.info(f"Saved heatmap plot: {output_file}")
    log.info("--- Path Heatmap Visualization Complete ---")
    
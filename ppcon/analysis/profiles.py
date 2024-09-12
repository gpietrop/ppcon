import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ppcon.config import RESULTS_DIR
from ppcon.analysis.utils_analysis import (get_reconstruction, moving_average, get_variance_list_season_ga,
                                           get_profile_list_season_ga, dict_color, dict_ga, dict_season)
from ppcon.dict import *


def reconstruct_profiles(variable, date_model=None, epoch_model=None, mode="test"):
    """
    Reconstructs and visualizes environmental example_profiles for a specified variable using a trained model.

    :param variable: str
        The variable to reconstruct (e.g., "NITRATE", "BBP700").
    :param date_model: str
        The date associated with the model used for reconstruction (e.g., "2023-08-19").
    :param epoch_model: int
        The epoch number of the model to be loaded for evaluation.
    :param mode: str
        The mode determining the dataset used for reconstruction (e.g., "all").

    :return: None
    """
    if not (date_model and epoch_model):
        date_model = dict_models[variable][0]
        epoch_model = dict_models[variable][1]

    # Define the directory paths for saving figures
    base_path = os.path.join(RESULTS_DIR, f"{variable}/")
    os.makedirs(base_path, exist_ok=True)

    path_analysis = os.path.join(base_path, f"profiles_{mode}/")
    os.makedirs(path_analysis, exist_ok=True)

    # Retrieve the reconstructed and measured example_profiles
    reconstruction = get_reconstruction(variable, date_model, epoch_model, mode)
    if len(reconstruction) != 5:
        raise ValueError("Expected 5 elements in reconstruction data (lat, lon, day_rad, generated, measured)")

    lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = reconstruction
    number_samples = len(generated_var_list)

    if number_samples == 0:
        print("No samples found for reconstruction.")
        return

    for index_sample in range(number_samples):
        lat = lat_list[index_sample]
        lon = lon_list[index_sample]
        generated_profile = generated_var_list[index_sample].detach().numpy()
        measured_profile = measured_var_list[index_sample].detach().numpy()

        max_pres = dict_max_pressure.get(variable, None)
        if max_pres is None:
            raise ValueError(f"Maximum pressure for variable '{variable}' not found.")

        depth = np.linspace(0, max_pres, len(generated_profile))

        # Apply moving average
        measured_profile = moving_average(measured_profile, 3)
        generated_profile = moving_average(generated_profile, 3)

        # Plotting the example_profiles
        plt.figure(figsize=(6, 7))
        plt.plot(measured_profile, depth, lw=3, color="#2CA02C", label="Measured")
        plt.plot(generated_profile, depth, lw=3, linestyle=(0, (3, 1, 1, 1)), color="#1F77B4", label="PPCon")
        plt.gca().invert_yaxis()

        plt.xlabel(f"{dict_var_name[variable]} [{dict_unit_measure[variable]}]")
        plt.ylabel(r"Depth [$m$]")

        if variable == "BBP700":
            ax = plt.gca()
            x_labels = ax.get_xticks()
            ax.set_xticklabels(['{:,.0e}'.format(x) for x in x_labels])

        plt.legend()
        plt.tight_layout()

        # Save the figure
        plt.savefig(os.path.join(path_analysis, f"profile_{round(lat, 2)}_{round(lon, 2)}.png"))
        plt.close()


def plot_profile_mean_by_season_and_region(variable, date_model=None, epoch_model=None, mode="test"):
    """
    Generates and visualizes environmental example_profiles across different seasons and geographical areas.

    :param variable: str
        The variable to visualize example_profiles for (e.g., "NITRATE", "CHLA", "BBP700").
    :param date_model: str, optional
        The date associated with the model used for reconstruction. If not provided, a default date is used based on the variable.
    :param epoch_model: int, optional
        The epoch number of the model to be loaded for evaluation. If not provided, a default epoch is used based on the variable.
    :param mode: str, optional
        The mode determining the dataset used for profile generation (e.g., "test"). Default is "test".

    :return: None
    """

    # Set default model date and epoch if not provided
    if not date_model or not epoch_model:
        date_model = dict_models[variable][0]
        epoch_model = dict_models[variable][1]

    # Define the directory paths for saving figures
    base_path = os.path.join(RESULTS_DIR, f"{variable}/")
    os.makedirs(base_path, exist_ok=True)

    # Retrieve the reconstructed and measured example_profiles
    lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = get_reconstruction(
        variable, date_model, epoch_model, mode
    )

    # Create subplots for the seasons
    fig, axs = plt.subplots(2, 2, figsize=(7, 7))

    for ga in dict_ga.keys():
        for index_season, (season, _) in enumerate(dict_season.items()):
            generated, measured = get_profile_list_season_ga(
                season, ga, lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list
            )
            max_pres = dict_max_pressure[variable]
            depth = np.linspace(0, max_pres, len(generated.detach().numpy()))

            row, col = divmod(index_season, 2)
            axs[row, col].plot(generated.detach().numpy(), depth, linestyle="dashed", color=dict_color[ga])
            axs[row, col].plot(measured.detach().numpy(), depth, label=ga, linestyle="solid", color=dict_color[ga])
            axs[row, col].invert_yaxis()

    # Set titles and labels for each subplot
    season_titles = ["Winter", "Spring", "Summer", "Autumn"]
    for index_season, ax in enumerate(axs.flat):
        ax.set_title(season_titles[index_season])
        ax.set_xlabel(f"{dict_var_name[variable]} ({dict_unit_measure[variable]})")
        ax.set_ylabel(r"Depth [$m$]")
        if variable == "BBP700":
            ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0e}'))

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, f"profiles_variance.png"))
    plt.show()
    plt.close()

    return


def plot_profile_variance_by_season_and_region(variable, date_model=None, epoch_model=None, mode="test"):
    """
    Generates and visualizes the variance of environmental example_profiles across different seasons and geographical areas.

    :param variable: str
        The variable to visualize variance for (e.g., "NITRATE", "CHLA", "BBP700").
    :param date_model: str, optional
        The date associated with the model used for reconstruction. If not provided, a default date is used based on the variable.
    :param epoch_model: int, optional
        The epoch number of the model to be loaded for evaluation. If not provided, a default epoch is used based on the variable.
    :param mode: str, optional
        The mode determining the dataset used for variance calculation (e.g., "test"). Default is "test".

    :return: None
    """

    # Set default model date and epoch if not provided
    if not date_model or not epoch_model:
        date_model = dict_models[variable][0]
        epoch_model = dict_models[variable][1]

    # Define the directory paths for saving figures
    base_path = os.path.join(RESULTS_DIR, f"{variable}/")
    os.makedirs(base_path, exist_ok=True)

    # Retrieve the reconstructed and measured example_profiles
    lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = get_reconstruction(
        variable, date_model, epoch_model, mode
    )

    # Create subplots for the seasons
    fig, axs = plt.subplots(2, 2, figsize=(7, 7))

    for ga in dict_ga.keys():
        for index_season, season in enumerate(dict_season.keys()):
            variance = get_variance_list_season_ga(
                season, ga, lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list
            )
            max_pres = dict_max_pressure[variable]
            depth = np.linspace(0, max_pres, len(variance.detach().numpy()))

            row, col = divmod(index_season, 2)
            axs[row, col].plot(variance.detach().numpy(), depth, color=dict_color[ga], label=ga)
            axs[row, col].invert_yaxis()

    # Set titles and labels for each subplot
    season_titles = ["Winter", "Spring", "Summer", "Autumn"]
    for index_season, ax in enumerate(axs.flat):
        ax.set_title(season_titles[index_season])
        ax.set_xlabel(f"{variable} ({dict_unit_measure[variable]})")
        ax.set_ylabel(r"Depth [$m$]")

    # Adjust layout and add legend
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="small")
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, f"profiles_variance.png"))
    plt.show()
    plt.close()

    return




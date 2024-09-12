import os

import numpy as np
from torch.nn.functional import mse_loss
import matplotlib.pyplot as plt
import seaborn as sns

from ppcon.config import RESULTS_DIR
from ppcon.analysis.utils_analysis import get_reconstruction, dict_ga, dict_season
from ppcon.utils.utils_train import from_day_rad_to_day
from ppcon.dict import *


def make_boxplots(variable, date_model=None, epoch_model=None, mode="test"):
    """
    Generates box plots of loss metrics for different seasons and geographical areas.

    :param variable: str
        The variable to generate box plots for (e.g., "NITRATE", "CHLA", "BBP700").
    :param date_model: str, optional
        The date associated with the model used for reconstruction. If not provided, a default date is used based on the variable.
    :param epoch_model: int, optional
        The epoch number of the model to be loaded for evaluation. If not provided, a default epoch is used based on the variable.
    :param mode: str, optional
        The mode determining the dataset used for box plot generation (e.g., "test"). Default is "test".

    :return: None
    """

    # Set default model date and epoch if not provided
    if not date_model or not epoch_model:
        date_model = dict_models[variable][0]
        epoch_model = dict_models[variable][1]

    # Define the directory paths for saving figures
    base_path = os.path.join(RESULTS_DIR, f"{variable}/")
    os.makedirs(base_path, exist_ok=True)

    # Initialize the list to store losses
    list_loss = [[[] for _ in range(len(dict_ga))] for _ in range(len(dict_season))]

    # Retrieve the reconstructed and measured example_profiles
    lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = get_reconstruction(
        variable, date_model, epoch_model, mode
    )

    number_samples = len(generated_var_list)

    # Calculate losses and categorize them by season and geographical area
    for i in range(number_samples):
        day_sample = from_day_rad_to_day(day_rad=day_rad_list[i])
        lat = lat_list[i]
        lon = lon_list[i]
        generated_profile = generated_var_list[i]
        measured_profile = measured_var_list[i]

        for index_s, (season, days_range) in enumerate(dict_season.items()):
            if days_range[0] <= day_sample <= days_range[1]:
                for index_ga, (ga, lat_lon_range) in enumerate(dict_ga.items()):
                    if lat_lon_range[0][0] <= lat <= lat_lon_range[0][1] and lat_lon_range[1][0] <= lon <= lat_lon_range[1][1]:
                        loss_sample = np.sqrt(mse_loss(generated_profile, measured_profile))
                        list_loss[index_s][index_ga].append(float(loss_sample))

    # Create box plots for each season
    fig, axs = plt.subplots(2, 2, figsize=(9, 6))

    for index_s, season in enumerate(dict_season.keys()):
        row = index_s // 2
        col = index_s % 2
        sns.boxplot(data=list_loss[index_s], palette="magma", ax=axs[row, col])
        axs[row, col].set_title(season)
        axs[row, col].set_xlabel(f"{variable} ({dict_unit_measure[variable]})")
        axs[row, col].set_ylabel('Fitness')
        axs[row, col].set_xticks(range(len(dict_ga)))
        axs[row, col].set_xticklabels(dict_ga.keys())
        if variable == "NITRATE":
            axs[row, col].set_ylim([0.0, 1.0])
        elif variable == "CHLA":
            axs[row, col].set_ylim([0.0, 0.15])
        elif variable == "BBP700":
            axs[row, col].set_ylim([0.0, 0.0004])

    # Hide x labels and tick labels for top plots and y ticks for right plots
    for ax in axs.flat:
        ax.label_outer()

    plt.tight_layout()
    plt.savefig(os.path.join(base_path, f"bp.png"))
    plt.show()
    plt.close()

    return

import os
import numpy as np
from torch.nn.functional import mse_loss
import matplotlib
import matplotlib.pyplot as plt

from ppcon.dict import *
from ppcon.utils.utils_train import from_day_rad_to_day
from ppcon.analysis.utils_analysis import get_reconstruction, dict_color, dict_ga, dict_season


def compute_rmse(variable, date_model=None, epoch_model=None, mode="test", make_fig=False):
    """
    Calculates the RMSE (Root Mean Square Error) across different seasons and geographical areas for a specified variable.

    :param variable: str
        The variable to calculate RMSE for (e.g., "NITRATE", "BBP700").
    :param date_model: str, optional
        The date associated with the model used for reconstruction. If not provided, a default date is used based on the variable.
    :param epoch_model: int, optional
        The epoch number of the model to be loaded for evaluation. If not provided, a default epoch is used based on the variable.
    :param mode: str, optional
        The mode determining the dataset used for RMSE calculation (e.g., "test"). Default is "test".
    :param make_fig: bool, optional
        If True, generate and save bar charts showing the distribution of samples across geographical areas for each season. Default is False.

    :return: None
    """

    # Set default model date and epoch if not provided
    if not date_model or not epoch_model:
        date_model = dict_models[variable][0]
        epoch_model = dict_models[variable][1]

    # Set font properties for plots
    font = {'family': 'normal', 'weight': 'bold', 'size': 22}
    matplotlib.rc('font', **font)

    # Define the directory path for saving results
    path_analysis = os.path.join(os.getcwd(), f"../results/{variable}/{date_model}/")
    os.makedirs(path_analysis, exist_ok=True)

    list_loss = np.zeros((len(dict_season), len(dict_ga)))
    list_number_samples = np.zeros((len(dict_season), len(dict_ga)))

    # Retrieve the reconstructed and measured example_profiles
    lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = get_reconstruction(
        variable, date_model, epoch_model, mode
    )

    number_samples = len(generated_var_list)

    # Calculate RMSE for each season and geographical area
    for index_sample in range(number_samples):
        day_sample = from_day_rad_to_day(day_rad=day_rad_list[index_sample])
        lat = lat_list[index_sample]
        lon = lon_list[index_sample]
        generated_profile = generated_var_list[index_sample]
        measured_profile = measured_var_list[index_sample]

        for index_season, (season, days_range) in enumerate(dict_season.items()):
            if days_range[0] <= day_sample <= days_range[1]:
                for index_ga, (ga, lat_lon_range) in enumerate(dict_ga.items()):
                    if lat_lon_range[0][0] <= lat <= lat_lon_range[0][1] and lat_lon_range[1][0] <= lon <= lat_lon_range[1][1]:
                        loss_sample = np.sqrt(mse_loss(generated_profile, measured_profile))
                        list_loss[index_season, index_ga] += loss_sample
                        list_number_samples[index_season, index_ga] += 1

    # Avoid division by zero
    list_loss = np.divide(list_loss, list_number_samples, out=np.zeros_like(list_loss), where=list_number_samples != 0)

    # Print RMSE results
    for index_ga, ga in enumerate(dict_ga.keys()):
        print(f"{ga} loss")
        loss_sum = np.sum(list_loss[:, index_ga] * list_number_samples[:, index_ga])
        num_samples_sum = np.sum(list_number_samples[:, index_ga])
        avg_loss = loss_sum / num_samples_sum if num_samples_sum > 0 else float('nan')
        print(f"Overall error: {round(avg_loss, 2)} \t #samples: {int(num_samples_sum)}")
        for index_season, season in enumerate(dict_season.keys()):
            season_loss = list_loss[index_season, index_ga]
            season_samples = list_number_samples[index_season, index_ga]
            print(f"{season} season: \t error: {round(season_loss, 2)} \t #samples: {int(season_samples)}")

    # Generate and save bar charts if requested
    if make_fig:
        for index_season, season in enumerate(dict_season.keys()):
            plt.bar(x=range(len(dict_ga)), height=list_number_samples[index_season], color=list(dict_color.values()))
            plt.xticks(range(len(dict_ga)), dict_ga.keys())
            plt.ylim([0, np.max(list_number_samples) + 10])
            plt.title(f"{season} season data distribution -- {variable}")
            plt.xlabel("Geographical Area")
            plt.ylabel("Number of Samples")
            plt.tight_layout()
            plt.savefig(os.path.join(path_analysis, f"hist_{variable}_{mode}_{season}_{epoch_model}.png"))
            plt.close()

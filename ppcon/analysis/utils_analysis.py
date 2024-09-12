import os

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

import seaborn as sns

from ppcon.utils.dataset import FloatDataset
from ppcon.utils.utils_train import upload_and_evaluate_model, from_day_rad_to_day

sns.set_theme(context='paper', style='whitegrid', font='sans-serif', font_scale=1.5,
              color_codes=True, rc=None)

pal = sns.color_palette("magma")

dict_color = {'NWM': pal[0], 'SWM': pal[1], 'TYR': pal[3], 'ION': pal[4], 'LEV': pal[5]}

dict_ga = {
    'NWM': [[40, 45], [-2, 9.5]],
    'SWM': [[32, 40], [-2, 9.5]],
    'TYR': [[37, 45], [9.5, 16]],
    'ION': [[30, 37], [9.5, 22]],
    'LEV': [[30, 37], [22, 36]]
}

dict_season = {
    'W': [0, 91],
    'SP': [92, 182],
    'SU': [183, 273],
    'A': [274, 365]
}


def moving_average(data, window_size):
    """
    Computes the moving average of a 1D array with a specified window size.

    :param data: array-like
        The input data for which the moving average is to be calculated.
    :param window_size: int
        The size of the moving window. If an even number is provided, it is incremented by 1 to ensure symmetry.

    :return: numpy.ndarray
        An array containing the moving average values, with the same length as the input data.
    """
    if window_size % 2 == 0:
        window_size += 1  # Ensure window size is odd for symmetry

    pad_size = window_size // 2
    padded_data = np.pad(data, pad_size, mode='edge')
    cumsum_vec = np.cumsum(np.insert(padded_data, 0, 0))

    moving_avg = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    return moving_avg


def count_samples(variable):
    """
    Counts the number of samples in the training, test, and removed datasets for a given variable.

    :param variable: str
        The variable for which to count samples (e.g., "NITRATE", "CHLA", "BBP700").

    :return: tuple
        A tuple containing three integers:
        - The number of samples in the training dataset.
        - The number of samples in the test dataset.
        - The number of samples in the removed dataset.
    """
    path_ds = os.path.join(os.getcwd(), f"../ds/{variable}/")

    ds_train = FloatDataset(os.path.join(path_ds, "float_ds_sf_train.csv"))
    ds_test = FloatDataset(os.path.join(path_ds, "float_ds_sf_test.csv"))
    ds_removed = FloatDataset(os.path.join(path_ds, "float_ds_sf_removed.csv"))

    return len(ds_train), len(ds_test), len(ds_removed)


def get_profile_list_season_ga(season, ga, lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list):
    """
    Averages the generated and measured example_profiles for a given season and geographical area.

    :param season: str
        The season to filter example_profiles by (e.g., 'W' for Winter, 'SP' for Spring).
    :param ga: str
        The geographical area to filter example_profiles by (e.g., 'NWM', 'SWM').
    :param lat_list: list of floats
        The list of latitudes corresponding to the example_profiles.
    :param lon_list: list of floats
        The list of longitudes corresponding to the example_profiles.
    :param day_rad_list: list of floats
        The list of day radians corresponding to the example_profiles.
    :param generated_var_list: list of tensors
        The list of generated variable example_profiles as tensors.
    :param measured_var_list: list of tensors
        The list of measured variable example_profiles as tensors.

    :return: tuple of tensors
        A tuple containing two tensors:
        - The averaged generated profile for the specified season and geographical area.
        - The averaged measured profile for the specified season and geographical area.
    """

    len_profile = generated_var_list[0].size(dim=0)
    generated_profile = torch.zeros(len_profile)
    measured_profile = torch.zeros(len_profile)
    number_seasonal_samples = 0

    for i in range(len(generated_var_list)):
        day_sample = from_day_rad_to_day(day_rad=day_rad_list[i])
        if dict_season[season][0] <= day_sample <= dict_season[season][1]:
            if dict_ga[ga][0][0] <= lat_list[i] <= dict_ga[ga][0][1] and dict_ga[ga][1][0] <= lon_list[i] <= \
                    dict_ga[ga][1][1]:
                generated_profile += generated_var_list[i]
                measured_profile += measured_var_list[i]
                number_seasonal_samples += 1

    if number_seasonal_samples > 0:
        generated_profile /= number_seasonal_samples
        measured_profile /= number_seasonal_samples
    else:
        generated_profile = torch.zeros(len_profile)
        measured_profile = torch.zeros(len_profile)

    return generated_profile, measured_profile


def get_variance_list_season_ga(season, ga, lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list):
    """
    Calculates the standard deviation (variance) of generated example_profiles for a specific season and geographical area.

    :param season: str
        The season to filter example_profiles by (e.g., 'W' for Winter, 'SP' for Spring).
    :param ga: str
        The geographical area to filter example_profiles by (e.g., 'NWM', 'SWM').
    :param lat_list: list of floats
        The list of latitudes corresponding to the example_profiles.
    :param lon_list: list of floats
        The list of longitudes corresponding to the example_profiles.
    :param day_rad_list: list of floats
        The list of day radians corresponding to the example_profiles.
    :param generated_var_list: list of tensors
        The list of generated variable example_profiles as tensors.
    :param measured_var_list: list of tensors
        The list of measured variable example_profiles as tensors (not used in this function).

    :return: torch.Tensor
        A tensor containing the standard deviation of the generated example_profiles at each depth level.
    """

    len_profile = generated_var_list[0].size(dim=0)
    generated_profiles = []

    for i in range(len(generated_var_list)):
        day_sample = from_day_rad_to_day(day_rad=day_rad_list[i])
        if dict_season[season][0] <= day_sample <= dict_season[season][1]:
            if dict_ga[ga][0][0] <= lat_list[i] <= dict_ga[ga][0][1] and dict_ga[ga][1][0] <= lon_list[i] <= \
                    dict_ga[ga][1][1]:
                generated_profiles.append(generated_var_list[i])

    if len(generated_profiles) == 0:
        raise ValueError(f"No example_profiles found for season '{season}' and geographical area '{ga}'.")

    std_reconstruction = torch.zeros(len_profile)
    for k in range(len_profile):
        depth_values = [prof[k].item() for prof in generated_profiles]
        std_reconstruction[k] = np.std(depth_values)

    return std_reconstruction


def get_reconstruction(variable, date_model, epoch_model, mode):
    """
    Reconstructs environmental data using a pre-trained model and returns the generated and measured example_profiles.

    :param variable: str
        The variable to be reconstructed (e.g., "NITRATE", "BBP700").
    :param date_model: str
        The date associated with the model used for reconstruction (e.g., "2023-08-19").
    :param epoch_model: int
        The epoch number of the model to be loaded for evaluation.
    :param mode: str
        The mode determining the dataset used for reconstruction (e.g., "test", "train", "all").

    :return: tuple of lists
        A tuple containing five lists:
        - lat_list (list of floats): List of latitudes corresponding to the reconstructed data points.
        - lon_list (list of floats): List of longitudes corresponding to the reconstructed data points.
        - day_rad_list (list of floats): List of day of year in radians corresponding to the reconstructed data points.
        - generated_var_list (list of tensors): List of tensors representing the reconstructed variable data.
        - measured_var_list (list of tensors): List of tensors representing the measured variable data.
    """
    # Load the input dataset
    path_float = os.path.join(os.getcwd(), f"ds/{variable}/float_ds_sf_{mode}.csv")
    if mode == "all":
        path_float = os.path.join(os.getcwd(), f"ds/{variable}/float_ds_sf.csv")
    dataset = FloatDataset(path_float)
    ds = DataLoader(dataset, shuffle=True)

    # Load model information and directory
    dir_model = os.path.join(os.getcwd(), f"results/{variable}/{date_model}/model")
    info = pd.read_csv(os.path.join(os.getcwd(), f"results/{variable}/{date_model}/info.csv"))

    # Upload and evaluate the model
    model_day, model_year, model_lat, model_lon, model = upload_and_evaluate_model(
        dir_model=dir_model, info_model=info, ep=epoch_model
    )

    # Initialize lists to store the outputs
    lat_list = []
    lon_list = []
    day_rad_list = []
    measured_var_list = []
    generated_var_list = []

    # Iterate over the dataset and generate the reconstruction
    for sample in ds:
        year, day_rad, lat, lon, temp, psal, doxy, measured_var = sample

        # Model outputs
        output_day = model_day(day_rad.unsqueeze(1))
        output_year = model_year(year.unsqueeze(1))
        output_lat = model_lat(lat.unsqueeze(1))
        output_lon = model_lon(lon.unsqueeze(1))

        # Prepare inputs for the main model
        inputs = torch.cat((
            torch.transpose(output_day.unsqueeze(0), 0, 1),
            torch.transpose(output_year.unsqueeze(0), 0, 1),
            torch.transpose(output_lat.unsqueeze(0), 0, 1),
            torch.transpose(output_lon.unsqueeze(0), 0, 1),
            torch.transpose(temp.unsqueeze(0), 0, 1),
            torch.transpose(psal.unsqueeze(0), 0, 1),
            torch.transpose(doxy.unsqueeze(0), 0, 1)
        ), 1)

        generated_var = model(inputs.float()).detach()

        # Post-processing based on the variable type
        if variable == "NITRATE":
            generated_var = torch.squeeze(generated_var)[:-10]
            measured_var = torch.squeeze(measured_var)[:-10]
        elif variable == "BBP700":
            generated_var = torch.squeeze(generated_var) / 1000
            measured_var = torch.squeeze(measured_var) / 1000
        else:
            generated_var = torch.squeeze(generated_var)
            measured_var = torch.squeeze(measured_var)

        # Store results in the respective lists
        lat_list.append(lat.item())
        lon_list.append(lon.item())
        day_rad_list.append(day_rad.item())
        generated_var_list.append(generated_var)
        measured_var_list.append(measured_var)

    return lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list

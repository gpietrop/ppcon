import os

import numpy as np
import pandas as pd
import torch
import netCDF4 as nc

from torch.utils.data import DataLoader, TensorDataset

from ppcon.utils.dataset import FloatDataset
from ppcon.utils.utils_train import upload_and_evaluate_model
from ppcon.dict import *


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def discretize_user(pres, var, max_pres, interval):
    """
    Discretizes a profile based on the specified pressure intervals and creates a tensor.

    :param pres: array-like
        An array of pressure values corresponding to the profile.
    :param var: array-like
        An array of variable values corresponding to the pressure values.
    :param max_pres: float
        The maximum pressure value to consider for discretization.
    :param interval: float
        The interval scale for discretization.

    :return: torch.Tensor or None
        A tensor containing the discretized values of the variable at the specified pressure intervals.
        Returns None if an appropriate index cannot be found during discretization.
    """
    size = int(max_pres / interval)
    discretization_pres = np.arange(0, max_pres, interval)

    out = torch.zeros(size)

    for i in range(size):
        pressure_discretize = discretization_pres[i]
        idx = find_nearest(pres, pressure_discretize)
        if idx is None:
            return None
        out[i] = torch.from_numpy(np.asarray(var[idx]))

    return out


def find_file_info(file_name):
    """
    Search for a specific file in the Float_Index.txt file and return its details.

    :param file_name: str
        The name of the file to search for (e.g., "MR6902733_003.nc").
    :return: dict or None
        A dictionary with the file details, including:
        - file_path (str): The path to the file as listed in Float_Index.txt.
        - latitude (str): The latitude where the data was recorded.
        - longitude (str): The longitude where the data was recorded.
        - timestamp (str): The timestamp associated with the data.
        - variables (str): A string listing the variables available in the file.

        Returns None if the file is not found in Float_Index.txt.
    """
    file_path = os.getcwd() + "/../example_profiles/Float_Index.txt"
    with open(file_path, 'r') as file:
        content = file.readlines()

    for line in content:
        if file_name in line:
            # Split the line into its components
            parts = line.strip().split(',')
            file_info = {
                "file_path": parts[0],
                "latitude": parts[1],
                "longitude": parts[2],
                "timestamp": parts[3],
                "variables": parts[4].strip()
            }
            return file_info

    return None


def create_input_from_file(file_name, variable="NITRATE"):
    """
    Creates an input data tuple from a NetCDF file based on the specified variable.

    :param file_name: str
        The name of the file (without the path) from which to extract the data.
    :param variable: str, optional
        The name of the variable to extract from the file. Defaults to "NITRATE".
    :return: tuple
        A tuple containing the following elements:
        - year (int): The year extracted from the timestamp in the file.
        - month (int): The month extracted from the timestamp in the file.
        - day (int): The day extracted from the timestamp in the file.
        - lat (float): The latitude extracted from the file.
        - lon (float): The longitude extracted from the file.
        - temp (tuple): A tuple containing temperature data and corresponding pressure levels.
        - psal (tuple): A tuple containing salinity data and corresponding pressure levels.
        - doxy (tuple): A tuple containing dissolved oxygen data and corresponding pressure levels.
        - variable (str): The name of the variable requested.
        - var (tuple or None): A tuple containing the data of the specified variable and corresponding pressure levels,
          or None if the variable is not present in the file.
    """

    file_info = find_file_info(file_name)
    flag_variable = variable in file_info["variables"]

    # Extract the date from the timestamp in file_info
    timestamp = file_info['timestamp']
    year = int(timestamp[:4])
    month = int(timestamp[4:6])
    day = int(timestamp[6:8])

    # Extract latitude and longitude from file_info
    lat = float(file_info['latitude'])
    lon = float(file_info['longitude'])

    # Load the NetCDF dataset
    path_ds = os.getcwd() + f"/../example_profiles/{file_name[2:9]}/{file_name}.nc"
    ds = nc.Dataset(path_ds)

    # Extract variables
    temp = (ds["TEMP"][:].data[:], ds["PRES_TEMP"][:].data[:])
    psal = (ds["PSAL"][:].data[:], ds["PRES_PSAL"][:].data[:])
    if "DOXY" not in ds.variables:
        raise ValueError("No oxygen variable information.")
    doxy = (ds["DOXY"][:].data[:], ds["PRES_DOXY"][:].data[:])

    if flag_variable:
        var = (ds[variable][:].data[:], ds[f"PRES_{variable}"][:].data[:])
    else:
        var = None

    # Prepare the input tuple
    input_data = (year, month, day, lat, lon, temp, psal, doxy, variable, var)
    return input_data


def preprocessing_data_user(year, month, day, lat, lon, tuple_temp, tuple_psal, tuple_doxy, variable, tuple_var=None):
    """
    Preprocesses environmental data for a specific date and location.

    :param year: int
        The year of the data (e.g., 2024).
    :param month: int
        The month of the data (1-12).
    :param day: int
        The day of the month (1-31).
    :param lat: float
        Latitude (-90 to 90 degrees).
    :param lon: float
        Longitude (-180 to 180 degrees).
    :param tuple_temp: tuple (list or tensor, list or tensor)
        Tuple of temperature values and corresponding pressure values.
    :param tuple_psal: tuple (list or tensor, list or tensor)
        Tuple of salinity values and corresponding pressure values.
    :param tuple_doxy: tuple (list or tensor, list or tensor)
        Tuple of dissolved oxygen values and corresponding pressure values.
    :param variable: str
        Name of the variable being processed (e.g., "temperature").
    :param tuple_var: tuple (list or tensor, list or tensor), optional
        Optional tuple of additional variable values and corresponding pressure values.
    :return: tuple
        A tuple containing preprocessed tensors for the following:
        - year (tensor): The year in tensor form.
        - day in radians (tensor): The day of the year converted to radians.
        - lat (tensor): Latitude as a tensor.
        - lon (tensor): Longitude as a tensor.
        - temp (tensor): Preprocessed temperature data.
        - psal (tensor): Preprocessed salinity data.
        - doxy (tensor): Preprocessed dissolved oxygen data.
        - optional variable (tensor or None): Preprocessed data for the optional variable, or None if not provided.
    :raises ValueError:
        If the data and pressure vectors have different lengths.
    """

    # Validate inputs
    assert -90 <= lat <= 90, "Latitude must be between -90 and 90 degrees."
    assert -180 <= lon <= 180, "Longitude must be between -180 and 180 degrees."

    pp_year = torch.tensor([float(year)])
    day_total = month * 30 + day
    pp_day_rad = torch.tensor([day_total * 2 * np.pi / 365])

    pp_lat = torch.tensor([lat], dtype=torch.float32)
    pp_lon = torch.tensor([lon], dtype=torch.float32)

    max_pres = dict_max_pressure[variable]
    interval = dict_interval[variable]

    def convert_to_tensor_and_check_length(data_tuple):
        data, pressure = data_tuple

        # Convert lists to tensors if necessary
        if isinstance(data, list):
            data = torch.tensor(data, dtype=torch.float32)
        if isinstance(pressure, list):
            pressure = torch.tensor(pressure, dtype=torch.float32)

        # Check if data and pressure have the same length
        if data.shape[0] != pressure.shape[0]:
            raise ValueError("The data and pressure vectors must have the same length.")

        return data, pressure

    temp, pres_temp = convert_to_tensor_and_check_length(tuple_temp)
    psal, pres_psal = convert_to_tensor_and_check_length(tuple_psal)
    doxy, pres_doxy = convert_to_tensor_and_check_length(tuple_doxy)

    if pres_temp[-1] < dict_max_pressure[variable]:
        if pres_temp[-1] < 0.8 * dict_max_pressure[variable]:
            raise ValueError(f"The pressure of temperature only reach {pres_temp[-1]} m.")
        temp = torch.cat((torch.from_numpy(temp), torch.tensor([temp[-1]])))
        pres_temp = torch.cat((torch.from_numpy(pres_temp), torch.tensor([dict_max_pressure[variable]])))
        # raise ValueError(f"The pressure of temperature does not reach {dict_max_pressure[variable]} m.")
    if pres_psal[-1] < dict_max_pressure[variable]:
        if pres_psal[-1] < 0.8 * dict_max_pressure[variable]:
            raise ValueError(f"The pressure of salinity only reach {pres_psal[-1]} m.")
        psal = torch.cat((torch.from_numpy(psal), torch.tensor([psal[-1]])))
        pres_psal = torch.cat((torch.from_numpy(pres_psal), torch.tensor([dict_max_pressure[variable]])))
        # raise ValueError(f"The pressure of salinity does not reach {dict_max_pressure[variable]} m.")
    if pres_doxy[-1] < dict_max_pressure[variable]:
        if pres_doxy[-1] < 0.8 * dict_max_pressure[variable]:
            raise ValueError(f"The pressure of oxygen only reach {pres_doxy[-1]} m.")
        doxy = torch.cat((torch.from_numpy(doxy), torch.tensor([doxy[-1]])))
        pres_doxy = torch.cat((torch.from_numpy(pres_doxy), torch.tensor([dict_max_pressure[variable]])))

        # raise ValueError(f"The pressure of oxygen does not reach {dict_max_pressure[variable]} m.")

    pp_temp = discretize_user(pres_temp, temp, max_pres, interval).unsqueeze(0)
    pp_psal = discretize_user(pres_psal, psal, max_pres, interval).unsqueeze(0)
    pp_doxy = discretize_user(pres_doxy, doxy, max_pres, interval).unsqueeze(0)

    if tuple_var is not None:
        var, pres_var = convert_to_tensor_and_check_length(tuple_var)
        if pres_var[-1] < dict_max_pressure[variable]:
            if pres_var[-1] < 0.8 * dict_max_pressure[variable]:
                raise ValueError(f"The pressure of {variable} only reach {pres_psal[-1]} m.")
            var = torch.cat((torch.from_numpy(var), torch.tensor([var[-1]])))
            pres_var = torch.cat((torch.from_numpy(pres_var), torch.tensor([dict_max_pressure[variable]])))
        pp_var = discretize_user(pres_var, var, max_pres, interval).unsqueeze(0)
    else:
        preprocessed_data = (
            pp_year,
            pp_day_rad,
            pp_lat,
            pp_lon,
            pp_temp,
            pp_psal,
            pp_doxy,
        )

        return preprocessed_data

    preprocessed_data = (
        pp_year,
        pp_day_rad,
        pp_lat,
        pp_lon,
        pp_temp,
        pp_psal,
        pp_doxy,
        pp_var
    )

    return preprocessed_data


def moving_average(data, window_size):
    """
    Calculates the moving average of a 1D array with a specified window size.

    :param data: array-like
        The input data for which the moving average is to be calculated.
    :param window_size: int
        The size of the moving window. If an even number is provided, it is incremented by 1 to ensure symmetry.

    :return: numpy.ndarray
        The array of moving averages with the same length as the input data.
    """
    if window_size % 2 == 0:
        window_size += 1  # Ensure window size is odd for symmetry
    pad_size = window_size // 2
    padded_data = np.pad(data, pad_size, mode='edge')
    cumsum_vec = np.cumsum(np.insert(padded_data, 0, 0))
    return (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size


def get_reconstruction_user(variable, date_model, epoch_model, mode, data=None):
    """
    Reconstructs environmental data using a trained model and optionally compares it to measured data.

    :param variable: str
        The variable being reconstructed (e.g., "NITRATE", "BBP700").
    :param date_model: str
        The date associated with the model used for reconstruction (e.g., "2023-08-19").
    :param epoch_model: int
        The epoch number of the model to be loaded for evaluation.
    :param mode: str
        The mode determining the dataset source. If "all", uses a different dataset path.
    :param data: tuple, optional
        A tuple of tensors representing the dataset to use for reconstruction. If not provided, the dataset is loaded from a CSV file.

    :return: tuple
        If measured data is available:
        - lat_list (list of floats): List of latitudes corresponding to the reconstructed data points.
        - lon_list (list of floats): List of longitudes corresponding to the reconstructed data points.
        - day_rad_list (list of floats): List of day of year in radians corresponding to the reconstructed data points.
        - generated_var_list (list of tensors): List of tensors representing the reconstructed variable data.
        - measured_var_list (list of tensors): List of tensors representing the measured variable data (if available).

        If measured data is not available:
        - lat_list (list of floats): List of latitudes corresponding to the reconstructed data points.
        - lon_list (list of floats): List of longitudes corresponding to the reconstructed data points.
        - day_rad_list (list of floats): List of day of year in radians corresponding to the reconstructed data points.
        - generated_var_list (list of tensors): List of tensors representing the reconstructed variable data."""
    # Determine the dataset source

    if data is not None:
        # If data is provided as tensors, create a TensorDataset directly
        dataset = TensorDataset(*data)
    else:
        path_float = os.getcwd() + f"/ds/{variable}/float_ds_sf_{mode}.csv"
        if mode == "all":
            path_float = os.getcwd() + f"/ds/{variable}/float_ds_sf.csv"
        dataset = FloatDataset(path_df=path_float)  # Assuming FloatDataset handles CSV loading and tensor conversion

    ds = DataLoader(dataset, shuffle=True)

    dir_model = os.getcwd() + f"/results/{variable}/{date_model}/model"
    info = pd.read_csv(os.getcwd() + f"/results/{variable}/{date_model}/info.csv")

    # Upload and evaluate the model
    model_day, model_year, model_lat, model_lon, model = upload_and_evaluate_model(
        dir_model=dir_model, info_model=info, ep=epoch_model
    )

    lat_list = []
    lon_list = []
    day_rad_list = []
    measured_var_list = []
    generated_var_list = []

    for sample in ds:
        if len(sample) == 8:
            year, day_rad, lat, lon, temp, psal, doxy, measured_var = sample
        else:
            year, day_rad, lat, lon, temp, psal, doxy = sample
            measured_var = None

        output_day = model_day(day_rad.unsqueeze(1))
        output_year = model_year(year.unsqueeze(1))
        output_lat = model_lat(lat.unsqueeze(1))
        output_lon = model_lon(lon.unsqueeze(1))

        output_day = torch.transpose(output_day.unsqueeze(0), 0, 1)
        output_year = torch.transpose(output_year.unsqueeze(0), 0, 1)
        output_lat = torch.transpose(output_lat.unsqueeze(0), 0, 1)
        output_lon = torch.transpose(output_lon.unsqueeze(0), 0, 1)
        temp = torch.transpose(temp.unsqueeze(0), 0, 1)
        psal = torch.transpose(psal.unsqueeze(0), 0, 1)
        doxy = torch.transpose(doxy.unsqueeze(0), 0, 1)

        x = torch.cat((output_day, output_year, output_lat, output_lon, temp, psal, doxy), 1)
        generated_var = model(x.float())
        generated_var = generated_var.detach()

        lat_list.append(lat.item())
        lon_list.append(lon.item())
        day_rad_list.append(day_rad.item())

        if variable == "NITRATE":
            generated_var = torch.squeeze(generated_var)[:-10]
            if measured_var is not None:
                measured_var = torch.squeeze(measured_var)[:-10]
        if variable == "BBP700":
            generated_var = torch.squeeze(generated_var) / 1000
            if measured_var is not None:
                measured_var = torch.squeeze(measured_var) / 1000
        else:
            generated_var = torch.squeeze(generated_var)
            if measured_var is not None:
                measured_var = torch.squeeze(measured_var)

        generated_var_list.append(generated_var)
        if measured_var is not None:
            measured_var_list.append(measured_var)

    if not measured_var_list:
        return lat_list, lon_list, day_rad_list, generated_var_list
    return lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list

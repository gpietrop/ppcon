import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ppcon.dict import dict_max_pressure, dict_var_name, dict_unit_measure
from ppcon.utils.utils_user import (preprocessing_data_user, create_input_from_file, get_reconstruction_user,
                                     moving_average)


sns.set_theme(context='paper', style='whitegrid', font='sans-serif', font_scale=1.5,
              color_codes=True, rc=None)


def generate_profiles_from_file(file_name, variable, date_model=None, epoch_model=None):
    """
    Generates and visualizes example_profiles from a NetCDF file for a specified variable using a pre-trained model.

    :param file_name: str
        The name of the file (without the path) from which to extract the data.
    :param variable: str
        The variable to generate example_profiles for (e.g., "NITRATE", "CHLA", "BBP700").
    :param date_model: str, optional
        The date associated with the model used for reconstruction. If not provided, a default date is used based on the variable.
    :param epoch_model: int, optional
        The epoch number of the model to be loaded for evaluation. If not provided, a default epoch is used based on the variable.

    :return: torch.Tensor
        The last generated profile as a tensor.

    Example:
    --------
    >>> prof = generate_profiles_from_file("MR6903266_262", "CHLA")
    >>> print(prof)
    tensor([...])
    """

    dict_models = {
        "NITRATE": ["2023-12-16", 100],
        "CHLA": ["2023-12-17", 150],
        "BBP700": ["2023-12-15", 125]
    }
    if not date_model:
        date_model = dict_models[variable][0]
    if not epoch_model:
        epoch_model = dict_models[variable][1]

    global generated_profile

    input_data = create_input_from_file(file_name, variable=variable)
    data = preprocessing_data_user(*input_data)

    # Get lat, lon, day_rad, generated, and measured variables lists
    reconstruction = get_reconstruction_user(
        variable=variable,
        date_model=date_model,
        epoch_model=epoch_model,
        mode='all',
        data=data
    )
    if len(reconstruction) == 5:
        lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = reconstruction
    else:
        lat_list, lon_list, day_rad_list, generated_var_list = reconstruction
        measured_var_list = None

    number_samples = len(generated_var_list)

    for index_sample in range(number_samples):
        generated_profile = generated_var_list[index_sample]
        if measured_var_list:
            measured_profile = measured_var_list[index_sample]
        else:
            measured_profile = None

        max_pres = dict_max_pressure[variable]
        depth = np.linspace(0, max_pres, len(generated_profile.detach().numpy()))
        plt.figure(figsize=(6, 7))

        if measured_profile is not None:
            measured_profile = moving_average(measured_profile.detach().numpy(), 3)
        generated_profile = moving_average(generated_profile.detach().numpy(), 3)

        if measured_profile is not None:
            plt.plot(measured_profile, depth, lw=3, color="#2CA02C", label=f"Measured")
        plt.plot(generated_profile, depth, lw=3, linestyle=(0, (3, 1, 1, 1)), color="#1F77B4", label=f"PPCon")
        plt.gca().invert_yaxis()

        plt.xlabel(f"{dict_var_name[variable]} [{dict_unit_measure[variable]}]")
        plt.ylabel(r"Depth [$m$]")

        if variable == "BBP700":
            ax = plt.gca()
            x_labels = ax.get_xticks()
            ax.set_xticklabels(['{:,.0e}'.format(x) for x in x_labels])

        plt.legend()
        plt.tight_layout()

        plt.legend()
        plt.show()
        plt.close()

    return generated_profile


def generate_profiles_from_input(variable, year, month, day, lat, lon, tuple_temp, tuple_psal, tuple_doxy,
                                 tuple_var=None, date_model=None, epoch_model=None):
    """
    Generates and visualizes example_profiles based on provided input data for a specified variable using a pre-trained model.

    :param variable: str
        The variable to generate example_profiles for (e.g., "NITRATE", "CHLA", "BBP700").
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
    :param tuple_var: tuple (list or tensor, list or tensor), optional
        Optional tuple of additional variable values and corresponding pressure values.
    :param date_model: str, optional
        The date associated with the model used for reconstruction. If not provided, a default date is used based on the variable.
    :param epoch_model: int, optional
        The epoch number of the model to be loaded for evaluation. If not provided, a default epoch is used based on the variable.

    :return: torch.Tensor
        The last generated profile as a tensor.
    """
    dict_models = {
        "NITRATE": ["2023-12-16", 100],
        "CHLA": ["2023-12-17", 150],
        "BBP700": ["2023-12-15", 125]
    }
    if not date_model:
        date_model = dict_models[variable][0]
    if not epoch_model:
        epoch_model = dict_models[variable][1]

    global generated_profile

    input_data = (year, month, day, lat, lon, tuple_temp, tuple_psal, tuple_doxy, variable, tuple_var)
    data = preprocessing_data_user(*input_data)

    # Get lat, lon, day_rad, generated, and measured variables lists
    reconstruction = get_reconstruction_user(
        variable=variable,
        date_model=date_model,
        epoch_model=epoch_model,
        mode='all',
        data=data
    )
    if len(reconstruction) == 5:
        lat_list, lon_list, day_rad_list, generated_var_list, measured_var_list = reconstruction
    else:
        lat_list, lon_list, day_rad_list, generated_var_list = reconstruction
        measured_var_list = None

    number_samples = len(generated_var_list)

    for index_sample in range(number_samples):
        generated_profile = generated_var_list[index_sample]
        if measured_var_list:
            measured_profile = measured_var_list[index_sample]
        else:
            measured_profile = None

        max_pres = dict_max_pressure[variable]
        depth = np.linspace(0, max_pres, len(generated_profile.detach().numpy()))
        plt.figure(figsize=(6, 7))

        if measured_profile:
            measured_profile = moving_average(measured_profile.detach().numpy(), 3)
        generated_profile = moving_average(generated_profile.detach().numpy(), 3)

        if measured_profile:
            plt.plot(measured_profile, depth, lw=3, color="#2CA02C", label=f"Measured")
        plt.plot(generated_profile, depth, lw=3, linestyle=(0, (3, 1, 1, 1)), color="#1F77B4", label=f"PPCon")
        plt.gca().invert_yaxis()

        plt.xlabel(f"{dict_var_name[variable]} [{dict_unit_measure[variable]}]")
        plt.ylabel(r"Depth [$m$]")

        if variable == "BBP700":
            ax = plt.gca()
            x_labels = ax.get_xticks()
            ax.set_xticklabels(['{:,.0e}'.format(x) for x in x_labels])

        plt.legend()
        plt.tight_layout()

        plt.legend()
        plt.show()
        plt.close()

    return generated_profile

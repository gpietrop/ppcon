import unittest
from unittest.mock import patch, MagicMock
import torch
import os
from ppcon.analysis.profiles import (
    reconstruct_profiles,
    plot_profile_mean_by_season_and_region,
    plot_profile_variance_by_season_and_region
)


class TestProfileFunctions(unittest.TestCase):

    @patch('ppcon.analysis.profiles.get_reconstruction')
    @patch('ppcon.analysis.profiles.moving_average')
    @patch('ppcon.analysis.profiles.os.makedirs')
    @patch('ppcon.analysis.profiles.plt.show')  # Mock plt.show to prevent actual plot windows during testing
    @patch('ppcon.analysis.profiles.plt.savefig')  # Mock plt.savefig to prevent actual file creation during testing
    def test_reconstruct_profiles(self, mock_savefig, mock_show, mock_makedirs, mock_moving_average,
                                  mock_get_reconstruction):
        # Mock the functions to return controlled outputs
        mock_get_reconstruction.return_value = (
            [45.0, 46.0],  # lat_list
            [12.0, 13.0],  # lon_list
            [0.5, 0.6],  # day_rad_list
            [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.5, 2.5, 3.5])],  # generated_var_list
            [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.5, 2.5, 3.5])]  # measured_var_list
        )
        mock_moving_average.side_effect = lambda x, _: torch.tensor(x)  # Mock moving_average to return tensor

        # Run the function
        reconstruct_profiles("NITRATE")

        # Ensure the directories were created
        mock_makedirs.assert_called()

        # Ensure the plotting function was called
        mock_savefig.assert_called()  # Ensure savefig was called to save the figure

        # Ensure that get_reconstruction was called with expected parameters
        mock_get_reconstruction.assert_called_once_with("NITRATE", "2023-12-16", 100, "test")

    @patch('ppcon.analysis.profiles.get_reconstruction')
    @patch('ppcon.analysis.profiles.get_profile_list_season_ga')
    @patch('ppcon.analysis.profiles.os.makedirs')
    @patch('ppcon.analysis.profiles.plt.show')  # Mock plt.show to prevent actual plot windows during testing
    @patch('ppcon.analysis.profiles.plt.close')  # Mock plt.close to prevent closing plots during testing
    def test_plot_profile_mean_by_season_and_region(self, mock_close, mock_show, mock_makedirs,
                                                    mock_get_profile_list_season_ga, mock_get_reconstruction):
        # Mock the get_reconstruction function to return controlled outputs
        mock_get_reconstruction.return_value = (
            [45.0],  # lat_list
            [12.0],  # lon_list
            [0.5],  # day_rad_list
            [torch.tensor([1.0, 2.0, 3.0])],  # generated_var_list
            [torch.tensor([1.0, 2.0, 3.0])]  # measured_var_list
        )

        mock_get_profile_list_season_ga.return_value = (
            torch.tensor([1.0, 2.0, 3.0]),  # generated profile
            torch.tensor([1.5, 2.5, 3.5])  # measured profile
        )

        # Run the function
        plot_profile_mean_by_season_and_region("CHLA")

        # Ensure the directories were created
        mock_makedirs.assert_called()

        # Ensure the plotting functions were called
        mock_show.assert_called_once()
        self.assertGreaterEqual(mock_close.call_count, 1)  # Check that plt.close() was called at least once

    @patch('ppcon.analysis.profiles.get_reconstruction')
    @patch('ppcon.analysis.profiles.get_variance_list_season_ga')
    @patch('ppcon.analysis.profiles.os.makedirs')
    @patch('ppcon.analysis.profiles.plt.show')  # Mock plt.show to prevent actual plot windows during testing
    @patch('ppcon.analysis.profiles.plt.close')  # Mock plt.close to prevent closing plots during testing
    def test_plot_profile_variance_by_season_and_region(self, mock_close, mock_show, mock_makedirs,
                                                        mock_get_variance_list_season_ga, mock_get_reconstruction):
        # Mock the get_reconstruction function to return controlled outputs
        mock_get_reconstruction.return_value = (
            [45.0],  # lat_list
            [12.0],  # lon_list
            [0.5],  # day_rad_list
            [torch.tensor([1.0, 2.0, 3.0])],  # generated_var_list
            [torch.tensor([1.0, 2.0, 3.0])]  # measured_var_list
        )

        mock_get_variance_list_season_ga.return_value = torch.tensor([0.1, 0.2, 0.3])  # Mock variance values

        # Run the function
        plot_profile_variance_by_season_and_region("BBP700")

        # Ensure the directories were created
        mock_makedirs.assert_called()

        # Ensure the plotting functions were called
        mock_show.assert_called_once()
        self.assertGreaterEqual(mock_close.call_count, 1)  # Check that plt.close() was called at least once


if __name__ == '__main__':
    unittest.main()

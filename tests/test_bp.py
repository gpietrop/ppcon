import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from ppcon.analysis.bp import make_boxplots


class TestMakeBoxplots(unittest.TestCase):

    @patch('ppcon.analysis.bp.get_reconstruction')
    @patch('ppcon.analysis.bp.os.makedirs')
    @patch('ppcon.analysis.bp.plt.show')  # Mock plt.show to prevent actual plot windows during testing
    def test_make_boxplots(self, mock_show, mock_makedirs, mock_get_reconstruction):
        # Mock the get_reconstruction function to return controlled outputs
        mock_get_reconstruction.return_value = (
            [45.0, 46.0],  # lat_list
            [12.0, 13.0],  # lon_list
            [0.5, 0.6],  # day_rad_list
            [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.5, 2.5, 3.5])],  # generated_var_list
            [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.5, 2.5, 3.5])]  # measured_var_list
        )

        # Run the function with the mock setup
        make_boxplots("NITRATE")

        # Check that directories were created
        mock_makedirs.assert_called_once()

        # Check that the plotting function was called
        mock_show.assert_called_once()

        # Check that get_reconstruction was called with expected parameters
        mock_get_reconstruction.assert_called_once_with("NITRATE", "2023-12-16", 100, "test")

    @patch('ppcon.analysis.bp.get_reconstruction')
    @patch('ppcon.analysis.bp.os.makedirs')
    @patch('ppcon.analysis.bp.plt.show')  # Mock plt.show to prevent actual plot windows during testing
    def test_make_boxplots_custom_args(self, mock_show, mock_makedirs, mock_get_reconstruction):
        # Mock the get_reconstruction function to return controlled outputs
        mock_get_reconstruction.return_value = (
            [45.0],  # lat_list
            [12.0],  # lon_list
            [0.5],  # day_rad_list
            [torch.tensor([1.0, 2.0, 3.0])],  # generated_var_list
            [torch.tensor([1.0, 2.0, 3.0])]  # measured_var_list
        )

        # Run the function with custom model date and epoch
        make_boxplots("CHLA", "2024-01-01", 50)

        # Check that directories were created
        mock_makedirs.assert_called_once()

        # Check that the plotting function was called
        mock_show.assert_called_once()

        # Check that get_reconstruction was called with the correct custom parameters
        mock_get_reconstruction.assert_called_once_with("CHLA", "2024-01-01", 50, "test")


if __name__ == '__main__':
    unittest.main()

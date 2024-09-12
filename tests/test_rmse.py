import unittest
from unittest.mock import patch, MagicMock
import torch
import os
from ppcon.analysis.rmse import compute_rmse


class TestRMSEFunction(unittest.TestCase):

    @patch('ppcon.analysis.rmse.get_reconstruction')
    @patch('ppcon.analysis.rmse.os.makedirs')
    @patch('ppcon.analysis.rmse.plt.show')  # Mock plt.show to prevent actual plot windows during testing
    @patch('ppcon.analysis.rmse.plt.savefig')  # Mock plt.savefig to prevent actual file creation during testing
    def test_compute_rmse(self, mock_savefig, mock_show, mock_makedirs, mock_get_reconstruction):
        # Mock the get_reconstruction function to return controlled outputs
        mock_get_reconstruction.return_value = (
            [45.0, 46.0],  # lat_list
            [12.0, 13.0],  # lon_list
            [0.5, 0.6],  # day_rad_list
            [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.5, 2.5, 3.5])],  # generated_var_list
            [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.5, 2.5, 3.5])]  # measured_var_list
        )

        # Run the function
        compute_rmse("NITRATE", make_fig=True)

        # Ensure the directories were created
        mock_makedirs.assert_called()

        # Ensure the reconstruction function was called
        mock_get_reconstruction.assert_called_once_with("NITRATE", "2023-12-16", 100, "test")

        # Ensure the plotting functions were called when make_fig=True
        mock_savefig.assert_called()
        self.assertGreaterEqual(mock_savefig.call_count, 1)  # Ensure savefig was called at least once

        # Check console output for expected structure (mock print if necessary)
        # Alternatively, check for correct calculations if you mock the print statements


if __name__ == '__main__':
    unittest.main()

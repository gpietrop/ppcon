import unittest
from unittest.mock import patch, MagicMock
import torch
from ppcon.generate_profile import generate_profiles_from_file, generate_profiles_from_input


class TestGenerateProfiles(unittest.TestCase):

    @patch('ppcon.generate_profile.create_input_from_file')
    @patch('ppcon.generate_profile.preprocessing_data_user')
    @patch('ppcon.generate_profile.get_reconstruction_user')
    @patch('ppcon.generate_profile.moving_average')
    @patch('ppcon.generate_profile.plt.show')  # Mock plt.show to prevent actual plot windows during testing
    def test_generate_profiles_from_file(self, mock_show, mock_moving_average, mock_get_reconstruction_user, mock_preprocessing_data_user, mock_create_input_from_file):
        # Mock the functions to return controlled outputs
        mock_create_input_from_file.return_value = (2024, 1, 1, 45.0, 12.0, ([], []), ([], []), ([], []))
        mock_preprocessing_data_user.return_value = MagicMock()
        mock_get_reconstruction_user.return_value = ([45.0], [12.0], [0.5], [torch.tensor([1.0, 2.0, 3.0])], None)
        mock_moving_average.side_effect = lambda x, _: torch.tensor(x)  # Mock moving_average to return a tensor

        # Run the function
        result = generate_profiles_from_file("mock_file", "NITRATE")

        # Check that the functions were called
        mock_create_input_from_file.assert_called_once()
        mock_preprocessing_data_user.assert_called_once()
        mock_get_reconstruction_user.assert_called_once()
        mock_moving_average.assert_called()

        # Assert the function returns the expected output
        self.assertTrue(torch.equal(result, torch.tensor([1.0, 2.0, 3.0])))

    @patch('ppcon.generate_profile.preprocessing_data_user')
    @patch('ppcon.generate_profile.get_reconstruction_user')
    @patch('ppcon.generate_profile.moving_average')
    @patch('ppcon.generate_profile.plt.show')  # Mock plt.show to prevent actual plot windows during testing
    def test_generate_profiles_from_input(self, mock_show, mock_moving_average, mock_get_reconstruction_user, mock_preprocessing_data_user):
        # Mock the functions to return controlled outputs
        mock_preprocessing_data_user.return_value = MagicMock()
        mock_get_reconstruction_user.return_value = ([45.0], [12.0], [0.5], [torch.tensor([1.0, 2.0, 3.0])], None)
        mock_moving_average.side_effect = lambda x, _: torch.tensor(x)  # Mock moving_average to return a tensor

        # Run the function
        result = generate_profiles_from_input("NITRATE", 2024, 1, 1, 45.0, 12.0, ([], []), ([], []), ([], []))

        # Check that the functions were called
        mock_preprocessing_data_user.assert_called_once()
        mock_get_reconstruction_user.assert_called_once()
        mock_moving_average.assert_called()

        # Assert the function returns the expected output
        self.assertTrue(torch.equal(result, torch.tensor([1.0, 2.0, 3.0])))


if __name__ == '__main__':
    unittest.main()

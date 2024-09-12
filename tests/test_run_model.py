import unittest
from unittest.mock import patch, MagicMock
import os
from ppcon.run_model import run_training


class TestRunTraining(unittest.TestCase):

    @patch('ppcon.run_model.train_model')
    @patch('ppcon.run_model.FloatDataset')
    @patch('ppcon.run_model.plot_profiles')
    @patch('ppcon.run_model.save_ds_info')
    def test_run_training_default(self, mock_save_ds_info, mock_plot_profiles, mock_float_dataset, mock_train_model):
        # Mock the dataset loading
        mock_float_dataset.return_value = MagicMock()
        mock_float_dataset.return_value.__len__.return_value = 100  # Mock length of dataset

        # Run with default arguments
        run_training()

        # Ensure the dataset was loaded correctly
        self.assertTrue(mock_float_dataset.called)

        # Ensure the training function was called
        self.assertTrue(mock_train_model.called)

        # Ensure that the save and plot functions were called
        self.assertTrue(mock_save_ds_info.called)
        self.assertTrue(mock_plot_profiles.called)

    @patch('ppcon.run_model.train_model')
    @patch('ppcon.run_model.FloatDataset')
    @patch('ppcon.run_model.plot_profiles')
    @patch('ppcon.run_model.save_ds_info')
    def test_run_training_custom_args(self, mock_save_ds_info, mock_plot_profiles, mock_float_dataset,
                                      mock_train_model):
        # Mock the dataset loading
        mock_float_dataset.return_value = MagicMock()
        mock_float_dataset.return_value.__len__.return_value = 100  # Mock length of dataset

        # Run with custom arguments
        run_training(
            flag_toy=True,
            variable="CHLA",
            batch_size=16,
            epochs=10,
            lr=0.01,
            snaperiod=5,
            dropout_rate=0.5,
            lambda_l2_reg=0.0001,
            alpha_smooth_reg=0.0005,
            attention_max=10,
            flag_early_stopping=True
        )

        # Ensure the dataset was loaded correctly with toy flag
        self.assertTrue(mock_float_dataset.called)

        # Ensure the training function was called with the modified arguments
        self.assertTrue(mock_train_model.called)

        # Ensure that the save and plot functions were called
        self.assertTrue(mock_save_ds_info.called)
        self.assertTrue(mock_plot_profiles.called)

    def test_save_directory_creation(self):
        with patch('os.makedirs') as mock_makedirs:
            run_training()
            # Check that the directory creation was attempted
            mock_makedirs.assert_called()


if __name__ == '__main__':
    unittest.main()

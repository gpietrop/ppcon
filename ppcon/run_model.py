import os
import random
from datetime import date
import argparse

import torch
from torch.utils.data import DataLoader

from ppcon.config import DATASET_DIR, RESULTS_DIR
from ppcon.train.train import train_model
from ppcon.utils.dataset import FloatDataset
from ppcon.utils.plot_profile import plot_profiles
from ppcon.utils.utils_train import save_ds_info

# Setting the computation device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"We will use {device}")
random.seed(123)


def run_training(
        flag_toy=False,
        variable="NITRATE",
        batch_size=32,
        epochs=0,
        lr=1,
        snaperiod=25,
        dropout_rate=0.2,
        lambda_l2_reg=0.001,
        alpha_smooth_reg=0.001,
        attention_max=0,
        flag_early_stopping=False
):
    """
    Run the training process for a machine learning model with the specified parameters.

    :param flag_toy: bool, optional
        If True, use a toy dataset for training. Default is False.
    :param variable: str, optional
        The target variable to predict (e.g., "NITRATE"). Default is "NITRATE".
    :param batch_size: int, optional
        The number of samples per batch during training. Default is 32.
    :param epochs: int, optional
        The number of training epochs. Default is 0.
    :param lr: float, optional
        The learning rate for the optimizer. Default is 1.
    :param snaperiod: int, optional
        The period (in epochs) at which to save model snapshots. Default is 25.
    :param dropout_rate: float, optional
        The dropout rate to use in the model for regularization. Default is 0.2.
    :param lambda_l2_reg: float, optional
        The L2 regularization coefficient. Default is 0.001.
    :param alpha_smooth_reg: float, optional
        The smoothing regularization coefficient. Default is 0.001.
    :param attention_max: int, optional
        The maximum attention parameter for attention mechanisms in the model. Default is 0.
    :param flag_early_stopping: bool, optional
        If True, apply early stopping during training. Default is False.

    :return: None
    """

    # ===== Printing information about the run
    print(f"The variable predicted is {variable}\n"
          f"The total number of epochs that will be performed is {epochs}")

    train_dataset = FloatDataset(os.path.join(DATASET_DIR, variable, 'float_ds_sf_train.csv'))
    val_dataset = FloatDataset(os.path.join(DATASET_DIR, variable, 'float_ds_sf_test.csv'))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    save_dir = RESULTS_DIR + "/" + variable
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print(f"saving results in {save_dir}")

    # ===== Saving models hyperparameters
    save_ds_info(training_folder="default", flag_toy=flag_toy, batch_size=batch_size, epochs=epochs, lr=lr,
                 dp_rate=dropout_rate, lambda_l2_reg=lambda_l2_reg, save_dir=save_dir,
                 alpha_smooth_reg=alpha_smooth_reg)

    # ===== train the model
    train_model(train_loader, val_loader, epoch=epochs, lr=lr, dp_rate=dropout_rate, lambda_l2_reg=lambda_l2_reg,
                alpha_smooth_reg=alpha_smooth_reg, attention_max=attention_max, snaperiod=snaperiod, dir=save_dir,
                device=device, flag_early_stopping=flag_early_stopping)

    # ===== plot the results obtained on the validation set
    plot_profiles(DataLoader(val_dataset, batch_size=1, shuffle=True), variable=variable, dir=save_dir, ep=epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--flag_toy', type=bool, default=False)
    parser.add_argument('--variable', type=str, default="NITRATE", choices=["NITRATE", "CHLA", "BBP700"])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--snaperiod', type=int, default=25)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--lambda_l2_reg', type=float, default=0.001)
    parser.add_argument('--alpha_smooth_reg', type=float, default=0.001)
    parser.add_argument('--attention_max', type=float, default=0)
    parser.add_argument('--flag_early_stopping', type=bool, default=False)

    args = parser.parse_args()

    run_training(
        flag_toy=args.flag_toy,
        variable=args.variable,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        snaperiod=args.snaperiod,
        dropout_rate=args.dropout_rate,
        lambda_l2_reg=args.lambda_l2_reg,
        alpha_smooth_reg=args.alpha_smooth_reg,
        attention_max=args.attention_max,
        flag_early_stopping=args.flag_early_stopping
    )

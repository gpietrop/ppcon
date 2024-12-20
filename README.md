# ppcon

Python library for the implementation of PPCon (Profile Prediction with 1D Convolutional Networks)

```bash
pip install -r requirements.txt 
```

## Training PPCon
To train the PPCon model, import the following function from the library:

```python
from ppcon.run_model import run_training

run_training(variable="NITRATE",
             batch_size=32,
             epochs=100,
             lr=1,
             snaperiod=25,
             dropout_rate=0.2,
             lambda_l2_reg=0.001,
             alpha_smooth_reg=0.001,
             attention_max=0,
             flag_early_stopping=False
             )
```

where the inputs arguments stand for: 
* `--variable` is the biogeochemical variable considered (that can be: _NITRATE_, _CHLA_, _BBP700_).  
* `--batch_size` is the batch size for training.
* `--epochs` is the number of epochs for training.
*  `--lr` is the learning rate for training.
*  `--dropout_rate` is the dropout rate for training.
*  `--snaperiod` is the number of epochs after which the intermediate model is saved.
*  `--lambda_l2_reg` set the multiplicative loss factor for the lambda regularization.
* `--alpha_smooth_reg` set the multiplicative loss factor for the smooth regularization.
* `--attention_max`: max value for attention mechanism, if applicable.
* `--flag_early_stopping`: boolean flag to enable or disable early stopping.


The function `run_training()` will train the PPCon architecture with the same train and test dataset referenced in the original paper.

### Results and Models
The results, along with the trained models, will be automatically saved in the `results_ppcon` directory, 
located in the user's home directory.
Each subdirectory is specific to the chosen `variable` and includes: 
* The date of the model training 
* `.pt` model checkpoint files for different epochs.
* Logs containing training and testing loss details.

### Running the Script from the Command Line
You can also run the training script directly from the command line, without needing to import the library:
```bash
python3 run_model.py --variable <VARIABLE> --epochs <EPOCHS> --lr <LEARNING_RATE> --dropout_rate <DROPOUT_RATE> --snaperiod <SNAPSHOT_PERIOD> --lambda_l2_reg <L2_REG_STRENGTH> --batch_size <BATCH_SIZE> --alpha_smooth_reg <SMOOTH_REG_STRENGTH>
```
This allows for more flexibility when customizing the training process.

### Using Custom Datasets

To use a different dataset, you can replace the training and testing datasets in the `ds/variable/float_ds_sf_{train/test}.csv` files.
The file structure for datasets is predefined, and users can modify the contents by substituting their own datasets in the specified location.

### Modifying the Model Architecture
If running the script from the command line, you can also modify the model architecture. For example, you can add or remove convolutional layers. The default architectures are located in:
* `train/conv1med_dp.py` for convolutional layers.
* `train/mlp_dp.py` for multi-layer perceptron (MLP) layers.

## Running pretrained ppcon

Pretrained models can be used to predict profiles.  
The library allow to predict vertical profiles with the PPCon pretrained architerture starting from input provided by the users

### Predicting profiles starting from input

To use the PPCon model on a user input, import the following function from the library:


```python
from ppcon.generate_profile import generate_profiles_from_input

generate_profiles_from_input(variable="NITRATE", 
                             year=2024, 
                             month=1, 
                             day=1, 
                             lat, 
                             lon, 
                             tuple_temp, 
                             tuple_psal, 
                             tuple_doxy,
                             tuple_var=None, date_model=None, epoch_model=None)
```

where the inputs arguments stand for: 
* `"MR6901648_109"` is the path 
* `"CHLA"` is the variable that the user want to predict 

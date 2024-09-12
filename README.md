# PPCon 1.0: Biogeochemical Argo Profile Prediction with 1D Convolutional Networks

Python implementation for paper "PPCon 1.0: Biogeochemical Argo Profile Prediction with 1D Convolutional Networks", 
Geoscientific Model Development - 2024:

## Abstract
Effective observation of the ocean is vital for studying and assessing the state and evolution of the marine ecosystem, and for evaluating the impact of human activities. 
However, obtaining comprehensive oceanic measurements across temporal and spatial scales and for different biogeochemical variables remains challenging. 
Autonomous oceanographic instruments, such as Biogeochemical (BGC) Argo profiling floats, have helped expand our ability to obtain subsurface and deep-ocean measurements, but measuring biogeochemical variables such as nutrient concentration still remains more demanding and expensive than measuring physical variables. 
Therefore, developing methods to estimate marine biogeochemical variables from high-frequency measurements is very much needed. 
Current Neural Network (NN) models developed for this task are based on a Multilayer Perceptron (MLP) architecture, trained over point-wise pairs of input-output features.
Although MLPs can produce smooth outputs if the inputs change smoothly, Convolutional Neural Networks (CNNs) are inherently designed to handle profile data effectively.
In this study, we present a novel one-dimensional (1D) CNN model to predict profiles leveraging the typical shape of vertical profiles of a variable as a prior constraint during training. 
In particular, the Predict Profiles Convolutional (PPCon) model predicts nitrate, chlorophyll and backscattering (bbp700) starting from the date, geolocation, and temperature, salinity, and oxygen profiles. 
Its effectiveness is demonstrated using a robust BGC-Argo dataset collected in the Mediterranean Sea for training and validation. 
Results, which include quantitative metrics and visual representations, prove the capability of PPCon to produce smooth and accurate profile predictions improving previous MLP applications.

## Instructions

The code runs with Python 3.8.5 on Ubuntu 20.04 and macOS. Install the required packages using:

```bash
pip install -r requirements.txt 
```

To run the code, enter the following command:

```bash
python3 run_model.py --variable --epochs --lr --dropout_rate --snaperiod --lambda_l2_reg --batch_size --alpha_smooth_reg
```
where the inputs arguments stand for: 
* `--variable` is the biogeochemical variable considered (that can be: _NITRATE_, _CHLA_, _BBP700_)  
* `--epochs` is the number of epochs for the training
*  `--lr` is the learning rate for the training
*  `--dropout_rate` is the dropout rate for the training
*  `--snaperiod` is the number of epochs after which the intermediate model is saved
*  `--lambda_l2_reg` set the multiplicative loss factor for the lambda regularization
*  `--batch_size` is the batch size for the training
*  `--alpha_smooth_reg` set the multiplicative loss factor for the smooth regularization

The datasets are preprocessed and stored in tensor form, ready for training. 
They are split into training and testing sets and can be found in the  _ds_ folder. 

The results and models from the paper are located in the `results`, directory, which contains subdirectories for each variable. 
Each subdirectory includes the date of the model training,  `.pt` files for different epochs, and information about training and testing loss. 

Pretrained models can be used to generate new data. 
An example of how to generate new data using the pretrained model can be found in  `make_genrated_dataset/make_generated_ds.py`, 
specifically in the function `get_reconstruction`.

The scripts for reproducing the plots from the paper are located in the `analysis` folder. 
Example usage can be found in the `analysis/main_analysis.py` function. 
More specifically:
* To get __Figure 3-5__ the functions are contained in _analysis/comparison_architecture.py_
* To get __Figure 6__ the functions are contained in _analysis/scatter_error.py_
* To get __Figure 7-9__ the functions are contained in _analysis/hovmoller_diagram.py_
* To get __Figure B1__ the functions are contained in _analysis/profile.py_

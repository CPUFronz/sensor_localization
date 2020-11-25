# Localization from Activity Sensor Data

This is the code for the [SenSys '20](http://sensys.acm.org/2020/) poster ["Localization from activity sensor data"](https://dl.acm.org/doi/10.1145/3384419.3430404).

## Setup

A Conda *.yml file is provided for recreating the development environment. Just run ``conda env create -f environment.yml`` to install all necessary packages.

## How to Run

``train_models.py`` trains the model using [auto-sklearn](https://automl.github.io/auto-sklearn/), it is set to take 3 hours
``localize.py`` runs the localization and puts the results into a *.csv file
``results.ipynb`` visualizes the results and stores the plots, we used in the paper

To run, activate the Conda environment, place the data into the data folder and type ``python train_models.py && python localize.py``.

## Citing

If you find our work useful, please cite it using the following BibTex entry:
```
@inproceedings{papst2020,
  title = {Localization from activity sensor data},
  author = {Papst, Franz and Stricker, Naomi and Saukh, Olga},
  booktitle = {Proceedings of the 18th Conference on Embedded Networked Sensor Systems},
  pages = {703--704},
  year = {2020}
}
```

# Star Cluster Kinematic Classification Project

## Overview

This project implements a machine learning approach to classify astronomical star clusters (Open vs. Globular) based primarily on their kinematic properties (how their member stars move). It utilizes data from the Gaia DR3 mission and leverages Python libraries for data acquisition, analysis, and machine learning.

This project was developed in response to the Astronomy prompt for the STEAM Innovation Challenge 2025, which encourages improving star cluster classification methods to better understand stellar and galactic evolution.

## Features

* Queries the Gaia DR3 archive for stellar data associated with known star clusters.
* Applies astrometric and kinematic filters to identify likely cluster members.
* Calculates cluster-level kinematic features, including:
    * Internal velocity dispersion (3D and components)
    * Orbital parameters (Energy, Angular Momentum Lz) using the `gala` library.
* Trains a Random Forest classifier to distinguish between Open and Globular clusters based on these kinematic features.
* Evaluates the model's performance using standard metrics (accuracy, precision, recall, confusion matrix).
* Identifies the most important kinematic features for the classification task.

## Project Structure

* `1_data_acquisition.py`: Script to query Gaia, clean data, and save likely cluster members to `cleaned_cluster_members.fits`.
* `2_ml_classification.py`: Script to load cleaned data, perform feature engineering, train the Random Forest model, and evaluate its performance. Outputs evaluation metrics and plots (`confusion_matrix.png`, `feature_importance.png`).
* `CODE_EXPLANATION.md`: Detailed explanation of the Python scripts.
* `README.md`: This file.

## Dependencies

This project requires Python 3.x and the following libraries:

* numpy
* pandas
* astropy
* astroquery
* scikit-learn
* gala-astro
* matplotlib
* seaborn

You can install them using pip:

```bash
pip install -r requirements.txt
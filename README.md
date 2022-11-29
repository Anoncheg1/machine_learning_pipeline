# Machine learning pipeline
semi-automatic data preparation for machine learning

# Steps of pipeline:
1. pre_process() - select required parts of data, save id field.
2. process_by_hands() - domain-specific transformations
3. split() - split data to train and test
4. select_and_set_types()
  - select model specific fields
  - remove columns with one value
  - encode columns boolean to [0;1] and convert strings with value '0.1234' to float or integer type
5. outliers_numerical() - remove outlier records which value is lower than 0.0001 and greater than 0.9999 qunatile
6. fill_na()
  - replace NA values for numerical with mean, for categorical with most frequent
  - if categorical columns has count of NA values greater than 0.6 or two unique values we do not replace NA
  - if numerical columns has two unique values one of which is NA we save NA we do not replace NA
  - convert integer types to int, convert decimal numbers to float
7. sparse_classes() - for columns with count of unique values greater than 60 replace values which is less than 1% of rows count with string values of "others". Separate for testing and trainging dataset.
8. encode_categorical_pipe() -
  - encode categorical columns with one-hot encoding if count of unique values is between 2 and 10
  - else encode categorical columns with label encoder.
  - if after label encoder there is NA values, they are
  - Convert all columns to float. Separate for testing and trainging dataset.
  - for both testing and trainging dataset we use same label encoders to preserve consistency
9. remove columns which is not in both training and testing dataset and sort columns in the same order.


# Clusterization pipeline:
1. manually select columns
2. agressive remove outliers
3. encode categorical columns
4. standardization and (optional) manually centering for PCA
5. feature engeneerig - feature synthesis
6. manually divide features to lower their importance
7. PCA transformation (optional)
7. hierarchical clustering / Affinity Propogation/ Gaussian Mixture

# Notes
In all steps we keep Id field to check consistency and be able to recover original field by id from processed.

# Files:
- myown_pack - shared pipeline library
- pipeline_mart_norma_auto.py - pipeline work example
- pipeline_otchet_cluster_analysis.py - pipeline for cluster analysis
- pipeline_otchet_cluster_analysis_plot.py - plot functions for pipeline_otchet_cluster_analysis.py

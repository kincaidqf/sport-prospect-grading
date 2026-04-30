# MLflow Logging Specification

## Summary

Log only the information needed to explain:
- what model was run
- what settings changed
- how performance changed over training
- which final result should be reported in the paper

MLflow logging must be organized by:
- experiment name
- parent run
- model family

Only the loss-curve PNG should be logged to MLflow as a visual artifact. Other visuals remain local.

## Required Organization

### Experiment name
Each run must belong to a clear experiment name so related runs can be grouped together.

### Parent run
Each top-level training invocation should create one parent run.
The parent run should contain the high-level summary of the experiment.

### Model family
Each run must clearly identify the model family, such as:
- regression
- classification
- text

This is necessary so results can be filtered and compared across model types.

## Essential Logging Categories

### 1. Run identity
- run name
- experiment name
- parent run
- model family
- target variable
- date/time
- git commit SHA

Why:
This identifies exactly which experiment produced a result.

### 2. Model setup
- full resolved config as one artifact
- core model choice / estimator name
- backbone name for text models
- key architecture settings only
  - hidden dimension
  - output dimension
  - dropout
  - freeze/unfreeze setting
- whether draft pick was used as a feature

Why:
This is the minimum needed to describe the model in a short report.

### 3. Hyperparameters
- learning rate
- batch size
- number of epochs
- weight decay
- early stopping patience
- gradient clip
- random seed
- optimizer name
- search-space values actually used for tuning
  - alpha / C
  - XGBoost grid values
- final selected hyperparameter values

Why:
These are the main knobs that affect performance and need to be tied to results.

### 4. Data and split summary
- dataset size
- train / validation / test sizes
- split definition
  - random split or year split
- train / validation / test year ranges when applicable
- class balance for classification
- target distribution summary for regression

Why:
This is essential context for interpreting the results section.

### 5. Training progression over time
Per epoch:
- train loss
- validation loss

Also log:
- best validation loss
- best epoch
- total epochs completed

Artifact:
- loss-curve PNG

Why:
This is the clearest evidence of learning progress and possible overfitting.

### 6. Final performance metrics
For the final selected model, log:

Regression:
- R2
- RMSE
- MAE

Classification:
- accuracy
- ROC-AUC

If available, keep these for both validation and test, but test metrics are the most important for the report.

Why:
These are the headline quantitative results for the paper.

### 7. Model selection / tuning result
- which candidate won
- final selected hyperparameter values
- validation score used to choose the winner
- compact summary artifact of tested candidates and their validation scores

Why:
This is enough to justify why the final model was chosen without logging every possible detail.

### 8. Reproducibility metadata
- Python version
- main library versions
  - torch
  - transformers
  - scikit-learn
  - xgboost
  - mlflow
- device used
  - CPU
  - MPS
  - CUDA

Why:
This supports reproducibility in an undergraduate research setting without overloading the report.

## MLflow Artifacts To Keep

Required:
- resolved config file
- loss-curve PNG
- compact candidate-summary artifact
- final trained model artifact

Not required in MLflow:
- confusion matrix PNG
- ROC PNG
- residual plot PNG
- feature-importance PNGs
- heatmaps
- other local visual outputs

## Assumptions

- Only essential information for a short CS research report should be logged.
- Loss progression is the only visual artifact that must go to MLflow.
- Candidate-level logging should be summarized, not exhaustive.
- Split-level metrics are enough; subgroup/cohort breakdowns are not required.

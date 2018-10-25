Sberbank Data Science Journey 2018: AutoML
==========================================

This is general information about [Sberbank Data Science Journey 2018](http://sdsj.sberbank.ai/)

## Description

SDSJ AutoML — AutoML competition aimed at machine learning models that automatically process data, completely automatically choosing models, architectures, hyper-parameters, etc.

## Data description

In this competition, you will be predicting based on different datasets in [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) format. 

You will find the following columns:

- `line_id` — an Id for each line
- `target` — target variable (only for train dataset), continuous variable for regression tasks and binary labels (0/1) for classification
- `<type>_<feature>` — type of the feature (`type`):
    - `number` — number feature (also could be continuous, categorical or binary variable)
    - `string` — string feature
    - `datetime` — date feature in `2010-01-01` or `2010-01-01 10:10:10` format
    - `id` — Id (special purpose categorical variable)


## Submission Instructions

The model compressed in ZIP file should be submitted to evaluation system. Submissions will run in local environment using [Docker](https://www.docker.com/what-docker), time and resources for testing are limited. In common case, this is no necessary to be expirienced with Docker for participant.

In root folder of zip archive must be `metadata.json` file containing follow:

```json
{
    "image": "sberbank/python",
    "entry_points": {
        "train_classification": "python train.py --mode classification --train-csv {train_csv} --model-dir {model_dir}",
        "train_regression": "python train.py --mode regression --train-csv {train_csv} --model-dir {model_dir}",
        "predict": "python predict.py --test-csv {test_csv} --prediction-csv {prediction_csv} --model-dir {model_dir}"
    }
}
```

There are:
`image` — the name of docker image, that will run the submission,   
`entry_points` — commands which run the submission (`train_*` — train models for classification and regression, `predict` — prediction with that traineв models). Root directory for the submission will be root directory of zip-achive.

Commands should match the patterns, that will replaced with necessary values dirung execution in test system:
- `{train_csv}`, `{test_csv}` — path to CSV-file containing train or test data
- `{model_dir}` — path to previously created directory that must contain trained model for using
- `{prediction_csv}` — path to file containing predictions

When submission will run, in evironment variable `TIME_LIMIT` will set maximum time (in sec) for the model execution.

It is guaranteed that the model will have at least 300 sec for train and prediction, however for big datasets this limit will be extended.

For running submission follow environments could be used:

- `sberbank/python` — Python3 with many libraries imported ([details](images/sberbank-python))
- `gcc` - for C/C++ submissions
- `node` — for JavaScript
- `openjdk` — for Java
- `mono` — for C#

Also any other image could be used if it is available on [DockerHub](http://dockerhub.com). If it will be needed, you could build your own image with the requred software and libraries(see [instruction](https://docs.docker.com/engine/reference/builder/)); you have to publish it on DockerHub for using.

## Requirements

There are requirements for running submission in container:

- available resources
  - **12 Gb** RAM
  - 4 vCPU
- submission will not have access to internet
- maximum size of extracted submission archive **1 Gb**
- submission will extracted to file system архив in RAM (ramfs), available for writing
- other container content available only for reading
- dataset CSV have size less than **3 Gb**


## Evaluation

1. There is specific metrics will be calculated for each dataset on its test part (RMSE for regression, ROC-AUC for binary classification).
2. We calculate common score for every participant based on scores for each dataset as follow: 
- best score (of all success and tested submits) get 1 point, baseline equals 0 points
- participants who are between best score and baseline on leaderboard get scaled weighted points from 0 to 1, depends on their position
- submits lower baseline on score get 0 points
- if best score and baseline score are the same, all participants get 0 points
- if submission return error or don't met time limits it gets 0 points

3. Total score for the participant we calculate as sum of weighted points for each dataset. On public leaderboard order based on total score.


## Examples: submits and datasets


### How to: local validation

This example running on [baseline](https://github.com/sberbank-ai/sdsj2018-automl) and public kernel [vlarine](https://github.com/vlarine/sdsj2018_lightgbm_baseline).

Public datasets for local validation: [sdsj2018_automl_check_datasets.zip](https://s3.eu-central-1.amazonaws.com/sdsj2018-automl/public/sdsj2018_automl_check_datasets.zip)

### 1. Docker

First of all you should install Docker for your OS ([details](https://docs.docker.com/install/)). After that pull docker image from DockerHub (it takes some time):

`docker pull sberbank/python`

> Please note that you should have about 20Gb free on your HDD  
> And if you are a Mactard your Docker have memory limits which could be changed in Docker preferences on "Advanced" tab 

### 2. Train the model

There is how to run model training on first dataset:
```
docker run \
  -v {workspace_dir}:/workspace_0001 \
  -v {train_csv}:/data/input/train.csv:ro \
  -v {model_dir}:/data/output/model \
  -w /workspace_0001 \
  -e TIME_LIMIT=300 \
  --memory 12g \
  --name solution_0001_train \
  sberbank/python \
  python train.py --mode classification --train-csv /data/input/train.csv --model-dir /data/output/model
```

Here:

- `{workspace_dir}` — directory which contains `metadata.json`;
- `{model_dir}` — directory which will contain thained model (should be created before);
- `{train_csv}` — file contains train dataset

> Every path should be absolute

After training this container shoud be stopped or else command with the same image name will not running:
```
docker stop solution_0001_train
docker rm solution_0001_train
```
The same is valid for `solution_0001_test`.

### 3. Validation

Now you should do prediction as follow:

- `{prediction_dir}` — directory which contains file with predictions
```
docker run \
      -v {workspace_dir}:/workspace_0001 \
      -v {test_csv}:/data/input/test.csv:ro \
      -v {model_dir}:/data/input/model \
      -v {prediction_dir}:/data/output \
      -w /workspace_0001 \
      -e TIME_LIMIT=300 \
      --memory 12g \
      --name solution_0001_test \
      sberbank/python \
      python predict.py --test-csv /data/input/test.csv --model-dir /data/input/model --prediction-csv /data/output/prediction.csv
```
After that compare predition with corresponding test-target.csv file, calculate the score and evaluate your model.

> It was example for first dataset - valitation process for other datasets is the same

## Useful links

- [Description](https://www.automl.org/blog-2nd-automl-challenge/) of winner's submissions: Chalearn AutoML challenge (2017-2018) and the archive [with this submission](http://ml.informatik.uni-freiburg.de/downloads/automl_competition_2018.zip)

- [Some articles](http://www.fast.ai/2018/07/12/auto-ml-1/) regardong AutoML from [fast.ai](http://www.fast.ai/)

- [Book](https://www.ml4aad.org/book/) about AutoML

- [Open repositories](https://sdsj.sberbank.ai/ru/start) of SDSJ participans

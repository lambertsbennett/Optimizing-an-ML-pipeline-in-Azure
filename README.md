# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about individuals applying for bank loans. The task we set out to accomplish here is to develop a model that,
based on the information provided about each applicant, predicts whether they will receive a loan.

The best performing model was found to be a voting ensemble with 91.8% accuracy. However, many models were of comparable accuracy.

## Scikit-learn Pipeline
The Scikit-learn pipeline obtains the provided data from the provided URL. Following data download, a number of data cleaning steps are carried out including:
- Removing NAs from the dataset.
- One-hot encoding job titles, contact, and education variables.
- Encoding a number of other categorical variables.
- Encoding months of the year.
- Encoding the target variable.

Once the data has been prepared it is split into a training and test set. A test set size of 33% of total entries was selected as a compromise between ensuring adequate representation in the test data and providing sufficient data for model training. 

The classification method used here is logistic regression. Logistic regression uses a fitted logistic function and a threshold. The parameters available within the training script are C (which indicates the regularization strength i.e. preference for sparser models) and maximum number of iterations.

Azure's Hyperdrive service was used for hyperparameter tuning with the following key elements:

#### Parameter sampling
I decided to use random parameter sampling. I have very little background in using logistic regression and figured that choosing a broad parameter space and using random sampling would be a way to hopefully get a performant model.

#### Early stopping policy
I selected the BanditPolicy stopping policy because it allows one to select a cut-off at which models reporting metrics worse than the current best model are terminated. This allows a relatively intuitive method to screen models, only retaining those with similar or better performance. This policy offers a little more flexibility than truncation and median stopping.

The best model parameters here were a C value of 7.22 and a max_iter value of 97. The model's accuracy was ~91.4%.

## AutoML
The autoML pipeline is very similar to the Scikit-learn pipeline described above with several notable differences:
- The data are retrieved from the provided URL.
- The data are cleaned using the same process as described above. 
- The data are **not** split into train and test sets.
- The variables and target dataframes are merged prior to the autoML process.
- The joined dataset is used as input in the autoML configuration and the autoML run is processed locally.

The best model selected by autoML was a voting ensemble (~91.8% accurate). The model selected used a slight amount of l1 regularization, meaning that some penalty was placed the number of non-zero model coefficients.

## Pipeline comparison
The two models performed very similarly in terms of accuracy, with the hyperdive model achieving 91.4% accuracy and the autoML model achieving 91.8% accuracy. The difference in accuracy could come down to slight variations in the cross-validation process. Architecturally, the models are quite different. I'm not entirely certain what a "Voting Ensemble" is, but I assume it is making predictions based on the output of a variety of models, similar to other ensemble methods. 

## Future work
In the future it might be helpful to explore more feature engineering steps prior to training. Also, many of the AutoML runs use a scaler prior to model training and evaluation. The encoded data does not really benefit from this scaling, so selectively scaling continuous variables instead of all, might be helpful. Also, running AutoML for much longer would likely find better models in this case.


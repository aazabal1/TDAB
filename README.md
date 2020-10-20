# TDAB
The purpose of this code is to answer the following case study:
```
The Client is a marine electronics company. They produce hardware and software for sailing yachts. They have installed some sensors on one of the boats and provide the dataset they’ve collected. They don’t have any experience with machine learning, and they don’t have a solid understanding of what exactly they want to do with this data. The .csv file with the data and data dictionary are provided.
```
```
Your first task is to analyse the data, give the client some feedback on data collection and handling process, and suggest some ideas of how this data can be used to enhance their product and make it more popular among professional sailors and boat manufacturers.

Your supervisor told you that on top of whatever you come up with, what you should definitely do is ‘tack prediction’. “A tack is a specific maneuver in sailing and alerting the sailor of the necessity to tack in the near future would bring some advantage to them compared to other sailors, who would have to keep an eye out on the conditions all the time to decide when to tack,” he writes in his email. The supervisor, who has some experience in sailing labels the tacks in the data from the client (added as ‘Tacking’ column in the data).

Your second task is to build a forecasting model that would be alerting sailors of the tacking event happening ahead.
````
### 1. Explore Data
To do so, first the file called ```Data_Exploration.py``` contains all steps to explore the raw data before building any machine learning model.
### 2. Clean Data
Next run ```data_cleaning.py``` to clean the raw data.
### 3. Select features to input to model
Select the appropriate features to input into the model from ```select_features.py```. These features have been selected using the ```Feature_Engineering.ipynb```notebook.
### 4. Create input training and test data
Use the file ```data_input_model.py``` to create the input data and split it into the test and train datasets
### 5. Train model
Train the model by running ```model_train.py```
### 6. Get predictions and evaluate results
Obtain predictions and evaluate these results by running the ```evaluate_model.py``` code

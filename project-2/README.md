# Disaster Response Pipeline Project

## by Darren Gidado
 
![jpg](images/aid.jpg)
 
### Introduction:

In this project, we used data engineering skills to analyze disaster data from Figure Eight. This data was used to build a model for an API that classifies disaster messages.

In the project folder there is a data set containing real messages that were sent during disaster events. A machine learning pipeline was created to categorize these events so that the messages could be sent to an appropriate disaster relief agency.

This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. This project showscases software skills, including creating basic data pipelines using clean, organized code.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/

![png](images/disaster-response-project1.png)

![png](images/disaster-response-project2.png)

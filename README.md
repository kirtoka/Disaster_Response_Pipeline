# Disaster Response Pipeline

This is a project for the UDACITY Nanodegree "Data Scientist"

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)


## Table of content
* [Project Overview](#Chap1)
* [File descriptions](#Chap2)
* [Requirements](#Chap3)
* [Installation instructions](#Chap4)
* [Project conclusion](#Chap5)
* [License](#Chap6)


## 🌻 Project Overview <a name=Chap1></a>

This project is part of the Udacity Data Scientist Nanodegree Program.

After a natural disaster, thousands of people will send out messages to ask for help, inform and warn other people such as 'I need food'; 'Please, we need tents and water'; 'I would like to know if the earthquake is over';'There's nothing to eat and water, we starving and thirsty'. Thousands of such messages pop up during a very short time. It is hard to filter out meaningful information without wasting too much time. The government does not have enough time to read all the messages and send them to various departments. The goal of this project is to provide a multi-class classifier to categorize such disaster messages.

Starting with raw csv-files that contain messages about disasters from different sources (e.g., social media), we define
pipelines that clean/transform the information and build a model to classify the disaster messages in certain categories. More detailed: 
1. ETL pipeline that cleans, transforms, and stores the raw message data. 
2. ML pipeline that creates a model that can classify new disaster messages. 
3. A web-app that nicely visualizes the data and that can be used to classify new disaster messages.  

See the screenshots below to get an idea how the web-app looks like.  

![Start screen](./screenshots/Screenshot_2.png?raw=true "Enter message")

![Request](./screenshots/Screenshot_3.png?raw=true "Result")

![Request](./screenshots/screen2.png?raw=true "Chart1")

![Request](./screenshots/screen3.png?raw=true "Chart2")


## 🍀 File descriptions <a name=Chap2></a>
├── data\
│ ├── DisasterResponse.db \
│ ├── disaster_categories.csv \
│ ├── disaster_messages.csv\
│ ├── helper_etl_pipeline.ipynb >>> Helper Jupyter Notebook explaining steps for process_data.py \
│ └── process_data.py >>> ETL pipeline - a Python script that loads the messages and categories datasets\
                          merges the two datasets,cleans the data,stores it in a SQLite database\
├── models\
│ ├── functions.py >>> File containing helper functions \
│ ├── classifier.pkl >>> Pretrained model\
│ ├── helper_ml_pipeline.ipynb >>> Helper Jupyter Notebook explaining steps for train_classifier.py \
│ └── train_classifier.py >>> ML pipeline - a Python script that builds a text processing and machine learning pipeline\
                              which trains and tunes a model using GridSearchCV, and then exports the final model as classifier.pkl\
├── screenshots\
│ ├── request_1.PNG \
│ ├── start_screen.PNG \

├── web_app\
│ ├── static\
│ │ ├── logos\
│ │ │ ├─ githublogo.png\
│ │ │ └─ linkedinlogo.png\
│ ├── templates\
│ │ ├─ go.html\
│ │ └─ master.html\
│ └── run.py

├── requirements.txt
├── README.md


## 📙 Requirements <a name=Chap3></a>

This project uses the following python libraries:
* [plotly](https://plotly.com/): Interactive, open-source, and browser-based graphing library for Python
* [nltk](https://www.nltk.org/): Natural Language Toolkit library
* [pandas](https://pandas.pydata.org/): Library to handle datasets
* [re](https://docs.python.org/3/library/re.html): Library to handle regular expressions
* [scikit-learn](https://scikit-learn.org/stable/): Machine Learning library 
* [argparse](https://docs.python.org/3/library/argparse.html): Parser for command-line options
* [sqlalchemy](https://www.sqlalchemy.org/): Library to handle SQL databases
* [collections](https://docs.python.org/3/library/collections.html): Container database library
* [flask](https://flask.palletsprojects.com/en/2.0.x/): Python based web framework
* [joblib](https://pypi.org/project/joblib/): Library to read/write `.pkl` files
* [os](https://docs.python.org/3/library/os.html): Miscellaneous operating system interfaces
* [sys](https://docs.python.org/3/library/sys.html): System specific params and functions
* [tqdm](https://tqdm.github.io/): Library for nice progress bar visualization 
* [warnings](https://docs.python.org/3/library/warnings.html): Warning control library


## ⛳ Installation instructions <a name=Chap4></a>

0. Installation requirements: Python 3.8.8 or higher
1. Install the required Python packages using a virtual environment and the `requirements.txt` file. 

```console
py -3 -m venv myvenv
myvenv/Scripts/activate
pip install -r requirements.txt
```
2. Build a model by running the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        ```console
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        ```
    - To run ML pipeline that trains and saves the classifier  
        ```console
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
        ```  
3. Switch to the app's directory to run your web app.
    ```
    cd app
    py run.py
    ```

4. Go to localhost:3001/ or http://0.0.0.0:3001/


## ✔️ Project conclusion <a name=Chap5></a>

In this project, a baseline multi-class classifier and several approaches to improve this baseline classifier has been
trained and tested, see [helper_ml_pipeline jupyter notebook](./models/helper_ml_pipeline.ipynb). 
Unfortunately, most experiments did not lead to a significant improvement of the baseline classifier. Therefore,
further investigations should be done to make the model better.  
One idea might be to use Support Vector Classification that showed good results for the recall at least
for the category `water`. However, the training for all categories takes a long time.


## 🙏 Licensing, Acknowledgements <a name=Chap6></a>
Authors: Natalia Chirtoca.
This project is published in 2021 under [MIT](https://es.wikipedia.org/wiki/Licencia_MIT) license.
csv-files are provided by [Figure Eight](https://www.figure-eight.com/) ([appen](https://appen.com/) 
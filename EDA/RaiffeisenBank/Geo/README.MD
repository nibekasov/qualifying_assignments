# Social Activity Monitor (SAM) Analysis

## Results of research

1.1 Have the users been to the same places? Do they know each other?
All 3 people visited the same places, but the relationship could only be established between 2 and 3, but the fact that 1 visited similar points on the map could be useful for some modelling or marketing actions.

1.2 Introduce a social activity metric - it should take into account the number of places a user visited, how far apart they are and how long they stayed - the metric should allow us to compare users and find the most active ones.
Depending on the method we will find the most active customers if as follows 2 3 1 if we don't use external data If we use external data user 2 1 3 is most active because 3 don't visit points of interest like bars, cinemas If we use distance as one of the main metrics (I think it's not realy useful because he travels to Argentina, but if he would visit iceland he would be counted as less active) 1 2 3 are most socially active.

So depending on our analysis few takeaways schould be considered:
1. Best approach to determine activity is amaount of visited points of interest that also relies on time spend and distance between people ( with some not trivial function as hormonic mean for example). This approach shows that treveler is more active ( but data is not full enoght on longer distances that could change) that could be controversial. So i used other approach where wheights of distances is less important, than 2 persone is more active, cause more active in social hours and visits more social places as bars, theathers and so on
2. Using time of the day solely as feature could be controversial, cause if person in other part of the world his activity could shift a lot - so we should consider this in ananlysis
3. We need to determine schould we consider regular life of hollidays life too? It's controversial because at hollidays people shift lifestile a lot. If we would consider some analysis based on holidays we schould extract point of interest for all possible destinations of our people. It is crucial so we could calculate metrics better 

## Few wordsfor reviewer

Solution in one file with visualisation and outputs: 
https://github.com/nibekasov/qualifying_assignments/blob/main/EDA/RaiffeisenBank/Geo/Task_RaiFaizenBank.ipynb

The main part of the code is in here ( same nn\otebook but without unnesesery functions): https://github.com/nibekasov/qualifying_assignments/blob/main/EDA/RaiffeisenBank/Geo/notebooks/02_data_analysis.ipynb

All functions used for this notebook is in this directory: 
https://github.com/nibekasov/qualifying_assignments/tree/main/EDA/RaiffeisenBank/Geo/src

I recomend to focus mostly on this file with preprocessing and metrics calculation: 
https://github.com/nibekasov/qualifying_assignments/blob/main/EDA/RaiffeisenBank/Geo/src/data_analysis.py

In other py files we could find functions to extract data to local data base, functions for visualuisaation and other preprocessing.
## Project Description

The Social Activity Monitor (SAM) Analysis project aims to provide insights into social behaviors by analyzing geolocation data. It leverages data processing techniques to quantify social activity levels based on location data, points of interest (POIs), and time spent in different locales. The project is designed to assist researchers and businesses, such as banking institutions, to understand social patterns for various applications, including credit risk analysis and targeted marketing.

## Installation

Instructions on setting up the project environment.

```bash
git clone https://github.com/nibekasov/qualifying_assignments/edit/main/EDA/RaiffeisenBank/Geo.git
cd Geo
pip install -r requirements.txt
```
## Structure of project potential
```bash
project_name/
│
├── data/
│   ├── raw/                        # Raw data, immutable
│   └── processed/                  # Cleaned and processed data
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   └── 02_data_analysis.ipynb
│ 
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── data_analysis.py
│   └── visualization.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data_preprocessing.py
│   ├── test_data_analysis.py
│   └── test_visualization.py
│
├── Task_RaiFaizenBank.html
├── Task_RaiFaizenBank.ipynb        # One file solution
├── requirements.txt                # Project dependencies
└── README.md                       # Project description and instructions

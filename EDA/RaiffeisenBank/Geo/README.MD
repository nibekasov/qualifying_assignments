# Social Activity Monitor (SAM) Analysis

## Project Description

The Social Activity Monitor (SAM) Analysis project aims to provide insights into social behaviors by analyzing geolocation data. It leverages data processing techniques to quantify social activity levels based on location data, points of interest (POIs), and time spent in different locales. The project is designed to assist researchers and businesses, such as banking institutions, to understand social patterns for various applications, including credit risk analysis and targeted marketing.

## Installation

Instructions on setting up the project environment.

```bash
git clone https://github.com/nibekasov/qualifying_assignments/edit/main/EDA/RaiffeisenBank/Geo.git
cd SAM-Analysis
pip install -r requirements.txt

```bash
project_name/
│
├── data/
│   ├── raw/                  # Raw data, immutable
│   └── processed/            # Cleaned and processed data
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_data_analysis.ipynb
│   └── 03_results_visualization.ipynb
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
├── requirements.txt          # Project dependencies
└── README.md                 # Project description and instructions

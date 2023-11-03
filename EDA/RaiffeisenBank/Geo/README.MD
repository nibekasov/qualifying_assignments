Привет!

Вам предстоит выполнить задание, которое находится в Task.ipynb. Такие файлы открываются с помощью Jupyter Notebook + Python. Если он у вас не установлен, то рекомендую обратить внимание на дистрибутив Ananconda. Удачи!

предполагаемая структура проекта

'''
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
''';

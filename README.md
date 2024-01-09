# Recommendation system

Yeslamov Temirlan, 22200836

Aldanbergen Zholdas, 22211514

Recommendation system - Diabetes

# Project description
This myGit project is recommendation system with the theme of diabetes and is the part of course "Assistance Systems". The dataset that is used is in csv format. The table contains different parameters and the outcome. Different parameters affect the outcome which is Diabetic or Non-Diabetic. The dataset is used to train the models to predict the outcome with new data. In jupyter notebook, the data analysis, outlier analysis, missing value analysis are performed and the results accuracy of models are recorded. The model is used in GUI application to process the user inputs and output the outcome, graph, metrics.

# Installation
To install this project the repository needs to be cloned. Virtual environment can be created by running following commands:

```bash
python -m venv [name of venv]
```

On Windows, run: 
```bash
[name_of_the_venv]\Scripts\activate. 
```
On macOS and Linux, run: 
```bash
source [name_of_the_venv]/bin/activate.
```

you can deactivate it by running the command:
```bash
deactivate
```
To install the necessary packages, run following command:
```bash
pip install -r requirements.txt
```
Also you can see jupyter notebook named "recommendation_system.ipynb" which contains all data analysis. Running jupyter notebook might take a lot of time

# Basic usage
Run diabetes.py using command line python diabetes.py

The GUI window will be opened. User should input the parameters using sliders and dial, after push the button to see the results. The results are shown via bar chart and outputs such as recommendation, prediction, most influential factor. 

The visualisation part, the bar chart shows the results of SHAP values, which show the contribution of each parameter to the prediction. The most influential factor shows the parameter that contributed the most to the outcome where positive means the diabetes case. Recommendation is based on the most influential factor, so the person can understand which factor mainly causes the diabetes

Data analysis can be found in jupyter notebook

# Work done
Zholdas performed tasks:

1) Graphical User Interface
2) Pandas with Numpy

Temirlan performed tasks:

3) Visualization
4) Scikit-Learn

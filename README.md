# Recommendation system

Yeslamov Temirlan, 22200836

Aldanbergen Zholdas, 22211514

Recommendation system - Diabetes

# Project description
This project, a part of the "Assistance Systems" course, focuses on developing a sophisticated diabetes recommendation system. Utilizing a comprehensive dataset in CSV format, the system analyzes various health parameters to predict diabetes outcomes. These parameters include Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age, and Pregnancies, which play a crucial role in determining whether an individual is Diabetic or Non-Diabetic.

The core of the project lies in its meticulous data processing and analysis performed using Jupyter Notebook. This includes:

**Data Analysis**: A thorough examination of the dataset to understand the distribution and relationships of different variables.
**Outlier Analysis**: Identification and handling of anomalies in the dataset to ensure model accuracy and reliability.
**Missing Value Analysis**: Addressing missing data by implementing robust strategies like median imputation, enhancing the dataset's integrity.
Post data processing, a series of machine learning models are trained and evaluated to select the best performer. The models include Logistic Regression, K-Nearest Neighbors, Decision Trees, Random Forest, Support Vector Machine, and Gradient Boosting, each offering unique strengths in prediction. Their performance is meticulously recorded, with a keen focus on accuracy, precision, recall, and ROC-AUC metrics.

The chosen model is then integrated into a Graphical User Interface (GUI) application, which stands as the user-facing component of this system. The GUI is designed to be intuitive and user-friendly, allowing users to input their health parameters and receive a predictive outcome. Alongside the prediction, the application also provides insightful graphs and metric evaluations, offering users a comprehensive view of their health assessment.

This project not only demonstrates a practical application of machine learning in healthcare but also provides a valuable tool for early diabetes detection and awareness. By leveraging data-driven insights, it aims to assist individuals in understanding their diabetes risk and encourages proactive health management.

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

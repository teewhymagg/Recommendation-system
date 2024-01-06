import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QWidget, QSlider, QDial, QLineEdit, QGroupBox, QMenuBar, QStatusBar
from PyQt6.QtCore import QRect, QSize, QMetaObject
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
import joblib
from joblib import dump
from joblib import load
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np


class Ui_MainWindow:

    def __init__(self):
        try:
            # Load the trained RandomForest model and RobustScaler
            self.rf_model = load('E:\\AIN-B-3\\assistance systems\\test1 recommend\\random_forest_model.joblib')
            self.scaler = load('E:\\AIN-B-3\\assistance systems\\test1 recommend\\robust_scaler.joblib')
        except Exception as e:
            print(f"Error loading model or scaler: {e}")

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(978, 804)

        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.createGroupBox()
        self.createWidgets()
        self.setupLayouts()

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.statusbar = QStatusBar(MainWindow)

        MainWindow.setMenuBar(self.menubar)
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)

    def createGroupBox(self):
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QRect(440, 400, 531, 351))
        self.groupBox.setObjectName("groupBox")

    def createWidgets(self):
        # Dial for BMI
        self.createDial("bmiDial", QRect(600, 70, 161, 161))

        # Sliders
        self.createSlider("AgeSlider", QRect(40, 50, 291, 31), 0, 100)
        self.createSlider("diabetesPedigreeFunctionSlider", QRect(40, 120, 291, 31), 0, 250)  # Multiplied by 100 for precision
        self.createSlider("glucoseSlider", QRect(40, 190, 291, 31), 40, 200)
        self.createSlider("bloodPressureSlider", QRect(40, 260, 291, 31), 24, 125)
        self.createSlider("skinThicknessSlider", QRect(40, 330, 291, 31), 0, 70)
        self.createSlider("insulinSlider", QRect(40, 400, 291, 31), 10, 600)
        
        # QLineEdit for Pregnancies
        self.lineEdit = QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QRect(40, 470, 201, 41))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setPlaceholderText("Enter number of pregnancies")  # Set placeholder text

        # Push Button
        self.pushButton = QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QRect(50, 540, 191, 41))
        self.pushButton.setObjectName("pushButton")

        # Labels
        self.createLabel("ageLabel", "Age", QRect(40, 20, 291, 16))
        self.createLabel("diabetesPedigreeFunctionLabel", "Diabetes Pedigree Function", QRect(40, 90, 291, 16))
        self.createLabel("glucoseLabel", "Glucose", QRect(40, 160, 291, 16))
        self.createLabel("bloodPressureLabel", "Blood Pressure", QRect(40, 230, 291, 16))
        self.createLabel("skinThicknessLabel", "Skin Thickness", QRect(40, 300, 291, 16))
        self.createLabel("insulinLabel", "Insulin", QRect(40, 370, 291, 16))
        self.createLabel("pregnanciesLabel", "Pregnancies", QRect(40, 440, 291, 16))
        self.createLabel("bmiLabel", "BMI", QRect(600, 25, 291, 21))

            # Creating Sliders with value labels
        self.createSliderWithValueLabel("ageSlider", QRect(40, 50, 291, 31), 0, 100, QRect(340, 50, 50, 31), "")
        self.createSliderWithValueLabel("diabetesPedigreeFunctionSlider", QRect(40, 120, 291, 31), 0, 250, QRect(340, 120, 50, 31), "", scale=100)
        self.createSliderWithValueLabel("glucoseSlider", QRect(40, 190, 291, 31), 40, 200, QRect(340, 190, 50, 31), "")
        self.createSliderWithValueLabel("bloodPressureSlider", QRect(40, 260, 291, 31), 24, 125, QRect(340, 260, 50, 31), "")
        self.createSliderWithValueLabel("skinThicknessSlider", QRect(40, 330, 291, 31), 0, 70, QRect(340, 330, 50, 31), "")
        self.createSliderWithValueLabel("insulinSlider", QRect(40, 400, 291, 31), 0, 300, QRect(340, 400, 50, 31), "")
        self.createDialWithValueLabel("bmiDial", QRect(600, 70, 161, 161), 18, 70, QRect(760, 70, 50, 31))
        
        # QLineEdit for Pregnancies
        self.lineEdit = QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QRect(40, 470, 201, 41))
        self.lineEdit.setObjectName("lineEdit")

        # QPushButton for Prediction
        self.pushButton = QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QRect(50, 540, 191, 41))
        self.pushButton.setObjectName("pushButton")

        self.prediction_label = QLabel(self.centralwidget)
        self.prediction_label.setGeometry(QRect(50, 600, 300, 50))  # Adjust the geometry as needed
        self.prediction_label.setObjectName("prediction_label")
        self.prediction_label.setText("Prediction will be shown here")

    # Connect the predict button to the makePrediction method
        self.pushButton.clicked.connect(self.makePrediction)

    def createDial(self, name, geometry):
        setattr(self, name, QDial(self.centralwidget))
        dial = getattr(self, name)
        dial.setGeometry(geometry)
        dial.setObjectName(name)

    def createSlider(self, name, geometry, min_val, max_val):
        setattr(self, name, QSlider(self.centralwidget))
        slider = getattr(self, name)
        slider.setGeometry(geometry)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setOrientation(Qt.Orientation.Horizontal)
        slider.setObjectName(name)

    def createLabel(self, name, text, geometry):
        setattr(self, name, QLabel(self.centralwidget))
        label = getattr(self, name)
        label.setGeometry(geometry)
        label.setObjectName(name)
        label.setText(QGuiApplication.translate("MainWindow", text, None))

    def createSliderWithValueLabel(self, name, slider_geometry, min_val, max_val, label_geometry, label_prefix, scale=1):
        slider = QSlider(self.centralwidget)
        slider.setGeometry(slider_geometry)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setOrientation(Qt.Orientation.Horizontal)
        slider.setObjectName(name)
        setattr(self, name, slider)

        label = QLabel(self.centralwidget)
        label.setGeometry(label_geometry)
        label.setObjectName(name + "Label")
        label.setText(label_prefix + str(slider.value() / scale))
        setattr(self, name + "Label", label)

        slider.valueChanged.connect(lambda value: self.updateLabel(name, value, label_prefix, scale))

    def updateLabel(self, slider_name, value, label_prefix, scale):
        label = getattr(self, slider_name + "Label")
        label.setText(label_prefix + str(value / scale))

    def createDialWithValueLabel(self, name, geometry, min_val, max_val, label_geometry):
        dial = QDial(self.centralwidget)
        dial.setGeometry(geometry)
        dial.setMinimum(min_val)
        dial.setMaximum(max_val)
        dial.setObjectName(name)
        setattr(self, name, dial)

        label = QLabel(self.centralwidget)
        label.setGeometry(label_geometry)
        label.setObjectName(name + "Label")
        label.setText(str(dial.value()))
        setattr(self, name + "Label", label)

        dial.valueChanged.connect(lambda value: self.updateDialLabel(name, value))

    def updateDialLabel(self, dial_name, value):
        label = getattr(self, dial_name + "Label")
        label.setText(str(value))

    def setupLayouts(self):
        # Define layout setup here if needed
        pass

    def retranslateUi(self, MainWindow):
        _translate = QGuiApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Predict"))
        
    def makePrediction(self):
        try:
            # Collect values from sliders and dial
            values = [
                int(self.lineEdit.text()) if self.lineEdit.text() else 0,  # Pregnancies, default to 0 if empty
                self.glucoseSlider.value(),
                self.bloodPressureSlider.value(),
                self.skinThicknessSlider.value(),
                self.insulinSlider.value(),
                self.bmiDial.value(),
                self.diabetesPedigreeFunctionSlider.value() / 100,  # Scale back the value
                self.ageSlider.value()
            ]
            print("Collected values:", values)  # Debug print

            # Convert values to a numpy array and reshape for a single prediction
            values_array = np.array(values).reshape(1, -1)
            print("Values array:", values_array)  # Debug print

            # Scale the values
            scaled_values = self.scaler.transform(values_array)
            print("Scaled values:", scaled_values)  # Debug print

            # Make a prediction
            prediction = self.rf_model.predict(scaled_values)
            print("Prediction:", prediction)  # Debug print

            # Display the prediction
            self.prediction_label.setText(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'}")
        except Exception as e:
            print(f"Error in making prediction: {e}")
            self.prediction_label.setText("Error in making prediction")


    def displayFeatureImportance(self):
        # Get feature importances from the model
        importances = self.rf_model.feature_importances_
        # Identify the most important feature
        most_important_feature = X.columns[np.argmax(importances)]
        return most_important_feature
    
# Load the dataset
file_path = 'diabetes.csv'
data = pd.read_csv(file_path)

# Separating the features and the target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply RobustScaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Displaying the first few rows of the scaled dataset
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_train_scaled_df.head()

# Train the RandomForest model
rf_model = RandomForestClassifier(random_state=12345)
rf_model.fit(X_train_scaled, y_train)

# Making predictions on the test set
y_pred = rf_model.predict(X_test_scaled)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)


# Assuming rf_model and scaler are your trained RandomForest model and RobustScaler
dump(rf_model, 'random_forest_model.joblib')
dump(scaler, 'robust_scaler.joblib')

# Identifying the most important feature
feature_importances = rf_model.feature_importances_
most_important_feature = X.columns[np.argmax(feature_importances)]
most_important_feature, feature_importances

if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
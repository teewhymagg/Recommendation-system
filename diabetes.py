import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QWidget, QSlider, QDial, QLineEdit, QGroupBox, QMenuBar, QStatusBar
from PyQt6.QtCore import QRect, QSize, QMetaObject
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from model import load_data, train_model, predict
import joblib

class Ui_MainWindow:
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
        # ... (Set text for other widgets similarly)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        # Load and train your model
        self.X, self.y = load_data('your_preprocessed_data.csv')
        self.model = train_model(self.X, self.y)
        # Load the scaler
        self.scaler = joblib.load('scaler.save')
        # Connect the Predict button
        self.pushButton.clicked.connect(self.make_prediction)
    
    def make_prediction(self):
        # Collect input data from the GUI
        input_data = [
            int(self.lineEdit.text()) if self.lineEdit.text() else 0,  # Pregnancies
            self.glucoseSlider.value(),  # Glucose
            self.bloodPressureSlider.value(),  # BloodPressure
            self.skinThicknessSlider.value(),  # SkinThickness
            self.insulinSlider.value(),  # Insulin
            self.bmiDial.value(),  # BMI
            self.diabetesPedigreeFunctionSlider.value() / 100.0,  # DiabetesPedigreeFunction
            self.ageSlider.value()  # Age
        ]

        scaled_input = self.scaler.transform([input_data])
        # Make a prediction
        prediction = predict(self.model, scaled_input[0])
        # Update the GUI with the prediction result
        self.prediction_label.setText(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'}")

# Usage example
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec())

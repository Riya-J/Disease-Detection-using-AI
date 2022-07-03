
from email import message
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  RandomizedSearchCV, train_test_split

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score,precision_score, recall_score, confusion_matrix ,ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')



from flask import Flask, render_template, request, url_for
import numpy as np
import pickle



diabetes_model = pickle.load(open('diabetes_model.pkl', 'rb'))

    
heart_model = pickle.load(open('heart_model.pkl', 'rb'))

lung_model = pickle.load(open('lung_model.pkl', 'rb'))

breast_model = pickle.load(open('breast_model.pkl', 'rb'))

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route("/")
def basic():
    return render_template("landing.html")

# DIABETES
@app.route("/diabetes", methods=['GET', 'POST'])
def diabetes():
    print("hello")
    return render_template("diabetes.html")

@app.route("/diabetes_predict", methods=['POST'])
def diab_predict():

    
    label = diabetes_model.predict([[request.form["preg"],request.form["glc"],request.form["bp"],
                            request.form["st"],request.form["ins"],request.form["bmi"],
                            request.form["dp"],request.form["age"]]]) 
    print(label[0], type)
    class1 = "You Have Diabetes" if label[0]==1 else "You Don't Have Diabetes"
    return render_template("resultpg.html", message=class1)

# HEART DISEASE
@app.route("/heart", methods=['GET', 'POST'])
def heart():
    print("hello")
    return render_template("heart.html")

@app.route("/heart_predict", methods=['POST'])
def heart_predict():

    
    label1 = heart_model.predict([[request.form["age"],request.form["sex"],request.form["cp"],
                            request.form["tb"],request.form["chol"],request.form["fbs"],request.form["recg"],
                            request.form["th"],request.form["ex"],request.form["old"],
                            request.form["slp"],request.form["ca"],request.form["thal"]]]) 
    print(label1[0])
    class2 = "You Have Heart Disease" if label1[0]==1 else "You Don't Have Heart Disease"
    return render_template("resultpg.html", message=class2)


# LUNG CANCER 

@app.route("/lung", methods=['GET', 'POST'])
def lung():
    print("hello")
    return render_template("lung.html")

@app.route("/lung_predict", methods=['POST'])
def lung_predict():

    
    label2 = lung_model.predict([[request.form["yf"],request.form["anx"],request.form["fati"],
                            request.form["wheez"],request.form["cou"],request.form["sb"],request.form["sd"],
                            request.form["cps"]]]) 
    print(label2[0])
    class3 = "You Have Lung Cancer" if label2[0]==1 else "You Don't Have Lung Cancer"
    return render_template("resultpg.html", message=class3)

# BREAST CANCER

@app.route("/breast", methods=['GET', 'POST'])
def breast():
    print("hello")
    return render_template("breastc.html")

@app.route("/breast_predict", methods=['POST'])
def breast_predict():

    
    label3 = breast_model.predict([[request.form["rad"],request.form["text"],request.form["peri"],
                            request.form["area"],request.form["smooth"],request.form["comp"],request.form["conc"],
                            request.form["cp"],request.form["sym"]]]) 
    print(label3[0])
    class4 = "You Have Breast Cancer" if label3[0]==1 else "You Don't Have Breast Cancer"
    return render_template("resultpg.html", message=class4)


if __name__ == "__main__":
    app.run(port = 3000)
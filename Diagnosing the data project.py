import numpy as np  
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt  
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, GridSearchCV  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.preprocessing import LabelEncoder, RobustScaler  
from sklearn.metrics import classification_report, confusion_matrix  
from numpy import mean,std

def load_data(filepath):  
 
    try:  
        df = pd.read_csv(filepath)  
        return df  
    except Exception as e:  
        print(f"Error loading data: {e}")  
        return None  

def preprocess_data(df):   
    X = df.iloc[:, :-1].values.astype('float32')  # Feature matrix  
    y = LabelEncoder().fit_transform(df.iloc[:, -1].astype('str'))  # Target variable  
    return X, y  

def run_knn(X, y):    
    cv = RepeatedStratifiedKFold(n_repeats=3, n_splits=10, random_state=5382)  
    model = KNeighborsClassifier()  
  
    param_grid = {'n_neighbors': np.arange(1, 21)}  
    grid = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)  
    grid.fit(X, y)  

    print(f'Best number of neighbors: {grid.best_params_["n_neighbors"]}')  

    n_score = cross_val_score(grid.best_estimator_, X, y, scoring='accuracy', cv=cv, n_jobs=-1)  
    print('Accuracy: %.2f (%.2f)' % (mean(n_score) * 100, std(n_score)))  

    return grid.best_estimator_  


def plot_confusion_matrix(y_true, y_pred):  
    confusion = confusion_matrix(y_true, y_pred)  
    plt.figure(figsize=(8, 6))  
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',  
                xticklabels=['Non-Diabetic', 'Diabetic'],  
                yticklabels=['Non-Diabetic', 'Diabetic'])  
    plt.ylabel('Actual')  
    plt.xlabel('Predicted')  
    plt.title('Confusion Matrix')  
    plt.show()  

if __name__ == "__main__":  
    df_diabetes = load_data(r"diabetes.csv")  
    if df_diabetes is not None:  
        X_diabetes, y_diabetes = preprocess_data(df_diabetes)  

        best_model_diabetes = run_knn(X_diabetes, y_diabetes)  
        y_pred_diabetes = best_model_diabetes.predict(X_diabetes)  
        
        print("Diabetes Classification Report:")  
        print(classification_report(y_diabetes, y_pred_diabetes))  
        plot_confusion_matrix(y_diabetes, y_pred_diabetes)  
 
    df_oil_spill = load_data(r"oil-spill.csv")  
    if df_oil_spill is not None:  
        X_oil_spill, y_oil_spill = preprocess_data(df_oil_spill)  


        best_model_oil_spill = run_knn(X_oil_spill, y_oil_spill)  
        y_pred_oil_spill = best_model_oil_spill.predict(X_oil_spill)  

        print("Oil Spill Classification Report:")  
        print(classification_report(y_oil_spill, y_pred_oil_spill))  
        plot_confusion_matrix(y_oil_spill, y_pred_oil_spill)


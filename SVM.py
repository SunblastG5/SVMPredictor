import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import streamlit as st

class SVM:

    @st.cache
    def ObtenerDatos():
        cancer = pd.read_csv("breastcancer.csv")
        cancer = pd.DataFrame({
            'texture_mean': cancer['texture_mean'],
            'area_mean': cancer['area_mean'],
            'diagnosis': cancer['diagnosis']
        })
        cancer['diagnosis'] = np.where(cancer['diagnosis']=='M', 0, 1)


        df_target = cancer["diagnosis"]
        df_feat = cancer[["texture_mean", "area_mean"]]
        return df_target, df_feat
    
    def EntrenarModelo(self):
        df_target, df_feat = SVM.ObtenerDatos()
        X_train, X_test, y_train, y_test = train_test_split(df_feat, np.ravel(df_target), test_size=0.30, random_state=101)
        self.model = SVC()
        self.model.fit(X_train,y_train)

    def Prediccion(self,t_mean, a_mean):
        d = {'texture_mean': [t_mean], 'area_mean': [a_mean]}
        df_predict = pd.DataFrame(data=d, dtype=np.float64)
        prediction = self.model.predict(df_predict)
        return prediction


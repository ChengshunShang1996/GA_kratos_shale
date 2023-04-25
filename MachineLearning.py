import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler, Normalizer, LabelEncoder, minmax_scale
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, KFold
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')

#importing all the required ML packages
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix

class MachineLearning():

    def __init__(self) -> None:

        #read data file
        #data_file = 'G_info.csv' 
        file_name = 'G_info.csv' 
        data_file = os.path.join(os.getcwd(),'kratos_results_data', file_name)
        self.df = pd.read_csv(data_file)
        print(self.df.info())
        #self.r_squared_list = []

    def data_processing(self, data_min_list, data_max_list, predict_index):
        
        #data processing
        self.df = self.df[self.df.strength_max != 0]   #delete 0.0 data rows
        self.df = self.df.drop_duplicates()            #delete repeated rows

        # drop outliers
        outlier_list = self.Zscore_outlier(self.df["strength_max"])
        for num in outlier_list:
            i = self.df[(self.df.strength_max == num)].index
            self.df = self.df.drop(i)

        #outlier_list = self.Zscore_outlier(self.df["strain_max"])
        #for num in outlier_list:
        #    i = self.df[(self.df.strain_max == num)].index
        #    self.df = self.df.drop(i)

        outlier_list = self.Zscore_outlier(self.df["young_modulus_max"])
        for num in outlier_list:
            i = self.df[(self.df.young_modulus_max == num)].index
            self.df = self.df.drop(i)

        print(self.df.info())

        self.data_array = self.df.values
        
        X = Y = []
        X, Y = self.select_prediction_parameter(predict_index, data_min_list, data_max_list)

        test_number = 25

        ##training data
        X_train = X[:-test_number]
        Y_train = Y[:-test_number]

        #testing data
        X_test = X[-test_number:]
        Y_test = Y[-test_number:]

        return X_train, Y_train, X_test, Y_test

    def xgboost_model_training(self, X_train, Y_train, X_test, Y_test):
        #xgb = XGBRegressor(booster='gbtree', max_depth=40, learning_rate=0.2, reg_alpha=0.01, n_estimators=2000, gamma=0.1, min_child_weight=1)
        xgb = XGBRegressor(booster='gbtree')
        xgb.fit(X_train,Y_train)

        pre = xgb.predict(X_test)

        #5指标重要性可视化
        #importance = xgb.feature_importances_
        #plt.figure(1)
        #plt.barh(y = range(importance.shape[0]),  #指定条形图y轴的刻度值
        #        width = importance,  #指定条形图x轴的数值
        #        tick_label =range(importance.shape[0]),  #指定条形图y轴的刻度标签
        #        color = 'orangered',  #指定条形图的填充色
        #        )
        #plt.title('Feature importances of XGBoost')

        #6计算评价指标
        #print(' MAE : ', mae(Y_test, pre))
        #print(' MSE : ',mse(Y_test, pre))
        #print(' RMSE : ', np.sqrt(mse(Y_test, pre)))
        print(' R_squared : ',metrics.r2_score(Y_test, pre))

        #self.plot_pre_and_test_results(pre,Y_test,"XGBoost results")

        return xgb

    def Zscore_outlier(self, df_part):
        out=[]
        m = np.mean(df_part)
        sd = np.std(df_part)
        for i in df_part: 
            z = (i-m)/sd
            if np.abs(z) > 3: 
                out.append(i)
        print("Outliers:",out)
        return out
    
    def my_normalizer(self, df, data_min_list, data_max_list):
        for data in df:
            for i in range(len(data)):
                a = data[i] - data_min_list[i]
                b = data_max_list[i] - data_min_list[i]
                c = a/b
                data[i] = c
                #data[i] = (data[i] - data_min_list[i]) / (data_max_list[i] - data_min_list[i])
        return df
    
    def select_prediction_parameter(self, parameter, data_min_list, data_max_list):
        X = self.data_array[:, :14]
        X = self.my_normalizer(X, data_min_list, data_max_list)
        # X_train = data_array[:, :5] 
        # X_train = data_array[:, :6]
        Y = self.data_array[:, parameter] 
        return X, Y
    
    def plot_pre_and_test_results(self, pre, Y_test, my_title):
        fig = plt.figure(2)
        ax = fig.add_subplot(111)
        list_x = list(range(1,len(pre)+1))
        lns1 = ax.plot(list_x, pre, 's' , color='red', label='predict')
        lns2 = ax.plot(list_x, Y_test, 'o' , color='blue', label='true')
        ax.set_xlabel("Cases Number")
        ax.set_ylabel("Strength (Pa)")
        #ax.set_ylabel("Max strain (%)")
        #ax.set_ylabel("Young modulus (Pa)")
        abs_error = abs(Y_test - pre)
        ax2 = ax.twinx()
        lns3 = ax2.bar(list_x, abs_error, color = "lightgray", width = 0.7, label='AE')
        ax2.set_ylabel("Absolute error")
        ax2.set_ylim(0, 5e8)
        ax.set_ylim(-0.9e8, 4e8)
        plt.title(my_title)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2, loc=2)
        #ax.legend()
        #ax2.legend()
        plt.show()
        #self.r_squared_list.append(metrics.r2_score(Y_test, pre))
    
    def ML_main(self, data_min_list, data_max_list, predict_index):

        X_train, Y_train, X_test, Y_test = self.data_processing(data_min_list, data_max_list, predict_index)
        xgb = self.xgboost_model_training(X_train, Y_train, X_test, Y_test)

        return xgb

    
if __name__ == "__main__":
    
    data_min_list = [0, 0, 5e8, 5e8, 1, 5e8, 5e8, 1, 2e6, 2e6, 0, 2e6, 2e6, 0]
    data_max_list = [15e6, 90, 1e11, 1e11, 3, 1e11, 1e11, 3, 2e8, 2e8, 50, 2e8, 2e8, 50]
    
    predict_index = 14

    run = MachineLearning()
    xgb = run.ML_main(data_min_list, data_max_list, predict_index)

    d = [0, 0, 6e8, 6e8, 2, 6e8, 6e8, 2, 2e7, 2e7, 10, 2e7, 2e7, 10]
    X_test = []
    X_test.append(d)
    X_test = run.my_normalizer(X_test, data_min_list, data_max_list)
    pre = xgb.predict(X_test)
    print(pre)


    




import numpy as np
import pandas as pd
from numpy import exp, sqrt, log, sin, cos
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class ZoomObfuscator(object):
    def __init__(self, scalerfile, factor, number, outputfile):
        self.scalerfile = scalerfile
        self.factor = factor
        self.number = number
        self.outputfile = outputfile
    
    def _csv2df(self):
        df = pd.read_csv(self.scalerfile).loc[:, "f1s":"DFTtarget"]
        return df
       
    def _apply(self):
        train_data = self._csv2df()
        sample = train_data.sample(n=self.number, random_state=42)
        sample.iloc[:, -1] = sample.iloc[:, -1].apply(lambda x: x*self.factor)
        remain = train_data.drop(index=sample.index)
        zoomobfuscator = pd.concat([sample, remain], axis=0)
        return zoomobfuscator
    
    def output(self):
        zoomobfuscator = self._apply()
        zoomobfuscator.to_csv(self.outputfile, index=True)
        return


class ShuffleObfuscator(object):
    def __init__(self, scalerfile, target, number, outputfile):
        self.scalerfile = scalerfile
        self.target = target
        self.number = number
        self.outputfile = outputfile
        
    def _csv2df(self):
        df = pd.read_csv(self.scalerfile).loc[:, "f1s":"DFTtarget"]
        return df    
    
    def _apply(self):
        train_data = self._csv2df()
        sample = train_data.sample(n=self.number, random_state=42)
        remain = train_data.drop(index=sample.index).reset_index()
        shuffle = sample.sample(frac=1, random_state=42).reset_index()[self.target]
        sample = sample.reset_index()
        sample[self.target] = shuffle
        shuffleobfuscator = pd.concat([sample, remain], ignore_index=True).set_index("index")
        return shuffleobfuscator
    
    def output(self):
        shuffleobfuscator = self._apply()
        shuffleobfuscator.to_csv(self.outputfile, index=True)
        return 


class DataProvider(object):
    def __init__(self, inputfile, feature_start, feature_end, target, scalerfile):
        self.inputfile = inputfile
        self.feature_start = feature_start
        self.feature_end = feature_end
        self.target = target
        self.scalerfile = scalerfile
        self.P = None
    
    def feature_standardization(self):
        data = pd.read_csv(self.inputfile).loc[:, self.feature_start:self.target]
        data_X = data.loc[:, self.feature_start:self.feature_end].values
        data_y = data.loc[:, self.target].values
        train_feature, test_feature = train_test_split(data_X, test_size=0.2, random_state=42)
        train_target, test_target = train_test_split(data_y, test_size=0.2, random_state=42)
        scaler = StandardScaler().fit(train_feature)
        train_feature_scaler = scaler.transform(train_feature)
        test_feature_scaler = scaler.transform(test_feature)
        train_feature_scaler_df, test_feature_scaler_df, train_target_df, test_target_df = [pd.DataFrame(i) for i in [train_feature_scaler, test_feature_scaler, train_target, test_target]]
        trainset_df = pd.concat([train_feature_scaler_df, train_target_df], axis=1)
        testset_df = pd.concat([test_feature_scaler_df, test_target_df], axis=1)
        dataset_df = pd.concat([trainset_df, testset_df])
        headname = list(["{}{}s".format(x, y) for x in ["f", "I", "R"] for y in range(1, 7)])
        headname.append("DFTtarget")
        dataset_df.to_csv(self.scalerfile, index=True, header=headname)
    
    def trainset_random_selection(self, selected_file, remained_file, n=None, frac=None):
        df = pd.read_csv(self.scalerfile)
        if n is not None and n > 0:
            subset = df.sample(n, axis=0, random_state=42)
        elif type(frac) == float and 0 < frac <= 1 :
            subset = df.sample(frac=frac, axis=0, random_state=42)
        else:
            raise AttributeError("n and frac should not be None at same time")
        remain = df.drop(labels=subset.index)
        subset.to_csv(selected_file, index=False)
        remain.to_csv(remained_file, index=False)
        
    def provide_variable(self):
        data = pd.read_csv(self.scalerfile).loc[:, "f1s":"DFTtarget"]
        data_X = data.loc[:, "f1s":"R6s"].values
        data_y = data.loc[:, "DFTtarget"].values
        self.P = data_y
        return data_X.T


class SISSOLearning(object):
    def __init__(self, formula, dataprovider:DataProvider):
        self.formula = formula
        self.dataprovider = dataprovider
        dataprovider.provide_variable()
        self.P = dataprovider.P
        self.P_pred = None
    
    def calc_params(self):
        f1s, f2s, f3s, f4s, f5s, f6s, I1s, I2s, I3s, I4s, I5s, I6s, R1s, R2s, R3s, R4s, R5s, R6s = self.dataprovider.provide_variable()
        reg = LinearRegression()
        reg.fit(np.array([eval(self.formula)]).T, self.P)
        coefficient = reg.coef_[0]
        intercept = reg.intercept_
        self.P_pred = reg.predict(np.array([eval(self.formula)]).T).flatten()
        return coefficient, intercept
    
    def linear_fitting(self, coef, intercept):
        f1s, f2s, f3s, f4s, f5s, f6s, I1s, I2s, I3s, I4s, I5s, I6s, R1s, R2s, R3s, R4s, R5s, R6s = self.dataprovider.provide_variable()
        self.P_pred = np.array([coef * eval(self.formula) + intercept]).T.flatten()
        return
    
    def predictive_index(self):
        r = format(np.corrcoef(self.P, self.P_pred)[0][1], '.3f')
        rmse = format(np.sqrt(mean_squared_error(self.P, self.P_pred)), '.3f')
        return r, rmse
        

class Drawing(object):
    def __init__(self, sissolearning:SISSOLearning):
        self.P = sissolearning.P
        self.P_pred = sissolearning.P_pred
        self.r, self.rmse = sissolearning.predictive_index()
    
    def compared_size(self):
        scatter_x = self.P
        target_true_length = self.P.max() - self.P.min()
        target_pred_length = self.P_pred.max() - self.P_pred.min()
        if target_true_length > target_pred_length:
            scatter_y = self.P
        else:
            scatter_y = self.P_pred
        return scatter_x, scatter_y
    
    def text_position(self):
        x, y = self.compared_size()
        x_length = x.max() - x.min()
        y_length = y.max() - y.min()
        text1_x_position = 1/50 * x_length + x.min()
        text2_x_position = 1/50 * x_length + x.min()
        text1_y_position = y.max() - 1/20 * y_length
        text2_y_position = y.max() - 2/15 * y_length
        text3_x_position = x.min() + 34/50 * x_length
        text3_y_position = y.min() + 1/40 * y_length
        return text1_x_position, text1_y_position, text2_x_position, text2_y_position, text3_x_position, text3_y_position
    
    def plot_scatter(self):
        plt.figure(figsize=(4,4), dpi=300)
        plt.scatter(self.P, self.P, alpha=0.3, label="DFT")
        plt.scatter(self.P, self.P_pred, label="SISSO")
        plt.xlabel("DFT", fontsize=15)
        plt.ylabel("Formula Predicted", fontsize=15)
        plt.tick_params(axis='both', labelsize=13)
        r_x, r_y, rmse_x, rmse_y, legend_x, legend_y = self.text_position()
        plt.text(r_x, r_y, "r = " + self.r, fontsize=15)
        plt.text(rmse_x, rmse_y, "RMSE = " + self.rmse, fontsize=15)
        plt.text(legend_x, legend_y, r"$\Delta E_{ads}\,\left( eV\right)$", fontsize=15)
        plt.show()
        

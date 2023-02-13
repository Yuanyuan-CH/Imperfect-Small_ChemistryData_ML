import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras import layers, regularizers
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


class DataProvider(object):
    def __init__(self, inputfile, feature_start, feature_end, target):
        self.inputfile = inputfile
        self.feature_start = feature_start
        self.feature_end = feature_end
        self.target = target 

    def data_preprocessing(self):
        data = pd.read_csv(self.inputfile).loc[:, self.feature_start:self.target]
        data_X = data.loc[:, self.feature_start:self.feature_end].values
        data_y = data.loc[:, self.target].values
        train_feature, validation_test_feature = train_test_split(data_X, test_size=0.2, random_state=42)
        train_target, validation_test_target = train_test_split(data_y, test_size=0.2, random_state=42)
        validation_feature, test_feature = train_test_split(validation_test_feature, test_size=0.5, random_state=42)
        validation_target, test_target = train_test_split(validation_test_target, test_size=0.5, random_state=42)
        scaler = StandardScaler().fit(train_feature)
        train_feature_scaler = scaler.transform(train_feature)
        validation_feature_scaler = scaler.transform(validation_feature)
        test_feature_scaler = scaler.transform(test_feature)
        return train_feature, validation_feature, test_feature, train_feature_scaler, validation_feature_scaler, test_feature_scaler, train_target, validation_target, test_target


class HyperparameterConfig(object):
    def __init__(self):
        self.hiddenlayer_num = 2
        self.neural_units = [64, 64]
        self.l1_rate = [0, 0]
        self.l2_rate = [0, 0]
        self.dropout_rate = [0, 0]
        self.lr_rate = 1e-3
        self.model_save = "BestmodelSave.h5"
        self.epochs_num = 500
        self.batch_size_num = 128
        self.predictiondata_save = "NNPredictionDataSave.csv"
        self.pretrained_model = "PretrainedModel.h5"


class NNLearning(object):
    def __init__(self, dataprovider:DataProvider, hyperparameterconfig:HyperparameterConfig):
        self.train_feature, self.validation_feature, self.test_feature, self.train_feature_scaler, self.validation_feature_scaler, self.test_feature_scaler, self.train_target, self.validation_target, self.test_target = dataprovider.data_preprocessing()
        self.hyperparameter_config = hyperparameterconfig
        self.model = None
        self.callback_list = None
        
    def build_model(self):
        _nnmodel = keras.models.Sequential()
        _nnmodel.add(keras.Input(shape=(self.train_feature_scaler.shape[1], )))
        for i in range(self.hyperparameter_config.hiddenlayer_num):
            _nnmodel.add(
                layers.Dense(
                    self.hyperparameter_config.neural_units[i],
                    activation='relu',
                    activity_regularizer=regularizers.l1(self.hyperparameter_config.l1_rate[i]),
                    kernel_regularizer=regularizers.l2(self.hyperparameter_config.l2_rate[i]) 
                )
            )        
            _nnmodel.add(layers.Dropout(self.hyperparameter_config.dropout_rate[i]))
        _nnmodel.add(layers.Dense(1))
        self.model = _nnmodel
    
    def model_compile(self):
        self.model.compile(
            optimizer = keras.optimizers.Adam(learning_rate=self.hyperparameter_config.lr_rate),
            loss = "mse",
            metrics = ["mae"]
        )
    
    def callbacks_list(self):
        self.callback_list = [
            keras.callbacks.ModelCheckpoint(
                filepath=self.hyperparameter_config.model_save,
                monitor="val_loss",
                save_best_only=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=10
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=30
            )
        ]
    
    def model_training(self):
        self.model_history = self.model.fit(
            self.train_feature_scaler, self.train_target,
            validation_data = (self.validation_feature_scaler, self.validation_target),
            epochs = self.hyperparameter_config.epochs_num,
            batch_size = self.hyperparameter_config.batch_size_num,
            callbacks = self.callback_list,
            verbose = 1
        )
    
    def save_model(self):
        self.model.save(self.hyperparameter_config.model_save)
        
    def prediction(self):
        pred_train_target = self.model.predict(self.train_feature_scaler).flatten()
        pred_validation_target = self.model.predict(self.validation_feature_scaler).flatten()
        pred_test_target = self.model.predict(self.test_feature_scaler).flatten()
        return pred_train_target, pred_validation_target, pred_test_target
    
    def predictive_index(self):
        pred_train_target, pred_validation_target, pred_test_target = self.prediction()
        r_train = format(np.corrcoef(self.train_target, pred_train_target)[0][1], '.3f')
        rmse_train = format(np.sqrt(mean_squared_error(self.train_target, pred_train_target)), '.3f')
        r_validation = format(np.corrcoef(self.validation_target, pred_validation_target)[0][1], '.3f')
        rmse_validation = format(np.sqrt(mean_squared_error(self.validation_target, pred_validation_target)), '.3f')
        r_test = format(np.corrcoef(self.test_target, pred_test_target)[0][1], '.3f')
        rmse_test = format(np.sqrt(mean_squared_error(self.test_target, pred_test_target)), '.3f')
        return r_train, rmse_train, r_validation, rmse_validation, r_test, rmse_test
    
    def prediction_datasave(self):
        pred_train_target, pred_validation_target, pred_test_target = self.prediction()
        df_namelist = [self.train_feature, self.validation_feature, self.test_feature, self.train_feature_scaler, self.validation_feature_scaler, self.test_feature_scaler, self.train_target, self.validation_target, self.test_target, pred_train_target, pred_validation_target, pred_test_target]
        train_feature_df, validation_feature_df, test_feature_df, train_feature_scaler_df, validation_feature_scaler_df, test_feature_scaler_df,  train_target_df, validation_target_df, test_target_df, pred_train_target_df, pred_validation_target_df, pred_test_target_df = [pd.DataFrame(i) for i in df_namelist]
        trainset_df = pd.concat([train_feature_df, train_target_df, train_feature_scaler_df, pred_train_target_df], axis=1)
        validationset_df = pd.concat([validation_feature_df, validation_target_df, validation_feature_scaler_df, pred_validation_target_df], axis=1)
        testset_df = pd.concat([test_feature_df, test_target_df, test_feature_scaler_df, pred_test_target_df], axis=1)
        dataset_df = pd.concat([trainset_df, validationset_df, testset_df])
        headname = ["f1", "f2", "f3", "f4", "f5", "f6", "ir1", "ir2", "ir3", "ir4", "ir5", "ir6", "ra1", "ra2", "ra3", "ra4", "ra5", "ra6", "target", "f1_scaler", "f2_scaler", "f3_scaler", "f4_scaler", "f5_scaler", "f6_scaler", "ir1_scaler", "ir2_scaler", "ir3_scaler", "ir4_scaler", "ir5_scaler", "ir6_scaler", "ra1_scaler", "ra2_scaler", "ra3_scaler", "ra4_scaler", "ra5_scaler", "ra6_scaler", "NNpred_target"]
        dataset_df.to_csv(self.hyperparameter_config.predictiondata_save, index=True, header=headname)


class TransferLearning(NNLearning):
    def __init__(self, dataprovider:DataProvider, hyperparameterconfig:HyperparameterConfig):
        super().__init__(dataprovider, hyperparameterconfig)
    
    def build_model(self):
        pretrained_model = keras.models.load_model(self.hyperparameter_config.pretrained_model)
        _reusedmodel = keras.models.Sequential(pretrained_model.layers[:-1]) 
        _reusedmodel.add(keras.layers.Dense(1, name='output'))
        self.model = _reusedmodel
    
    def freeze_reused_layer(self):
        for layer in self.model.layers[:-1]:
            layer.trainable = False
    
    def unfreeze_reused_layer(self):
        for layer in self.model.layers[:-1]:
            layer.trainable = True


class Drawing(object):
    def __init__(self, dataprovider:DataProvider, nnlearning:NNLearning):
        self.train_feature, self.validation_feature, self.test_feature, self.train_feature_scaler, self.validation_feature_scaler, self.test_feature_scaler, self.train_target, self.validation_target, self.test_target = dataprovider.data_preprocessing()
        self.model_history = nnlearning.model_history
        self.pred_train_target, self.pred_validation_target, self.pred_test_target = nnlearning.prediction()
        self.r_train, self.rmse_train, self.r_validation, self.rmse_validation, self.r_test, self.rmse_test = nnlearning.predictive_index()
    
    def plot_array(self):
        plt.figure(figsize=(4,4), dpi=300)
        x = np.arange(len(self.model_history.history['loss']))
        x += 1
        plt.plot(x, self.model_history.history['loss'], label="loss")
        plt.plot(x, self.model_history.history['val_loss'], label="val_loss")
        plt.legend(fontsize=15)
        plt.xlabel("Epochs", fontsize=15) 
        plt.ylabel("Loss", fontsize=15)
        plt.tick_params(axis='both', labelsize=13)    
    
    def compared_size(self):
        scatter_x = self.test_target
        target_true_length = self.test_target.max() - self.test_target.min()
        target_pred_length = self.pred_test_target.max() - self.pred_test_target.min()
        if target_true_length > target_pred_length:
            scatter_y = self.test_target
        else:
            scatter_y = self.pred_test_target
        return scatter_x, scatter_y
    
    def text_position(self, x, y):
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
        plt.scatter(self.test_target, self.test_target, alpha=0.3, label="DFT")
        plt.scatter(self.test_target, self.pred_test_target, label="NN", color="#1fab89")
        plt.xlabel("DFT", fontsize=15)
        plt.ylabel("NN Predicted", fontsize=15)
        plt.tick_params(axis='both', labelsize=13)   
        scatter_x, scatter_y = self.compared_size()
        r_x, r_y, rmse_x, rmse_y, legend_x, legend_y = self.text_position(scatter_x, scatter_y)
        plt.text(r_x, r_y, "r = " + self.r_test, fontsize=15)
        plt.text(rmse_x, rmse_y, "RMSE = " + self.rmse_test, fontsize=15)
        plt.text(legend_x, legend_y, r"$\Delta E_{ads}\,\left( eV\right)$", fontsize=15)
        plt.show()
        
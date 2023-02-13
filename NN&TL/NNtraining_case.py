from NNframe import DataProvider, HyperparameterConfig, NNLearning, TransferLearning, Drawing

if __name__ == "__main__":
    # Provide Data
    provider = DataProvider("CO@CuBTC_3584.csv", "freq1", "raman6", "adsorp_energy")
    # Set Param
    configs = HyperparameterConfig()
    configs.hiddenlayer_num = 2
    configs.neural_units = [64, 64]
    configs.l1_rate = [0, 0]
    configs.l2_rate = [0, 0]
    configs.dropout_rate = [0, 0]
    configs.lr_rate = 1e-3
    configs.model_save = "NNlearning.h5"
    configs.epochs_num = 500
    configs.batch_size_num = 128
    configs.predictiondata_save = "NNPredictionDataSave.csv"
    # Build & Train Model
    nnlearning_process = NNLearning(provider, configs)
    nnlearning_process.build_model()
    nnlearning_process.model_compile()
    nnlearning_process.callbacks_list()
    nnlearning_process.model_training()
    picture_drawing = Drawing(provider, nnlearning_process)
    picture_drawing.plot_array()
    # Prediction
    nnlearning_process.prediction()
    index_list = ["r_train", "rmse_train","r_validation", "rmse_validation", "r_test", "rmse_test"]
    value_list = [i for i in nnlearning_process.predictive_index()]
    for index, value in zip(index_list, value_list):
        print(index + " = " + value)
    nnlearning_process.prediction_datasave()
    # Drawing
    picture_drawing.plot_scatter()
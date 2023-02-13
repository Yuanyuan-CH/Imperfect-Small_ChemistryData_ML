from NNframe import DataProvider, HyperparameterConfig, NNLearning, TransferLearning, Drawing

if __name__ == "__main__":
    # Provide Data
    provider = DataProvider("CO@PFC-73-Cu_1305.csv", "freq1", "raman6", "adsorp_energy")
    # Set Param
    configs = HyperparameterConfig()
    configs.pretrained_model = "CO@CuBTC_NNlearning.h5"
    configs.lr_rate = 1e-3
    configs.epochs_num = 500
    configs.batch_size_num = 16
    # Build & Train ReusedModel
    tl_process = TransferLearning(provider, configs)
    tl_process.build_model()
    tl_process.freeze_reused_layer()
    tl_process.model_compile()
    tl_process.model_training()
    picture_drawing = Drawing(provider, tl_process)
    picture_drawing.plot_array()
    tl_process.unfreeze_reused_layer()
    configs.lr_rate = 1e-5
    configs.epochs_num = 600
    configs.model_save = "TL2CO@PFC73.h5"
    configs.predictiondata_save = "TLPredictionDataSave.csv"
    tl_process.model_compile()
    tl_process.model_training()
    tl_process.save_model()
    picture_drawing = Drawing(provider, tl_process)
    picture_drawing.plot_array()
    # Prediction
    tl_process.prediction()
    tl_process.prediction_datasave()
    index_list = ["r_train", "rmse_train", "r_validation", "rmse_validation", "r_test", "rmse_test"]
    value_list = [i for i in tl_process.predictive_index()]
    for index, value in zip(index_list, value_list):
        print(index + " = " + value)
    # Drawing 
    picture_drawing.plot_scatter()
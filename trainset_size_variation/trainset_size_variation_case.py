from SISSOfit import DataProvider, SISSOLearning, Drawing

if __name__ == "__main__":
    # Provide Data
    dataprovider = DataProvider("CO@CuBTC_3584.csv", "freq1", "raman6", "adsorp_energy", "CO@CuBTC_scaler_3584.csv")
    dataprovider.feature_standardization()
    # Set Trainset Size
    dataprovider.trainset_random_selection("trainset.csv", "testset.csv", n=50)
    # Fit Params
    dataprovider.scalerfile = "trainset.csv"
    sissolearning = SISSOLearning("((cos(R4s)+sin(f3s))-sqrt(exp(f5s)))", dataprovider)
    coef, intercept = sissolearning.calc_params()
    r_trainset, rmse_trainset = sissolearning.predictive_index()
    print("coefficient = " + str(coef), "\nintercept = " + str(intercept))
    print("r_trainset = " + str(r_trainset), "\nrmse_trainset = " + str(rmse_trainset))
    # Prediction
    dataprovider.scalerfile = "testset.csv"
    sissolearning.linear_fitting(coef, intercept)
    sissolearning.P = dataprovider.P
    r_testset, rmse_testset = sissolearning.predictive_index()
    print("r_testset = " + str(r_testset), "\nrmse_testset = " + str(rmse_testset))
    # Drawing
    drawing = Drawing(sissolearning)
    drawing.plot_scatter()
    

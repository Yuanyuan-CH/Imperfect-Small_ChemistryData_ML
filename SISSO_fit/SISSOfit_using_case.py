from SISSOfit import DataProvider, SISSOLearning, Drawing

if __name__ == "__main__":
    # Provide Data
    dataprovider = DataProvider("CO@CuBTC_3584.csv", "freq1", "raman6", "adsorp_energy", "CO@CuBTC_scaler_3584.csv")
    dataprovider.feature_standardization()
    # Train Model & Prediction
    sissolearning = SISSOLearning("((cos(R4s)+sin(f3s))-sqrt(exp(f5s)))", dataprovider)
    coef, intercept = sissolearning.calc_params()
    sissolearning.linear_fitting(coef, intercept)
    r, rmse = sissolearning.predictive_index()
    print("coefficient = " + str(coef), "\nintercept = " + str(intercept))
    print("r = " + str(r), "\nrmse = " + str(rmse))
    # Drawing
    drawing = Drawing(sissolearning)
    drawing.plot_scatter()
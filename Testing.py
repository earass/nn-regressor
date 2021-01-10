from keras.models import load_model
import numpy as np

def predict(model,Xts):
    preds = model.predict(Xts)
    print(preds)
    return preds

def get_test_data():
    # load test data
    Xts = np.loadtxt("TestData.csv")
    print('loaded test data')
    return Xts

def execute_testing(model_path='myModel.h5',save_results=False):
    model = load_model(model_path)
    Xts = get_test_data()
    Yts = predict(model, Xts)
    if save_results:
        np.savetxt("i191254_Predictions.csv", Yts)
    return


if __name__ == '__main__':
    execute_testing(save_results=False)
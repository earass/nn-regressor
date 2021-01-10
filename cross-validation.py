from keras.wrappers.scikit_learn import KerasRegressor
from Training import get_training_data, baseline_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np

def RMSE(y_true,y_pred):
    """ creating RMSE using MSE to be used in evaluation """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print('RMSE: %2.3f' % rmse)
    return rmse

def scorer():
    """ scorer func to evaluate using RMSE """
    return make_scorer(RMSE, greater_is_better=False)

def get_cv_results(X,y,model,param_grid,cv=5):
    """ performs cv evaluation based on Grid Search CV"""
    grid = GridSearchCV(model, param_grid, refit=True, verbose=3, cv=cv,scoring=scorer())

    # fitting the model for grid search
    grid.fit(X, y)

    return grid

def get_best_hyperparameters(X,Y):
    """ estimates the best score and hypermaters for the model  """
    estimator = KerasRegressor(build_fn=baseline_model, verbose=0)

    # hyperparameters to choose best from
    params = {
        'lr' : [0.001,0.005, 0.01],
        'batch_size':[5,10],
        'epochs': [50,100]
    }

    grid = get_cv_results(X,Y,model=estimator,param_grid=params)

    # best score achieved with a set of params
    score = grid.best_score_

    # params against the best score
    best_params = grid.best_params_

    print('Best Score: %2.3f' % score)
    print('Best params', best_params)

    return score, best_params

def get_multiple_runs(X,Y,no_of_runs=5):
    """ to get mean and std on 5-runs of evaluation """
    all_scores = []
    all_best_params = []
    for i in range(no_of_runs):
        score, best_params = get_best_hyperparameters(X,Y)
        all_scores.append(score)
        all_best_params.append(best_params)

    print('Mean of 5 scores: %2.3f' % np.mean(all_scores))
    print('Std of 5 scores: %2.3f' % np.std(all_scores))
    return


if __name__ == '__main__':
    X, Y = get_training_data()
    get_multiple_runs(X,Y)

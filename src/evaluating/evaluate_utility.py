import pandas as pd
import shap
from src.evaluating.model_fine_tuner import ModelFineTuner

def plot_shap_analysis(model, x_train, plot_size=[15, 6], title=None, max_display=20, plot_type="dot", col=None):
    """plot shap graph for a given model trained on a given dataframe
    Args:
        model (model): model to describe
        x_train (DataFrame): _description_
        plot_size (list, optional): Plot size. Defaults to [15,6].
        title (str, optional): plot title. Defaults to None.
        max_display (int, optional): Number of features to display. Defaults to 20.
        plot_type (str, optional): plot type, can be 'dot' or 'bar'. Defaults to "dot".
        col (list, optional): List of columns to display in the graph, if None, all columns are considered. Defaults to None.
    """
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(x_train)
    shap_values = pd.DataFrame(shap_values[1], columns=x_train.columns)
    if col == None:
        col = x_train.columns
    shap.summary_plot(shap_values[col].values, x_train[col], plot_size=plot_size,
                      title=title, max_display=max_display, plot_type=plot_type)


def discriminate_synth_data(df_real, df_synth, models_params, perf_metrics, best_perf_metric, n_cv=3, split_perf=False, cv_random_state=10, shap=False):
    """
    Train models to discriminate real vs simulated data. 
    NB: Check before that de_real and df_synth have same columns, same types and same encoding
    """
    df_real['synth']=0
    df_synth['synth']=1
    df = pd.concat([df_real, df_synth])
    y_classif = df['synth']
    X_classif = df.drop('synth', axis=1)
    
     # Initialize the ModelFineTuner
    fine_tuner = ModelFineTuner(
        models_params, perf_metrics, best_perf_metric, n_cv=n_cv, split_perf=split_perf)

    # Fitting of the models
    fine_tuner.fit(X_classif.values, y_classif.values)
    fine_tuner.refit_best_model(X_classif, y_classif)
    
    if shap:
        plot_shap_analysis(fine_tuner.best_model, X_classif)

    return fine_tuner.best_perf, fine_tuner.df_cv_results
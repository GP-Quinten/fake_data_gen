import pandas as pd
from src.evaluating.model_fine_tuner import ModelFineTuner


def discriminate_synth_data(
    df_real,
    df_synth,
    models_params,
    perf_metrics,
    best_perf_metric,
    n_cv=3,
    split_perf=False,
    cv_random_state=10,
):
    """
    Train models to discriminate real vs simulated data.
    NB: Check before that de_real and df_synth have same columns, same types and same encoding
    """
    df_real["synth"] = 0
    df_synth["synth"] = 1
    df = pd.concat([df_real, df_synth])
    y_classif = df["synth"]
    X_classif = df.drop("synth", axis=1)

    model_fine_tuner = ModelFineTuner(
        models_params, perf_metrics, best_perf_metric, n_cv, split_perf, cv_random_state
    )

    # Initialize the ModelFineTuner
    fine_tuner = ModelFineTuner(
        models_params, perf_metrics, best_perf_metric, n_cv=n_cv, split_perf=split_perf
    )

    # Fitting of the models
    fine_tuner.fit(X_classif.values, y_classif.values)
    fine_tuner.refit_best_model(X_classif, y_classif)

    return fine_tuner.best_perf, fine_tuner.df_cv_results

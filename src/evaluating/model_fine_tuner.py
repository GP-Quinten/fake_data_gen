import os
import logging
import ast
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier


class ModelFineTuner:
    """
    Attributes:
        models_params (dict): models and parameters to test
        n_models (int): number of model types to benchmark
        model_hyperparameters (dict): models and parameters to test
        n_cv (int): fold number for cross validation. Default: 3
        grid_searches (dict): contains key: model name, value: gridsearch output results
        perf_metrics (list): list of performance metrics to compute during grid search.
        best_perf_metric (str): reference performance metric for best model choice.
            Must be included in perf_metrics.
        split_perf (boolean):
            if False (Default): average of the performance metrics values over the folds
            if True: performance metrics value per fold
    """

    def __init__(
        self,
        models_params,
        perf_metrics,
        best_perf_metric,
        n_cv=3,
        split_perf=False,
        cv_random_state=10,
    ):
        """
        Constructs all the necessary attributes for the ModelFineTuner object.

        Args:
            models_params (dict): models and parameters to test
            perf_metrics (list): list of performance metrics to compute during grid search.
            best_perf_metric (str): reference performance metric for best model choice.
                Must be included in perf_metrics.
            n_cv (int): fold number for cross validation. Default: 3
            split_perf (boolean):
                if False (Default): average of the performance metrics values over the folds
                if True: performance metrics value per fold

        """
        logging.info("\n" + "*" * 20 + "\n* model_fine_tuner *\n" + "*" * 20)
        self.model_names = models_params.keys()
        self.n_models = len(models_params.keys())
        self.model_hyperparameters = models_params
        self.n_cv = n_cv
        self.perf_metrics = perf_metrics
        self.best_perf_metric = best_perf_metric
        self.split_perf = split_perf
        self.cv_random_state = cv_random_state
        self._init_models()

        # Initialize fit/performance_cv_summary attributes
        self.grid_searches = {}
        self.df_cv_results = pd.DataFrame()
        self.df_best_models = pd.DataFrame()

        # Initialize refit_best_model attributes
        self.best_model = ""
        self.best_model_name = ""
        self.best_grid_hyperparameters = {}
        self.best_perf = 0

    def _init_models(self):
        """
        Initializes models from models' name in self.model_hyperparameters.
        Default hyperparameters are used.

        """
        # Initialize decision tree model
        if "dt" in self.model_names:
            self.model_hyperparameters["dt"]["model"] = DecisionTreeClassifier(
                **self.model_hyperparameters["dt"]["default_hyperparameters"]
            )

        # Initialize random forest model
        if "rf" in self.model_names:
            self.model_hyperparameters["rf"]["model"] = RandomForestClassifier(
                **self.model_hyperparameters["rf"]["default_hyperparameters"]
            )

        # Initialize LightGBM model
        if "lgbm" in self.model_names:
            self.model_hyperparameters["lgbm"]["model"] = LGBMClassifier(
                **self.model_hyperparameters["lgbm"]["default_hyperparameters"]
            )

        # Initialize support vector machine model
        if "svm" in self.model_names:
            self.model_hyperparameters["svm"]["model"] = SVC(
                **self.model_hyperparameters["svm"]["default_hyperparameters"]
            )

        # Initialize logistic regression model
        if "lr" in self.model_names:
            self.model_hyperparameters["lr"]["model"] = LogisticRegression(
                **self.model_hyperparameters["lr"]["default_hyperparameters"]
            )

    def fit(self, X, y):
        """
        Performs grid search based on self.model_hyperparameters and stores results in self.grid_searches.
        self.grid_searches is completed.

        Args:
            X (pandas.DataFrame): features dataset
            y (pandas.Series): outcome array or series
        """
        # Fit a gridsearch for each model type
        for model_name in self.model_names:
            logging.info("Running GridSearchCV for %s." % model_name)
            model = self.model_hyperparameters[model_name]["model"]
            params_grid = self.model_hyperparameters[model_name]["grid_hyperparameters"]
            cv = StratifiedKFold(
                n_splits=self.n_cv, shuffle=True, random_state=self.cv_random_state
            )
            gs = GridSearchCV(
                model,
                params_grid,
                cv=self.n_cv,
                n_jobs=-1,
                refit=False,
                scoring=self.perf_metrics,
                return_train_score=False,
            )
            gs.fit(X, y)
            self.grid_searches[model_name] = gs

        # Get the corresponding performances
        self.performance_cv_summary()

    @staticmethod
    def _filter_string_list(string_l, substr_l):
        """Selects strings from the string_l list containing strings from the substr_l list.
        Args:
            string_l (list of strings): list to filter
            substr_l (list of strings): _description_
        Returns:
            list of strings: elements of string_l containing substrings from substr_l.
        """
        filtered_string_l = [s for s in string_l if any(sub in s for sub in substr_l)]
        return filtered_string_l

    def performance_cv_summary(self):
        """Summarizes models performances based on the grid search performances. Rank the models based on best_perf_metric."""
        # Create empty list that will store performance dataframes for each model type.
        dfs_cv_results = []
        for model_name in self.model_names:
            df_temp = pd.DataFrame(self.grid_searches[model_name].cv_results_)

            # Check if the output should contain performance at the split level.
            if self.split_perf:
                cols_to_keep = self._filter_string_list(
                    df_temp.columns, ["params"] + self.perf_metrics
                )
                cols_to_keep = [col for col in cols_to_keep if "rank" not in col]
            else:
                cols_to_keep = self._filter_string_list(
                    df_temp.columns, ["params", "mean", "std"]
                )
            df_temp = df_temp[cols_to_keep]

            # Add default_hyperparameters in the dataframe in the first column
            df_temp.insert(
                loc=0,
                column="default_hyperparameters",
                value=str(
                    self.model_hyperparameters[model_name]["default_hyperparameters"]
                ),
            )

            # Add estimator name (model_name) in the dataframe in the first column
            df_temp.insert(loc=0, column="estimator", value=model_name)
            dfs_cv_results.append(df_temp)

        # Concatenate all performance output dataframes
        df_cv_results = pd.concat(dfs_cv_results)
        df_cv_results.reset_index(inplace=True, drop=True)

        # Create column rank in the dataframe based on self.best_perf_metric
        df_cv_results.insert(
            loc=0,
            column="rank",
            value=df_cv_results["mean_test_{}".format(self.best_perf_metric)].rank(
                ascending=0, method="min"
            ),
        )
        df_cv_results = df_cv_results.sort_values("rank", axis=0)

        # Save df_cv_results
        self.df_cv_results = df_cv_results

        # Select the best model regarding self.best_perf_metric. We may have multiple best models
        self.df_best_models = self.df_cv_results[self.df_cv_results["rank"] == 1]
        self.df_best_models.reset_index(inplace=True)
        logging.info(f"The best model(s) is/are: {self.df_best_models}")

    def refit_best_model(self, X, y):
        """Select the first best model (among the best models if there are several) and refit it on the data.
        Args:
            X (pandas.DataFrame): features dataset
            y (pandas.Series): outcome array or series
        """
        # Select one best model to print/log
        df_best_model = self.df_best_models.iloc[0, :]
        best_model_name = df_best_model["estimator"]
        best_params = ast.literal_eval(df_best_model["default_hyperparameters"])
        best_params.update(df_best_model["params"])
        best_perf = df_best_model[f"mean_test_{self.best_perf_metric}"]
        logging.info(
            "We have {} best model(s). The best model is:\n- Estimator {},\n- Hyperparameters: {}".format(
                len(self.df_best_models), best_model_name, best_params
            )
        )
        best_model = self.model_hyperparameters[best_model_name]["model"]
        best_model.set_params(**best_params)
        best_model.fit(X, y.ravel())
        self.best_model = best_model
        self.best_model_name = best_model_name
        self.best_params = best_params
        self.best_perf = best_perf

    def save_fine_tuning_results(self, X, y, run_name, experiment_name, mlflow_path):
        """Save results of the multiple grid search and of best model found in a folder, using Mlflow.
        Args:
            X (pandas.DataFrame): features dataset
            y (pandas.Series): outcome array or series
            run_name (string): name of the Mlflow run
            mlflow_path (string): path of the folder where the Mlflow run will be saved
        """
        # Initialize MLflow
        mlflow_tracking_uri = mlflow_path
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        experiment = MlflowClient().get_experiment_by_name(experiment_name)
        exp_id = (
            experiment.experiment_id
            if experiment
            else MlflowClient().create_experiment(experiment_name)
        )
        # Start mlflow run
        with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
            self.df_cv_results.to_csv("df_cv_results.csv")
            mlflow.log_artifact("df_cv_results.csv")
            os.remove("df_cv_results.csv")
            self.refit_best_model(X, y)
            mlflow.sklearn.log_model(self.best_model, "best_model")
            mlflow.log_param("best_model_name", self.best_model_name)
            mlflow.log_dict(self.best_params, "best_params")
            mlflow.log_metric("best_perf", self.best_perf)
        mlflow.end_run()

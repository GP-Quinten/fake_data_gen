import mlflow
import numpy as np
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient
import logging


def setting_mlflow(
    uri: str,
    exp_name: str,
):
    """Setting MLFlow experiment : tracking uri and experiement id. Returning exp_id"""
    mlflow_tracking_uri = uri
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    experiment_name = exp_name
    experiment = MlflowClient().get_experiment_by_name(experiment_name)
    exp_id = (
        experiment.experiment_id
        if experiment
        else MlflowClient().create_experiment(experiment_name)
    )

    logging.info("MLFlow Experiment")
    logging.info(
        "Experiment Name : " + str(exp_name) + "; " + "Experiment ID : " + str(exp_id)
    )
    logging.info("-" * 30)
    logging.info("-" * 30)

    return exp_id


def log_parameters(parameters_names: list, parameters_to_log: list):
    """mlflow logging parameters"""
    for parameters_name, parameter_to_log in zip(parameters_names, parameters_to_log):
        mlflow.log_param(parameters_name, parameter_to_log)


def plot_losses(g_losses, d_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(
        np.arange(len(d_losses)),
        d_losses,
        label="Discriminator Loss",
    )
    plt.plot(np.arange(len(g_losses)), g_losses, label="Generator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    return plt.gcf()


def plot_losses_with_score(g_losses, d_losses, df_score, df_corr):
    df_score_mean = df_score.mean(axis=1)
    df_corr_mean = df_corr.mean(axis=1)
    plt.figure(figsize=(10, 5))
    plt.plot(
        np.arange(len(d_losses)),
        d_losses,
        label="Discriminator Loss",
    )
    plt.plot(np.arange(len(g_losses)), g_losses, label="Generator Loss")
    plt.plot(np.arange(len(df_score_mean)), df_score_mean, label="Statistic similarity")
    plt.plot(np.arange(len(df_corr_mean)), df_corr_mean, label="Correlation similarity")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    return plt.gcf()

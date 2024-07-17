import os
import sys

script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)

import torch
import mlflow
import warnings
import logging

warnings.filterwarnings("ignore")

import config.config as conf
from src.loading import (
    read_data,
    read_dict,
    save_csv,
    save_model,
    save_figure_s3,
)
from src.logger import init_logger
from src.parsers.parser_training_ctabgan import ctabgan_parser
from src.modelling.ctabgan.model import ctabgan
from src.evaluating import evaluate_train
from src.visualizing import visualizing
from src.utils import training_utils, training_ctabgan_utils


def log_mlflow_info(parameters_to_log: dict):
    """ Logging given parameters of metrics in mlflow"""
    logging.info(parameters_to_log)
    training_utils.log_parameters(list(parameters_to_log.keys()), list(parameters_to_log.values()))

def log_mlflow_fig(fig_plot_losses, fig_plot_all_losses, fig_diags_prev, cache_figures_column_plot, cache_figures_column_pair_plot):
    """ Logging given figures in MLFlow"""
    mlflow.log_figure(fig_plot_losses, f"convergence/losses.png")
    mlflow.log_figure(fig_plot_all_losses, f"convergence/all_losses.png")
    mlflow.log_figure(fig_diags_prev, f"top_prevalences.png")

    for name, fig in cache_figures_column_plot.items():
        mlflow.log_figure(fig, f"distributions/{name}_column_plot.png")

    for name, fig in cache_figures_column_pair_plot.items():
        mlflow.log_figure(fig, f"correlations/{name}_column_pair_plot.png")

def load_data():
    """ Loading data """
    # Load data
    df_prepared = read_data(
        conf.BUCKET_NAME, conf.PATH_OUTPUT_DATA, conf.FILE_PREPARED_DATA
    )
    metadata = read_dict(
        conf.BUCKET_NAME, os.path.join(conf.PATH_METADATA, conf.FILE_METADATA)
    )
    return df_prepared, metadata

def save_data(df_preproc, synthetic_data, model, fig_plot_losses, fig_plot_all_losses, fig_diags_prev, save_synth_data,  run_name):
    """ Saving data """
    # save preprocessed data
    save_csv(
        df_preproc, conf.BUCKET_NAME, conf.PATH_OUTPUT_DATA, conf.FILE_PREPROC_DATA
    )
    # save model
    save_model(
        model,
        conf.BUCKET_NAME,
        conf.PATH_MODEL,
        conf.FILE_MODEL,
    )
    # Not necessary to save generated data every time
    if save_synth_data:
        # save synthetic data generated
        save_csv(
                synthetic_data,
                conf.BUCKET_NAME,
                conf.PATH_OUTPUT_DATA,
                conf.FILE_SYNTHESIZED_DATA,
            )
    
    # save figures
    logging.info(
        "All evaluation post training scores and figures will be saved in {}".format(
            os.path.join(
                "s3://",
                conf.BUCKET_NAME,
                conf.PATH_EVALUATE,
                f"evaluate_post_training/{run_name}",
            )
        )
    )
    save_figure_s3(
        fig_plot_losses,
        conf.BUCKET_NAME,
        conf.PATH_EVALUATE,
        f"evaluate_post_training/{run_name}/convergence/losses",
    )

    save_figure_s3(
        fig_plot_all_losses,
        conf.BUCKET_NAME,
        conf.PATH_EVALUATE,
        f"evaluate_post_training/{run_name}/convergence/losses",
    )

    save_figure_s3(
            fig_diags_prev,
            conf.BUCKET_NAME,
            conf.PATH_EVALUATE,
            f"evaluate_post_training/{run_name}/prevalences/all_diags_prev.png",
        )
    


def main():
    # Initiate parser
    parser = ctabgan_parser()
    args = parser.parse_args()

    # Initiate logger
    init_logger(level=args.log_level, file=True, file_path="logs/logs.txt")

    # Initiate MLFLow
    exp_id = training_utils.setting_mlflow(args.mlflow_uri, args.exp_name)

    ## -- Load data -- ##
    df_prepared, metadata = load_data()

    ## -- Preprocessing -- ##
    df_preproc = df_prepared  ## no preprocessing
    real_data = df_preproc.copy()

    ## Compute diags prevalences for the real dataset
    diag_list, df_prev_real = evaluate_train.compute_df_prev(real_data, "real")

    ## looping on args we can loop on
    for epochs in args.epochs:
        for batch_size in args.batch_size:
            for lr in args.lr:
                for l2scale in args.l2scale:
                    torch.cuda.empty_cache()
                    run_name = f"Ep_{epochs}_bs_{batch_size}_lr_{lr}_l2_{l2scale}"
                    logging.info("Run Name : " + run_name)
                    with mlflow.start_run(
                        experiment_id=exp_id, run_name=run_name
                    ) :

                        ## -- Create model structure -- ##
                        synthesizer = ctabgan.CTABGAN(
                            df_real_data=real_data,
                            categorical_columns=conf.CONFIG_CTABGAN["CATEGORICAL_COLUMNS"] + diag_list,
                            log_columns=conf.CONFIG_CTABGAN["LOG_COLUMNS"],
                            mixed_columns=conf.CONFIG_CTABGAN["MIXED_COLUMNS"],
                            integer_columns=conf.CONFIG_CTABGAN["INTEGER_COLUMNS"],
                            problem_type=conf.CONFIG_CTABGAN["PROBLEM_TYPE"],
                            test_ratio=conf.CTABGAN_TEST_RATIO,
                            epochs=epochs,
                            batch_size=batch_size,
                            lr=lr,
                            l2scale=l2scale,
                        )

                        ## -- Training -- ##
                        logging.info("Fitting the synthesizer to the training dataset and generating synthetic data")
                        synthesizer.fit()
                        
                        ## -- Generation of samples -- ##
                        # Generating synthetic samples
                        synthetic_data = synthesizer.generate_samples()
                        logging.info("Synthetic samples generated")

                        # Making sure columns of synthetic data and real data are of the same type
                        for col in real_data.columns:
                            synthetic_data[col] = synthetic_data[col].astype(
                                real_data[col].dtype
                            )
                
                        ## -- Evaluate train -- ##
                        logging.info("EVALUATE TRAIN")

                        # Convergence of losses
                        (
                            fig_plot_losses,
                            fig_plot_all_losses,
                        ) = training_ctabgan_utils.training_losses_evolution(
                            synthesizer
                        )
                        

                        # Computing prevalences of synthetic dataset and comparing them to the real ones
                        _, df_prev_synth = evaluate_train.compute_df_prev(synthetic_data, "synthetic")

                        # Comparing the prevalences of both datasets
                        fig_diags_prev = visualizing.plot_diags_prev(df_prev_real, df_prev_synth)
                        

                        ## ----- Scores ----- ##
                        (quality_report, quality_df, overall_score, diagnostic_report, diagnostic_df) = (
                            evaluate_train.evaluate_train_quality_and_diagnostic(real_data, synthetic_data, metadata))

                        ## -- Visualization -- ##
                        ## Generate Sdv report visualizations and logging in mlflow ##
                        cache_figures = visualizing.viz_reports(
                            quality_report,
                            diagnostic_report,
                            cols=real_data.columns,
                        )
                        for name, fig in cache_figures.items():
                            mlflow.log_figure(fig, f"reports/{name}.png")

                        ## Generate Distribs and scatter plots ##
                        # Retrieve numerical colummns
                        numerical_columns = list(real_data.columns)
                        for column in list(real_data.columns):
                            if column in conf.CONFIG_CTABGAN["CATEGORICAL_COLUMNS"] + diag_list:
                                numerical_columns.remove(column)

                        (cache_figures_column_plot, cache_figures_column_pair_plot) = (
                            visualizing.plotting_distrib_and_correlations(real_data, synthetic_data, metadata, numerical_columns))

                        ## -- MLFlow logging -- ##
                        # Parameters
                        parameters_to_log = {
                            "epochs": epochs,
                            "batch size": batch_size,
                            "problem_type": conf.CONFIG_CTABGAN["PROBLEM_TYPE"],
                            "data_size": len(real_data),
                            "learning rate": lr,
                            "L2 SCALE": l2scale,
                        }
                        log_mlflow_info(parameters_to_log)
                        # Metrics
                        metrics_to_log = {
                            "Columns shapes Score": quality_df.loc[0, "Score"],
                            "Column Pair Trends Score": batch_size,
                            "Overall quality score": overall_score,
                            "Synthesis Score": diagnostic_df["Synthesis"],
                            "Coverage Score": diagnostic_df["Coverage"],
                        }
                        log_mlflow_info(metrics_to_log)
                        # figures
                        log_mlflow_fig(fig_plot_losses,
                                       fig_plot_all_losses,
                                       fig_diags_prev,
                                       cache_figures_column_plot,
                                       cache_figures_column_pair_plot)
                    
                        ## -- Saving all the data and figures -- ##
                        save_data(df_preproc,
                                  synthetic_data,
                                  synthesizer,
                                  fig_plot_losses,
                                  fig_plot_all_losses,
                                  fig_diags_prev,
                                  run_name,
                                  args.save_synth_data)


if __name__ == "__main__":
    main()

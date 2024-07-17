import torch
import mlflow
import warnings
import logging
import time
import sys
import os

script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)

warnings.filterwarnings("ignore")

from ctgan import load_demo

from config import config
from src.utils import training_utils, utils
from src import loading
from src.logger import init_logger
from src.parsers.parser_training_ctgan import training_parser
from src.modelling.ctgan import custom_ctgan
from src.evaluating import evaluate_train

MODEL = config.MODEL

def main():

    # Initiate parser
    parser = training_parser()
    args = parser.parse_args()

    # Initiate logger
    init_logger(level=args.log_level, file=True, file_path="logs/logs.txt")
    mlflow_tracking_uri = args.mlflow_uri
    exp_name = args.exp_name
    exp_id = training_utils.setting_mlflow(mlflow_tracking_uri, exp_name)

    if args.test_dataset:
        # test dataset is adults.csv
        real_data = load_demo()
        logging.info("Data loaded")
        discrete_columns = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
            "income",
        ]

    else:
        real_data = loading.read_data(
            config.BUCKET_NAME, args.dataset_path, args.dataset_filename
        )
        logging.info("Data loaded")

        columns_categories = utils.categorize_columns(real_data)

        discrete_columns = columns_categories["discrete"]

    if args.discrete_columns:
        discrete_columns = args.discrete_columns

    generator_lr = args.learning_rate
    discriminator_lr = args.learning_rate

    for (d_lr, g_lr) in zip(discriminator_lr, generator_lr):
        logging.info(f"Testing for discriminator lr = {d_lr} and generator lr = {g_lr}")
        run_name = f"CTGAN_{config.TODAY}"
        with mlflow.start_run(experiment_id=exp_id, run_name=run_name):

            model = custom_ctgan.custom_CTGAN(
                epochs=args.epochs,
                discriminator_lr=d_lr,
                generator_lr=g_lr,
                discriminator_decay=args.discriminator_decay,
                generator_decay=args.generator_decay,
                batch_size=args.batch_size,
            )

            parameters_to_log = {
                "discriminator_lr": d_lr,
                "generator_lr": g_lr,
                "discriminator_decay": args.discriminator_decay,
                "generator_decay": args.generator_decay,
                "epochs": args.epochs,
                "batch size": args.batch_size,
                "data_size": len(real_data),
            }

            parameters = [value for key, value in parameters_to_log.items()]
            parameters_names = [key for key, value in parameters_to_log.items()]

            training_utils.log_parameters(parameters_names, parameters)

            if args.n_to_eval != 0:
                t0 = time.time()
                logging.info("Fitting starts")
                model.fit(
                    real_data,
                    "SUBJECT_ID",
                    discrete_columns,
                    n_samples_to_eval=args.n_to_eval,
                )
                fitting_time = (time.time() - t0) / 60
                mlflow.log_param("fitting time in min", fitting_time)
                logging.info(f"Fitting ended in {fitting_time} minutes")
                mlflow.log_param(
                    f"number of rows sampled for evaluation", args.n_to_eval
                )

                stats_scores, corr_scores = model.return_scores()
                mlflow.log_input(stats_scores, context="evaluate")
                mlflow.log_input(corr_scores, context="evaluate")

            else:
                t0 = time.time()
                logging.info("Fitting starts")
                model.fit(real_data, discrete_columns, "SUBJECT_ID")
                fitting_time = (time.time() - t0) / 60
                mlflow.log_param("fitting time in min", fitting_time)
                logging.info(f"Fitting ended in {fitting_time} minutes")

            # return losses to plot them and register the last ones
            losses_d, losses_g = model.return_losses()
            last_d_loss = losses_d[-1]
            last_g_loss = losses_g[-1]
            mlflow.log_metric("last d loss", last_d_loss)
            mlflow.log_metric("last g loss", last_g_loss)

            if args.n_to_eval != 0:
                plot_losses = training_utils.plot_losses_with_score(
                    losses_g, losses_d, stats_scores, corr_scores,
                )
            else:
                plot_losses = training_utils.plot_losses(losses_g, losses_d)
            
            mlflow.log_figure(plot_losses, "plot_losses.png")
            loading.save_figure_s3(
                plot_losses,
                config.BUCKET_NAME,
                os.path.join(config.PATH_EVALUATE, "evaluate_while_training/"),
                "plot_losses",
            )

            if args.n_to_sample:
                synth_data = model.sample(args.n_to_sample)
                dict_metadata = loading.read_dict(
                    config.BUCKET_NAME,
                    os.path.join(config.PATH_METADATA, config.FILE_METADATA),
                )

                cache_figures, scores_dataset = evaluate_train.evaluate_after_training(
                    real_data,
                    synth_data,
                    dict_metadata,
                    args.col_to_plot,
                )

                for name, fig in cache_figures.items():
                    loading.save_figure_s3(
                        fig,
                        config.BUCKET_NAME,
                        os.path.join(config.PATH_EVALUATE, "evaluate_post_training/", "plots/"),
                        f"{name}_ctgan_{args.exp_name}",
                    )

                    mlflow.log_figure(fig, f"distribution_{name}.png")

                loading.save_csv(
                    scores_dataset,
                    config.BUCKET_NAME,
                    os.path.join(config.PATH_EVALUATE, "evaluate_post_training/", "dataframes/"),
                    f"df_scores_ctgan_{args.exp_name}.csv"
                    
                )

                for (col, value) in zip(
                    scores_dataset.columns.to_list(),
                    scores_dataset.loc[
                        0,
                    ],
                ):
                    mlflow.log_metric(col, value)

            mlflow.log_artifact(args.output_path)
            mlflow.end_run()

    if args.n_to_sample:
        loading.save_csv(
            synth_data,
            config.BUCKET_NAME,
            config.PATH_SYNTH_DATA, 
            config.FILE_SYNTHESIZED_DATA,
        )

    loading.save_model(model, config.BUCKET_NAME, config.PATH_MODEL, config.FILE_MODEL)


if __name__ == "__main__":
    main()

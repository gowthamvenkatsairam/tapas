from logger import MLFlowLogger
ml_logger = MLFlowLogger(tracking_uri="http://34.122.1.15:6006")
ml_logger.init_experiment(experiment_name="sqa", run_name="tapas_sqa")

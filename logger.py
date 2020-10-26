import mlflow
class BaseMLLogger:
    """
    Base class for tracking experiments.
    This class can be extended to implement custom logging backends like MLFlow, Tensorboard, or Sacred.
    """

    def __init__(self, tracking_uri, **kwargs):
        self.tracking_uri = tracking_uri
        print("EXPERIMENT INITIALIZED")

    def init_experiment(self, tracking_uri):
        raise NotImplementedError()

    @classmethod
    def log_metrics(cls, metrics, step):
        raise NotImplementedError()

    @classmethod
    def log_artifacts(cls, self):
        raise NotImplementedError()

    @classmethod
    def log_params(cls, params):
        raise NotImplementedError()
class MLFlowLogger(BaseMLLogger):
    """
    Logger for MLFlow experiment tracking.
    """

    def init_experiment(self, experiment_name, run_name=None, nested=True):
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(experiment_name)
            mlflow.start_run(run_name=run_name, nested=nested)
        except ConnectionError:
            raise Exception(
                f"MLFlow cannot connect to the remote server at {self.tracking_uri}.\n"
                f"MLFlow also supports logging runs locally to files. Set the MLFlowLogger "
                f"tracking_uri to an empty string to use that."
            )

    @classmethod
    def log_metrics(cls, metrics, step):
        try:
            mlflow.log_metrics(metrics, step=step)
        except ConnectionError:
            logger.warning(f"ConnectionError in logging metrics to MLFlow.")
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    @classmethod
    def log_params(cls, params):
        try:
            mlflow.log_params(params)
        except ConnectionError:
            logger.warning("ConnectionError in logging params to MLFlow")
        except Exception as e:
            logger.warning(f"Failed to log params: {e}")

    @classmethod
    def log_artifacts(cls, dir_path, artifact_path=None):
        try:
            mlflow.log_artifacts(dir_path, artifact_path)
        except ConnectionError:
            logger.warning(f"ConnectionError in logging artifacts to MLFlow")
        except Exception as e:
            logger.warning(f"Failed to log artifacts: {e}")

    @classmethod
    def end_run(cls):
        mlflow.end_run()
import json
import sagemaker
import boto3
import mlflow
from time import gmtime, strftime
from sagemaker.estimator import Estimator
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.model_metrics import (
    MetricsSource, 
    ModelMetrics, 
    FileSource
)

def register(
    training_job_name,
    model_package_group_name,
    model_approval_status,
    evaluation_result,
    output_s3_prefix,
    tracking_server_arn,
    model_data_quality_check_statistics=None,
    model_data_quality_check_contraints=None,
    model_data_bias_check_constraints=None,
    model_quality_check_statistics=None,
    model_quality_check_constraints=None,
    model_bias_check_constraints=None,
    model_explainability_contraints=None,
    drift_data_quality_check_statistics=None,
    drift_data_quality_check_contraints=None,
    drift_data_bias_check_contraints=None,
    drift_model_bias_check_config=None,
    drift_model_quality_check_statistics=None,
    drift_model_quality_check_constraints=None,
    drift_model_bias_check_constraints=None,
    drift_model_explainability_check_constraints=None,
    drift_model_explainability_check_config=None,
    experiment_name=None,
    pipeline_run_id=None,
    run_id=None,
):
    try:
        suffix = strftime('%d-%H-%M-%S', gmtime())
        mlflow.set_tracking_uri(tracking_server_arn)
        experiment = mlflow.set_experiment(experiment_name=experiment_name if experiment_name else f"{register.__name__ }-{suffix}")
        pipeline_run = mlflow.start_run(run_id=pipeline_run_id) if pipeline_run_id else None            
        run = mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run(run_name=f"register-{suffix}", nested=True)

        evaluation_result_path = f"evaluation.json"
        with open(evaluation_result_path, "w") as f:
            f.write(json.dumps(evaluation_result))
            
        mlflow.log_artifact(local_path=evaluation_result_path)
            
        estimator = Estimator.attach(training_job_name)
        
        model_metrics = ModelMetrics(
            model_data_statistics=MetricsSource(
                s3_uri=model_data_quality_check_statistics,
                content_type="application/json",
            ) if model_data_quality_check_statistics else None,
            model_data_constraints=MetricsSource(
                s3_uri=model_data_quality_check_contraints,
                content_type="application/json",
            ) if model_data_quality_check_contraints else None,
            bias_pre_training=MetricsSource(
                s3_uri=model_data_bias_check_constraints,
                content_type="application/json",
            ) if model_data_bias_check_constraints else None,
            model_statistics=MetricsSource(
                s3_uri=model_quality_check_statistics,
                content_type="application/json",
            ) if model_quality_check_statistics else None,
            model_constraints=MetricsSource(
                s3_uri=model_quality_check_constraints,
                content_type="application/json",
            ) if model_quality_check_constraints else None,
            bias_post_training=MetricsSource(
                s3_uri=model_bias_check_constraints,
                content_type="application/json",
            ) if model_bias_check_constraints else None,
            bias=MetricsSource(
                # This field can also be set as the merged bias report
                # with both pre-training and post-training bias metrics
                s3_uri=model_bias_check_constraints,
                content_type="application/json",
            ) if model_bias_check_constraints else None,
            explainability=MetricsSource(
                s3_uri=model_explainability_contraints,
                content_type="application/json",
            ) if model_explainability_contraints else None,
        )

        drift_check_baselines = DriftCheckBaselines(
            model_data_statistics=MetricsSource(
                s3_uri=drift_data_quality_check_statistics,
                content_type="application/json",
            ) if drift_data_quality_check_statistics else None,
            model_data_constraints=MetricsSource(
                s3_uri=drift_data_quality_check_contraints,
                content_type="application/json",
            ) if drift_data_quality_check_contraints else None,
            bias_pre_training_constraints=MetricsSource(
                s3_uri=drift_data_bias_check_contraints,
                content_type="application/json",
            ) if drift_data_bias_check_contraints else None,
            bias_config_file=FileSource(
                s3_uri=drift_model_bias_check_config,
                content_type="application/json",
            ) if drift_model_bias_check_config else None,
            model_statistics=MetricsSource(
                s3_uri=drift_model_quality_check_statistics,
                content_type="application/json",
            ) if drift_model_quality_check_statistics else None,
            model_constraints=MetricsSource(
                s3_uri=drift_model_quality_check_constraints,
                content_type="application/json",
            ) if drift_model_quality_check_constraints else None,
            bias_post_training_constraints=MetricsSource(
                s3_uri=drift_model_bias_check_constraints,
                content_type="application/json",
            ) if drift_model_bias_check_constraints else None,
            explainability_constraints=MetricsSource(
                s3_uri=drift_model_explainability_check_constraints,
                content_type="application/json",
            ) if drift_model_explainability_check_constraints else None,
            explainability_config_file=FileSource(
                s3_uri=drift_model_explainability_check_config,
                content_type="application/json",
            ) if drift_model_explainability_check_config else None
        )
    
        model_package = estimator.register(
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=["ml.m5.xlarge", "ml.m5.large"],
            transform_instances=["ml.m5.xlarge", "ml.m5.large"],
            model_package_group_name=model_package_group_name,
            approval_status=model_approval_status,
            model_metrics=model_metrics,
            drift_check_baselines=drift_check_baselines,
            model_name="from-idea-to-prod-pipeline-model",
            domain="MACHINE_LEARNING",
            task="CLASSIFICATION", 
        )

        mlflow.log_params({
            "model_package_arn":model_package.model_package_arn,
            "model_statistics_uri":model_quality_check_statistics if model_quality_check_statistics else '',
            "model_constraints_uri":model_quality_check_constraints if model_quality_check_constraints else '',
            "model_bias_constraints_uri":model_bias_check_constraints if model_bias_check_constraints else '',
            "model_explainability_constraints_uri":model_explainability_contraints if model_explainability_contraints else '',
            "data_statistics_uri":model_data_quality_check_statistics if model_data_quality_check_statistics else '',
            "data_constraints_uri":model_data_quality_check_contraints if model_data_quality_check_contraints else '',
            "data_bias_constraints_uri":model_data_bias_check_constraints if model_data_bias_check_constraints else '',
            "drift_data_statistics_uri":drift_data_quality_check_statistics if drift_data_quality_check_statistics else '',
            "drift_data_constraints_uri":drift_data_quality_check_contraints if drift_data_quality_check_contraints else '',
            "drift_data_bias_constraints_uri":drift_data_bias_check_contraints if drift_data_bias_check_contraints else '',
            "drift_model_statistics_uri":drift_model_quality_check_statistics if drift_model_quality_check_statistics else '',
            "drift_model_constraints_uri":drift_model_quality_check_constraints if drift_model_quality_check_constraints else '',
            "drift_model_bias_config_uri":drift_model_bias_check_config if drift_model_bias_check_config else '',
            "drift_model_bias_constraints_uri":drift_model_bias_check_constraints if drift_model_bias_check_constraints else '',
            "drift_model_explainability_constraints_uri":drift_model_explainability_check_constraints if drift_model_explainability_check_constraints else '',
            "drift_model_explainability_config_uri":drift_model_explainability_check_config if drift_model_explainability_check_config else '',

        })

        return {
            "model_package_arn":model_package.model_package_arn,
            "model_package_group_name":model_package_group_name,
            "pipeline_run_id":pipeline_run.info.run_id if pipeline_run else ''
        }

    except Exception as e:
        print(f"Exception in processing script: {e}")
        raise e
    finally:
        mlflow.end_run()
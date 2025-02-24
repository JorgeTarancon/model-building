import pandas as pd
import json
import boto3
import pathlib
import io
import os
import sagemaker
import mlflow
from time import gmtime, strftime, sleep
from sagemaker.deserializers import CSVDeserializer
from sagemaker.serializers import CSVSerializer

from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor
)
from sagemaker.inputs import TrainingInput

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    CreateModelStep,
    CacheConfig
)
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterFloat,
    ParameterString,
    ParameterBoolean
)
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    ModelQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.clarify_check_step import (
    DataBiasCheckConfig,
    ModelBiasCheckConfig,
    ModelPredictedLabelConfig,
    ClarifyCheckStep,
    ModelExplainabilityCheckConfig,
    SHAPConfig
)
from sagemaker.clarify import (
    BiasConfig,
    DataConfig,
    ModelConfig
)
from sagemaker import Model
from sagemaker.inputs import CreateModelInput
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.conditions import (
    ConditionGreaterThan,
    ConditionGreaterThanOrEqualTo
)
from sagemaker.workflow.parallelism_config import ParallelismConfiguration
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import (
    Join,
    JsonGet
)
from sagemaker.workflow.lambda_step import (
    LambdaStep,
    LambdaOutput,
    LambdaOutputTypeEnum,
)
from sagemaker.lambda_helper import Lambda

from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
    FileSource
)
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig 
from sagemaker.image_uris import retrieve
from sagemaker.workflow.function_step import step
from sagemaker.workflow.step_outputs import get_step
from sagemaker.model_monitor import DatasetFormat, model_monitoring

from pipelines.modeltraining.old_preprocess import preprocess
from pipelines.modeltraining.evaluate import evaluate
from pipelines.modeltraining.register import register

def get_sagemaker_client(region):
     return boto3.Session(region_name=region).client("sagemaker")

def get_pipeline_session(region, bucket_name):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        bucket_name: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=bucket_name,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_name=None):
    try:
        print(f"Getting project tags for {sagemaker_project_name}")

        sm_client = get_sagemaker_client(region)

        project_arn = sm_client.describe_project(ProjectName=sagemaker_project_name)['ProjectArn']
        project_tags = sm_client.list_tags(ResourceArn=project_arn)['Tags']

        print(f"Project tags: {project_tags}")

        for project_tag in project_tags:
            new_tags.append(project_tag)

    except Exception as e:
        print(f"Error getting project tags: {e}")

    return new_tags

def get_pipeline(
    region,
    sagemaker_project_id=None,
    sagemaker_project_name=None,
    role=None,
    bucket_name=None,
    bucket_prefix=None,
    input_s3_url=None,
    model_package_group_name=None,
    pipeline_name_prefix="training-pipeline",
    process_instance_type="ml.m5.large",
    train_instance_type="ml.m5.xlarge",
    test_score_threshold=0.70,
    tracking_server_arn=None,
):
    """Gets a SageMaker ML Pipeline instance.

    Returns:
        an instance of a pipeline
    """
    if input_s3_url is None:
        print("input_s3_url must be provided. Exiting...")
        return None

    session = get_pipeline_session(region, bucket_name)
    sm = session.sagemaker_client

    if role is None:
        role = sagemaker.session.get_execution_role(session)

    print(f"sagemaker version: {sagemaker.__version__}")
    print(f"Execution role: {role}")
    print(f"Input S3 URL: {input_s3_url}")
    print(f"Model package group: {model_package_group_name}")
    print(f"Pipeline name prefix: {pipeline_name_prefix}")
    print(f"Tracking server ARN: {tracking_server_arn}")

    pipeline_name = f"{pipeline_name_prefix}-{sagemaker_project_id}"
    experiment_name = pipeline_name

    output_s3_prefix = f"s3://{bucket_name}/{bucket_prefix}"
    # Set the output S3 url for model artifact
    output_s3_url = f"{output_s3_prefix}/output"
    # Set the output S3 url for feature store query results
    output_query_location = f'{output_s3_prefix}/offline-store/query_results'

    # Set the output S3 urls for processed data
    train_s3_url = f"{output_s3_prefix}/train"
    validation_s3_url = f"{output_s3_prefix}/validation"
    test_s3_url = f"{output_s3_prefix}/test"
    evaluation_s3_url = f"{output_s3_prefix}/evaluation"

    baseline_s3_url = f"{output_s3_prefix}/baseline"
    prediction_baseline_s3_url = f"{output_s3_prefix}/prediction_baseline"

    xgboost_image_uri = sagemaker.image_uris.retrieve(
            "xgboost", 
            region=region,
            version="1.5-1"
    )

    # If no tracking server ARN, try to find an active MLflow server
    if tracking_server_arn is None:
        r = sm.list_mlflow_tracking_servers(
            TrackingServerStatus='Created',
        )['TrackingServerSummaries']

        if len(r) < 1:
            print("You don't have any running MLflow servers. Exiting...")
            return None
        else:
            tracking_server_arn = r[0]['TrackingServerArn']
            print(f"Use the tracking server ARN:{tracking_server_arn}")

    # Parameters for pipeline execution

    # for data quality check step
    skip_check_data_quality = ParameterBoolean(name="SkipDataQualityCheck", default_value=False)
    register_new_baseline_data_quality = ParameterBoolean(name="RegisterNewDataQualityBaseline", default_value=False)
    supplied_baseline_statistics_data_quality = ParameterString(name="DataQualitySuppliedStatistics", default_value='')
    supplied_baseline_constraints_data_quality = ParameterString(name="DataQualitySuppliedConstraints", default_value='')

    # for data bias check step
    skip_check_data_bias = ParameterBoolean(name="SkipDataBiasCheck", default_value = False)
    register_new_baseline_data_bias = ParameterBoolean(name="RegisterNewDataBiasBaseline", default_value=False)
    supplied_baseline_constraints_data_bias = ParameterString(name="DataBiasSuppliedBaselineConstraints", default_value='')

    # for model quality check step
    skip_check_model_quality = ParameterBoolean(name="SkipModelQualityCheck", default_value = False)
    register_new_baseline_model_quality = ParameterBoolean(name="RegisterNewModelQualityBaseline", default_value=False)
    supplied_baseline_statistics_model_quality = ParameterString(name="ModelQualitySuppliedStatistics", default_value='')
    supplied_baseline_constraints_model_quality = ParameterString(name="ModelQualitySuppliedConstraints", default_value='')

    # for model bias check step
    skip_check_model_bias = ParameterBoolean(name="SkipModelBiasCheck", default_value=False)
    register_new_baseline_model_bias = ParameterBoolean(name="RegisterNewModelBiasBaseline", default_value=False)
    supplied_baseline_constraints_model_bias = ParameterString(name="ModelBiasSuppliedBaselineConstraints", default_value='')

    # for model explainability check step
    skip_check_model_explainability = ParameterBoolean(name="SkipModelExplainabilityCheck", default_value=False)
    register_new_baseline_model_explainability = ParameterBoolean(name="RegisterNewModelExplainabilityBaseline", default_value=False)
    supplied_baseline_constraints_model_explainability = ParameterString(name="ModelExplainabilitySuppliedBaselineConstraints", default_value='')

    # Set processing instance type
    process_instance_type_param = ParameterString(
        name="ProcessingInstanceType",
        default_value=process_instance_type,
    )

    # Set training instance type
    train_instance_type_param = ParameterString(
        name="TrainingInstanceType",
        default_value=train_instance_type,
    )

    # Set model approval param
    model_approval_status_param = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval"
    )

    # Minimal threshold for model performance on the test dataset
    test_score_threshold_param = ParameterFloat(
        name="TestScoreThreshold", 
        default_value=test_score_threshold
    )

    # S3 url for the input dataset
    input_s3_url_param = ParameterString(
        name="InputDataUrl",
        default_value=input_s3_url if input_s3_url else "None",
    )

    # Model package group name
    model_package_group_name_param = ParameterString(
        name="ModelPackageGroupName",
        default_value=model_package_group_name,
    )

    # MLflow tracking server ARN
    tracking_server_arn_param = ParameterString(
        name="TrackingServerARN",
        default_value=tracking_server_arn,
    )

    # Define step cache config
    cache_config = CacheConfig(
        enable_caching=True,
        expire_after="P30d" # 30-day
    )

    # Construct the pipeline

    # Get datasets
    step_get_datasets = step(
            preprocess, 
            role=role,
            instance_type=process_instance_type_param,
            name="preprocess",
            keep_alive_period_in_seconds=3600,
    )(
        input_data_s3_path=input_s3_url_param,
        output_s3_prefix=output_s3_prefix,
        tracking_server_arn=tracking_server_arn_param,
        experiment_name=experiment_name,
        pipeline_run_name=ExecutionVariables.PIPELINE_EXECUTION_ID,
    )

    ### Calculating the Data Quality

    # `CheckJobConfig` is a helper function that's used to define the job configurations used by the `QualityCheckStep`.
    # By separating the job configuration from the step parameters, the same `CheckJobConfig` can be used across multiple
    # steps for quality checks.
    # The `DataQualityCheckConfig` is used to define the Quality Check job by specifying the dataset used to calculate
    # the baseline, in this case, the training dataset from the data processing step, the dataset format, in this case,
    # a csv file with no headers, and the output path for the results of the data quality check.

    check_job_config = CheckJobConfig(
        role=role,
        instance_count=1,
        instance_type="ml.c5.xlarge",
        volume_size_in_gb=120,
        sagemaker_session=session,
    )

    data_quality_check_config = DataQualityCheckConfig(
        #baseline_dataset=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        baseline_dataset=step_get_datasets['train_data'],
        dataset_format=DatasetFormat.csv(header=False, output_columns_position="START"),
        output_s3_uri=Join(on='/', values=['s3:/', bucket_name, pipeline_name_prefix, ExecutionVariables.PIPELINE_EXECUTION_ID, 'dataqualitycheckstep'])
    )

    data_quality_check_step = QualityCheckStep(
        name="DataQualityCheckStep",
        skip_check=skip_check_data_quality,
        register_new_baseline=register_new_baseline_data_quality,
        quality_check_config=data_quality_check_config,
        check_job_config=check_job_config,
        supplied_baseline_statistics=supplied_baseline_statistics_data_quality,
        supplied_baseline_constraints=supplied_baseline_constraints_data_quality,
        model_package_group_name=model_package_group_name
    )

    #### Calculating the Data Bias

    # The job configuration from the previous step is used here and the `DataConfig` class is used to define how
    # the `ClarifyCheckStep` should compute the data bias. The training dataset is used again for the bias evaluation,
    # the column representing the label is specified through the `label` parameter, and a `BiasConfig` is provided.

    # In the `BiasConfig`, we specify a facet name (the column that is the focal point of the bias calculation),
    # the value of the facet that determines the range of values it can hold, and the threshold value for the label.
    # More details on `BiasConfig` can be found at
    # https://sagemaker.readthedocs.io/en/stable/api/training/processing.html#sagemaker.clarify.BiasConfig

    data_bias_analysis_cfg_output_path = f"s3://{bucket_name}/{pipeline_name_prefix}/databiascheckstep/analysis_cfg"

    data_bias_data_config = DataConfig(
        #s3_data_input_path=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        s3_data_input_path=step_get_datasets['train_data'],
        s3_output_path=Join(on='/', values=['s3:/', bucket_name, pipeline_name_prefix, ExecutionVariables.PIPELINE_EXECUTION_ID, 'databiascheckstep']),
        label="y",
        dataset_type="text/csv",
        s3_analysis_config_output_path=data_bias_analysis_cfg_output_path,
    )

    # We are using this bias config to configure clarify to detect bias based on the first feature in the featurized vector for Sex
    data_bias_config = BiasConfig(
        label_values_or_threshold=[1], facet_name=["job"], facet_values_or_threshold=[["blue-collar"]]
    )

    data_bias_check_config = DataBiasCheckConfig(
        data_config=data_bias_data_config,
        data_bias_config=data_bias_config,
    )

    data_bias_check_step = ClarifyCheckStep(
        name="DataBiasCheckStep",
        clarify_check_config=data_bias_check_config,
        check_job_config=check_job_config,
        skip_check=skip_check_data_bias,
        register_new_baseline=register_new_baseline_data_bias,
        model_package_group_name=model_package_group_name
    )

    # Instantiate an XGBoost estimator object
    estimator = sagemaker.estimator.Estimator(
        image_uri=xgboost_image_uri,
        role=role, 
        instance_type=train_instance_type_param,
        instance_count=1,
        output_path=output_s3_url,
        sagemaker_session=session,
        base_job_name=f"{pipeline_name}-train"
    )

    # Define algorithm hyperparameters
    estimator.set_hyperparameters(
        num_round=100, # the number of rounds to run the training
        max_depth=3, # maximum depth of a tree
        eta=0.5, # step size shrinkage used in updates to prevent overfitting
        alpha=2.5, # L1 regularization term on weights
        objective="binary:logistic",
        eval_metric="auc", # evaluation metrics for validation data
        subsample=0.8, # subsample ratio of the training instance
        colsample_bytree=0.8, # subsample ratio of columns when constructing each tree
        min_child_weight=3, # minimum sum of instance weight (hessian) needed in a child
        early_stopping_rounds=10, # the model trains until the validation score stops improving
        verbosity=1, # verbosity of printing messages
    )

    # train step
    step_train = TrainingStep(
        name="train",
        step_args=estimator.fit(
            {
                "train": TrainingInput(
                    step_get_datasets['train_data'],
                    content_type="text/csv",
                ),
                "validation": TrainingInput(
                    step_get_datasets['validation_data'],
                    content_type="text/csv",
                ),
            }
        ),
        depends_on=["DataQualityCheckStep", "DataBiasCheckStep"],
        cache_config=cache_config,
    )

    model = Model(
        image_uri=xgboost_image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=session,
        role=role,
    )

    step_args = model.create(
        instance_type="ml.m5.large",
        accelerator_type="ml.eia1.medium",
    )

    step_create_model = ModelStep(
        name="CreateModel",
        step_args=step_args,
    )

    ### Check the Model Quality

    # In this `QualityCheckStep` we calculate the baselines for statistics and constraints using the
    # predictions that the model generates from the test dataset (output from the TransformStep). We define
    # the problem type as 'Regression' in the `ModelQualityCheckConfig` along with specifying the columns
    # which represent the input and output. Since the dataset has no headers, `_c0`, `_c1` are auto-generated
    # header names that should be used in the `ModelQualityCheckConfig`.

    model_quality_check_config = ModelQualityCheckConfig(
        baseline_dataset=step_evaluate['prediction_baseline_data'],
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri=Join(on='/', values=['s3:/', bucket_name, pipeline_name_prefix, ExecutionVariables.PIPELINE_EXECUTION_ID, 'modelqualitycheckstep']),
        problem_type='BinaryClassification',
        inference_attribute='prediction',
        ground_truth_attribute='label'
    )

    model_quality_check_step = QualityCheckStep(
        name="ModelQualityCheckStep",
        skip_check=skip_check_model_quality,
        register_new_baseline=register_new_baseline_model_quality,
        quality_check_config=model_quality_check_config,
        check_job_config=check_job_config,
        supplied_baseline_statistics=supplied_baseline_statistics_model_quality,
        supplied_baseline_constraints=supplied_baseline_constraints_model_quality,
        model_package_group_name=model_package_group_name
    )

    ### Check for Model Bias

    # Similar to the Data Bias check step, a `BiasConfig` is defined and Clarify is used to calculate
    # the model bias using the training dataset and the model.


    model_bias_analysis_cfg_output_path = f"s3://{bucket_name}/{pipeline_name_prefix}/modelbiascheckstep/analysis_cfg"

    model_bias_data_config = DataConfig(
        s3_data_input_path=step_get_datasets['train_data'],
        s3_output_path=Join(on='/', values=['s3:/', bucket_name, pipeline_name_prefix, ExecutionVariables.PIPELINE_EXECUTION_ID, 'modelbiascheckstep']),
        s3_analysis_config_output_path=model_bias_analysis_cfg_output_path,
        label="y",
        dataset_type="text/csv",
    )

    model_config = ModelConfig(
        model_name=step_create_model.properties.ModelName,
        instance_count=1,
        instance_type='ml.m5.large',
    )

    # We are using this bias config to configure clarify to detect bias based on the first feature in the featurized vector for Sex
    model_bias_config = BiasConfig(
        label_values_or_threshold=[1], facet_name=["job"], facet_values_or_threshold=[["blue-collar"]]
    )

    model_bias_check_config = ModelBiasCheckConfig(
        data_config=model_bias_data_config,
        data_bias_config=model_bias_config,
        model_config=model_config,
        model_predicted_label_config=ModelPredictedLabelConfig()
    )

    model_bias_check_step = ClarifyCheckStep(
        name="ModelBiasCheckStep",
        clarify_check_config=model_bias_check_config,
        check_job_config=check_job_config,
        skip_check=skip_check_model_bias,
        register_new_baseline=register_new_baseline_model_bias,
        supplied_baseline_constraints=supplied_baseline_constraints_model_bias,
        model_package_group_name=model_package_group_name
    )

    ### Check Model Explainability

    # SageMaker Clarify uses a model-agnostic feature attribution approach, which you can used to understand
    # why a model made a prediction after training and to provide per-instance explanation during inference. The implementation
    # includes a scalable and efficient implementation of SHAP, based on the concept of a Shapley value from the field of
    # cooperative game theory that assigns each feature an importance value for a particular prediction.

    # For Model Explainability, Clarify requires an explainability configuration to be provided. In this example, we
    # use `SHAPConfig`. For more information of `explainability_config`, visit the Clarify documentation at
    # https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-explainability.html.

    model_explainability_analysis_cfg_output_path = "s3://{}/{}/{}/{}".format(
        bucket_name,
        pipeline_name_prefix,
        "modelexplainabilitycheckstep",
        "analysis_cfg"
    )

    model_explainability_data_config = DataConfig(
        s3_data_input_path=step_get_datasets['train_data'],
        s3_output_path=Join(on='/', values=['s3:/', bucket_name, pipeline_name_prefix, ExecutionVariables.PIPELINE_EXECUTION_ID, 'modelexplainabilitycheckstep']),
        s3_analysis_config_output_path=model_explainability_analysis_cfg_output_path,
        label=0,
        dataset_type="text/csv",
    )
    shap_config = SHAPConfig(
        seed=123,
        num_samples=10
    )
    model_explainability_check_config = ModelExplainabilityCheckConfig(
        data_config=model_explainability_data_config,
        model_config=model_config,
        explainability_config=shap_config,
    )
    model_explainability_check_step = ClarifyCheckStep(
        name="ModelExplainabilityCheckStep",
        clarify_check_config=model_explainability_check_config,
        check_job_config=check_job_config,
        skip_check=skip_check_model_explainability,
        register_new_baseline=register_new_baseline_model_explainability,
        supplied_baseline_constraints=supplied_baseline_constraints_model_explainability,
        model_package_group_name=model_package_group_name
    )

    # Evaluation step
    step_evaluate = step(
        evaluate,
        role=role,
        instance_type=process_instance_type_param,
        name="evaluate",
        keep_alive_period_in_seconds=3600,
    )(
        test_x_data_s3_path=step_get_datasets['test_x_data'],
        test_y_data_s3_path=step_get_datasets['test_y_data'],
        model_s3_path=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        output_s3_prefix=output_s3_prefix,
        tracking_server_arn=tracking_server_arn_param,
        experiment_name=step_get_datasets['experiment_name'],
        pipeline_run_id=step_get_datasets['pipeline_run_id'],
    )

    ### Register the model

    # The two parameters in `RegisterModel` that hold the metrics calculated by the `ClarifyCheckStep` and
    # `QualityCheckStep` are `model_metrics` and `drift_check_baselines`.

    # `drift_check_baselines` - these are the baseline files that will be used for drift checks in
    # `QualityCheckStep` or `ClarifyCheckStep` and model monitoring jobs that are set up on endpoints hosting this model.
    # `model_metrics` - these should be the latest baslines calculated in the pipeline run. This can be set
    # using the step property `CalculatedBaseline`

    # The intention behind these parameters is to give users a way to configure the baselines associated with
    # a model so they can be used in drift checks or model monitoring jobs. Each time a pipeline is executed, users can
    # choose to update the `drift_check_baselines` with newly calculated baselines. The `model_metrics` can be used to
    # register the newly calculated baslines or any other metrics associated with the model.

    # Every time a baseline is calculated, it is not necessary that the baselines used for drift checks are updated to
    # the newly calculated baselines. In some cases, users may retain an older version of the baseline file to be used
    # for drift checks and not register new baselines that are calculated in the Pipeline run.

    # register model step
    step_register = step(
        register,
        role=role,
        instance_type=process_instance_type_param,
        name="register",
        keep_alive_period_in_seconds=3600,
    )(
        training_job_name=step_train.properties.TrainingJobName,
        model_package_group_name=model_package_group_name_param,
        model_approval_status=model_approval_status_param,
        evaluation_result=step_evaluate['evaluation_result'],
        output_s3_prefix=output_s3_url,
        tracking_server_arn=tracking_server_arn_param,
        experiment_name=step_get_datasets['experiment_name'],
        pipeline_run_id=step_get_datasets['pipeline_run_id'],
        model_data_quality_check_statistics=data_quality_check_step.properties.CalculatedBaselineStatistics,
        model_data_quality_check_contraints=data_quality_check_step.properties.CalculatedBaselineConstraints,
        model_data_bias_check_constraints=data_bias_check_step.properties.CalculatedBaselineConstraints,
        model_quality_check_statistics=model_quality_check_step.properties.CalculatedBaselineStatistics,
        model_quality_check_constraints=model_quality_check_step.properties.CalculatedBaselineConstraints,
        model_bias_check_constraints=model_bias_check_step.properties.CalculatedBaselineConstraints,
        model_explainability_contraints=model_explainability_check_step.properties.CalculatedBaselineConstraints,
        drift_data_quality_check_statistics=data_quality_check_step.properties.BaselineUsedForDriftCheckStatistics,
        drift_data_quality_check_contraints=data_quality_check_step.properties.BaselineUsedForDriftCheckConstraints,
        drift_data_bias_check_contraints=data_bias_check_step.properties.BaselineUsedForDriftCheckConstraints,
        drift_model_bias_check_config=model_bias_check_config.monitoring_analysis_config_uri,
        drift_model_quality_check_statistics=model_quality_check_step.properties.BaselineUsedForDriftCheckStatistics,
        drift_model_quality_check_constraints=model_quality_check_step.properties.BaselineUsedForDriftCheckConstraints,
        drift_model_bias_check_constraints=model_bias_check_step.properties.BaselineUsedForDriftCheckConstraints,
        drift_model_explainability_check_constraints=model_explainability_check_step.properties.BaselineUsedForDriftCheckConstraints,
        drift_model_explainability_check_config=model_explainability_check_config.monitoring_analysis_config_uri,
    )

    # fail the pipeline execution step
    step_fail = FailStep(
        name="fail",
        error_message=Join(on=" ", values=["Execution failed due to AUC Score < ", test_score_threshold_param]),
    )

    # condition to check in the condition step
    condition_gte = ConditionGreaterThanOrEqualTo(
            left=step_evaluate['evaluation_result']['classification_metrics']['auc_score']['value'],
            right=test_score_threshold_param,
    )

    # conditional register step
    step_conditional_register = ConditionStep(
        name="check-metrics",
        conditions=[condition_gte],
        if_steps=[step_register],
        else_steps=[step_fail],
    )

    # Create a pipeline object
    pipeline = Pipeline(
        name=f"{pipeline_name}",
        parameters=[
            input_s3_url_param,
            process_instance_type_param,
            train_instance_type_param,
            model_approval_status_param,
            test_score_threshold_param,
            model_package_group_name_param,
            tracking_server_arn_param,

            skip_check_data_quality,
            register_new_baseline_data_quality,
            supplied_baseline_statistics_data_quality,
            supplied_baseline_constraints_data_quality,

            skip_check_data_bias,
            register_new_baseline_data_bias,
            supplied_baseline_constraints_data_bias,

            skip_check_model_quality,
            register_new_baseline_model_quality,
            supplied_baseline_statistics_model_quality,
            supplied_baseline_constraints_model_quality,

            skip_check_model_bias,
            register_new_baseline_model_bias,
            supplied_baseline_constraints_model_bias,

            skip_check_model_explainability,
            register_new_baseline_model_explainability,
            supplied_baseline_constraints_model_explainability
        ],
        steps=[step_conditional_register],
        pipeline_definition_config=PipelineDefinitionConfig(use_custom_job_prefix=True)
    )

    return pipeline
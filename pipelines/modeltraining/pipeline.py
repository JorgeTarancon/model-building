"""Example workflow pipeline script for abalone pipeline.
                                                                                 . -ModelStep
                                                                                .
    Process-> DataQualityCheck/DataBiasCheck -> Train -> Evaluate -> Condition .
                                                  |                              .
                                                  |                                . -(stop)
                                                  |
                                                   -> CreateModel-> ModelBiasCheck/ModelExplainabilityCheck
                                                           |
                                                           |
                                                            -> BatchTransform -> ModelQualityCheck

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput, CreateModelInput, TransformInput
from sagemaker.model import Model
from sagemaker.transformer import Transformer

from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
    FileSource
)
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterBoolean,
    ParameterInteger,
    ParameterString,
    ParameterFloat,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    CreateModelStep,
    TransformStep,
    CacheConfig
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.clarify_check_step import (
    DataBiasCheckConfig,
    ClarifyCheckStep,
    ModelBiasCheckConfig,
    ModelPredictedLabelConfig,
    ModelExplainabilityCheckConfig,
    SHAPConfig
)
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    ModelQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.model_monitor import DatasetFormat, model_monitoring
from sagemaker.clarify import (
    BiasConfig,
    DataConfig,
    ModelConfig
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """

     return boto3.Session(region_name=region).client("sagemaker")


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


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
        sm_client = get_sagemaker_client(region)
        project_arn = sm_client.describe_project(ProjectName=sagemaker_project_name)['ProjectArn']
        project_tags = sm_client.list_tags(ResourceArn=project_arn)['Tags']
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
    input_s3_url=None,
    model_package_group_name=None,
    pipeline_name_prefix="training-pipeline",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
    test_score_threshold=None,
    tracking_server_arn=None
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        bucket_name: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    ### PIPELINE PREPARATION ###
    if input_s3_url is None:
        print("input_s3_url must be provided. Exiting...")
        return None

    #sagemaker_session = get_session(region, bucket_name)
    #default_bucket = sagemaker_session.default_bucket()
    #if role is None:
        #role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, bucket_name)
    sm = pipeline_session.sagemaker_client

    if role is None:
        role = sagemaker.session.get_execution_role(pipeline_session)

    pipeline_name = f"{pipeline_name_prefix}-{sagemaker_project_id}"
    experiment_name = pipeline_name

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

    # S3 url for the input dataset
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=input_s3_url if input_s3_url else None,
    )
    
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)

    # Set processing instance type
    process_instance_type_param = ParameterString(
        name="ProcessingInstanceType",
        default_value=processing_instance_type,
    )

    # Set training instance type
    train_instance_type_param = ParameterString(
        name="TrainingInstanceType",
        default_value=training_instance_type,
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
    ### PIPELINE PREPARATION ###

    ### PREPROCESSING ###
    # processing step for feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=process_instance_type_param,
        instance_count=processing_instance_count,
        base_job_name=f"{pipeline_name_prefix}/preprocess",
        sagemaker_session=pipeline_session,
        role=role,
    )

    step_args = sklearn_processor.run(
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        arguments=["--input-data", input_data,
                   "--tracking-server-arn", tracking_server_arn_param,
                   "--experiment-name", experiment_name,
                   "--output-s3-prefix", f"s3://{bucket_name}/{pipeline_name_prefix}/output",],
    )

    step_process = ProcessingStep(
        name="Preprocess",
        step_args=step_args,
    )
    ### PREPROCESSING ###

    ### PIPELINE DEFINITION ###
    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            process_instance_type_param,
            processing_instance_count,
            train_instance_type_param,
            model_approval_status_param,
            input_data
        ],
        steps=[step_process],
        pipeline_definition_config=PipelineDefinitionConfig(use_custom_job_prefix=True)
    )
    ### PIPELINE DEFINITION ###

    return pipeline

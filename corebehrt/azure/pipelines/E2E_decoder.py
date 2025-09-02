"""
E2E_decoder pipeline implementation.
"""

from corebehrt.azure.pipelines.base import PipelineMeta, PipelineArg

E2E_DECODER = PipelineMeta(
    name="E2E_DECODER",
    help="Run the end-to-end pipeline with held out data.",
    inputs=[
        PipelineArg(name="data", help="Path to the raw input data.", required=True),
    ],
)


def create(component: callable):
    """
    Define the E2E_decoder full pipeline.

    Param component(job_type, name=None) is a constructor for components
    which takes arguments job_type (type of job) and optional argument
    name (name of component if different from type of job).
    """
    from azure.ai.ml import dsl, Input

    @dsl.pipeline(name="E2E_pipeline_decoder", description="Full E2E CoreBEHRT pipeline with decoder")
    def pipeline(data: Input) -> dict:
        create_data = component(
            "create_data",
        )(data=data)

        create_outcomes = component(
            "create_outcomes",
        )(
            data=data,
            features=create_data.outputs.features,
        )

        prepare_train = component(
            "prepare_training_data",
            name="prepare_train_decoder",
        )(
            features=create_data.outputs.features,
            tokenized=create_data.outputs.tokenized,
        )

        train_decoder = component("train_decoder")(
            prepared_data=prepare_train.outputs.prepared_data,
        )

        select_cohort_held_out = component(
            "select_cohort",
            name="select_held_out_cohort",
        )(
            features=create_data.outputs.features,
            tokenized=create_data.outputs.tokenized,
            outcomes=create_outcomes.outputs.outcomes,
        )

        prepare_held_out = component(
            "prepare_training_data",
            name="prepare_held_out_decoder",
        )(
            features=create_data.outputs.features,
            tokenized=create_data.outputs.tokenized,
            cohort=select_cohort_held_out.outputs.cohort,
            outcomes=create_outcomes.outputs.outcomes,
        )

        evaluate_decoder = component(
            "evaluate_decoder",
        )(
            model=train_decoder.outputs.model,
            test_data_dir=prepare_held_out.outputs.prepared_data,
        )

        return {
            "model": train_decoder.outputs.model,
            "predictions": evaluate_decoder.outputs.predictions,
        }

    return pipeline
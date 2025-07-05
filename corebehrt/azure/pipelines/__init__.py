from corebehrt.azure.pipelines.E2E import E2E
from corebehrt.azure.pipelines.E2E_full import E2E_full
from corebehrt.azure.pipelines.FINETUNE import FINETUNE
from corebehrt.azure.pipelines.E2E_XGB import E2E_XGB
from corebehrt.azure.pipelines.E2E_decoder import E2E_DECODER


PIPELINE_REGISTRY = [E2E, E2E_full, FINETUNE, E2E_DECODER, E2E_XGB]

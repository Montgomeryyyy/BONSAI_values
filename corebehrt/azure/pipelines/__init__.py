from corebehrt.azure.pipelines.E2E import E2E
from corebehrt.azure.pipelines.FINETUNE import FINETUNE
from corebehrt.azure.pipelines.E2E_decoder import E2E_DECODER

PIPELINE_REGISTRY = [E2E, FINETUNE, E2E_DECODER]

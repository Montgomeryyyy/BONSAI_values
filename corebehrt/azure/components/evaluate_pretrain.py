from corebehrt.azure.util import job

INPUTS = {
    "test_data_dir": {"type": "uri_folder"},
    "model": {"type": "uri_folder"},
}
OUTPUTS = {"embeddings": {"type": "uri_folder"}}

if __name__ == "__main__":
    from corebehrt.main import evaluate_pretrain

    job.run_main("evaluate_pretrain", evaluate_pretrain.main_evaluate_pretrain, INPUTS, OUTPUTS)

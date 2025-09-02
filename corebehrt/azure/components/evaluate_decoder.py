from corebehrt.azure.util import job

INPUTS = {
    "test_data_dir": {"type": "uri_folder"},
    "model": {"type": "uri_folder"},
}
OUTPUTS = {"predictions": {"type": "uri_folder"}}

if __name__ == "__main__":
    from corebehrt.main import evaluate_decoder

    job.run_main("evaluate_decoder", evaluate_decoder.main_evaluate, INPUTS, OUTPUTS)

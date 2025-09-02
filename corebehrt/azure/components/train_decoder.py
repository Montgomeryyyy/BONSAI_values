from corebehrt.azure.util import job

INPUTS = {
    "prepared_data": {"type": "uri_folder"},
    "restart_model": {
        "type": "uri_folder",
        "optional": True,
    },
}
OUTPUTS = {"model": {"type": "uri_folder"}}


if __name__ == "__main__":
    from corebehrt.main import train_decoder

    job.run_main("train_decoder", train_decoder.main_train, INPUTS, OUTPUTS)

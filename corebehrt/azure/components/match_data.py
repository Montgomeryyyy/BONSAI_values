from corebehrt.azure.util import job

INPUTS = {
    "prepared_data": {"type": "uri_folder"},
    "reference_data": {"type": "uri_folder"},
}
OUTPUTS = {"matched_data": {"type": "uri_folder"}}


if __name__ == "__main__":
    from corebehrt.main import match_data

    job.run_main("match_data", match_data.main_match_data, INPUTS, OUTPUTS)

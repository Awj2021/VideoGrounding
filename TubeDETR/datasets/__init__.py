from .vidstg import build as build_vidstg
from .hcstvg import build as build_hcstvg
from .chaos import build as build_chaos


def build_dataset(dataset_file: str, image_set: str, args, logger):
    if dataset_file == "vidstg":
        return build_vidstg(image_set, args)
    if dataset_file == "hcstvg":
        return build_hcstvg(image_set, args)
    if dataset_file == "chaos":
        return build_chaos(image_set, args, logger)
    raise ValueError(f"dataset {dataset_file} not supported")

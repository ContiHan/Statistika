from copy import deepcopy


_CURRENT_DATASET_CONFIG = None
_CURRENT_TARGET_DIAGNOSTICS = None


def set_current_dataset_config(config):
    global _CURRENT_DATASET_CONFIG
    _CURRENT_DATASET_CONFIG = deepcopy(config) if config is not None else None


def get_current_dataset_config():
    return deepcopy(_CURRENT_DATASET_CONFIG)


def set_current_target_diagnostics(diagnostics):
    global _CURRENT_TARGET_DIAGNOSTICS
    _CURRENT_TARGET_DIAGNOSTICS = (
        deepcopy(diagnostics) if diagnostics is not None else None
    )


def get_current_target_diagnostics():
    return deepcopy(_CURRENT_TARGET_DIAGNOSTICS)


def clear_runtime_context():
    global _CURRENT_DATASET_CONFIG, _CURRENT_TARGET_DIAGNOSTICS
    _CURRENT_DATASET_CONFIG = None
    _CURRENT_TARGET_DIAGNOSTICS = None

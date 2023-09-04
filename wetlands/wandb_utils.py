import os

import wandb
from dotenv import load_dotenv


def get_run_name():
    load_dotenv()

    run_name = os.getenv("RUN_NAME") if wandb.run is None else wandb.run.name

    return run_name


def get_model_path():
    load_dotenv()

    model_dir = os.getenv('MODELS_DIR')
    run_name = get_run_name()
    model_file = f'{run_name}_best_model.pth'
    model_path = os.path.join(model_dir, model_file)

    return model_path

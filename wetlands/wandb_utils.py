import os

import wandb
from dotenv import load_dotenv


def get_model_path():
    load_dotenv()

    model_dir = os.getenv('MODELS_DIR')
    model_name = os.getenv("MODEL_NAME")
    run_name = os.getenv("RUN_NAME") if wandb.run is None else wandb.run.name
    model_file = f'{run_name}_{model_name}.pth'
    model_path = os.path.join(model_dir, model_file)

    return model_path

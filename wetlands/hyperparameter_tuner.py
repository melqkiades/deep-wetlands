import time

import wandb

from wetlands import train_model


def create_sweep():

    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'val_iou',
            'goal': 'maximize'
        },
        'parameters': {
            'LEARNING_RATE': {
                'values': [0.00001, 0.00005, 0.0001, 0.0005, 0.001]
            },
            'UNET_INIT_DIM': {
                'values': [16, 32, 64, 128]
            },
            'UNET_BLOCKS': {
                'values': [2, 3, 4, 5]
            },
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_config, entity='deep-wetlands', project="sweeps")

    print('Sweep ID:', sweep_id)

    return sweep_id


def run_sweep(sweep_id):
    wandb.agent(sweep_id, function=train_model.full_cycle, count=100)


def main():
    sweep_id = create_sweep()
    run_sweep(sweep_id)


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))

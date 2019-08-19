import argparse
import yaml
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from test_tube import Experiment

from pl_model import PLModel


parser = argparse.ArgumentParser()
parser.add_argument('config_path')
args = parser.parse_args()
config_path = Path(args.config_path)
config = yaml.load(open(config_path))

checkpoint_callback = ModelCheckpoint(filepath='../model/test',
                                      save_best_only=True,
                                      verbose=True,
                                      monitor='avg_val_loss',
                                      mode='min')

model = PLModel(config)

exp = Experiment(save_dir='../logs/test')

trainer = Trainer(gpus=[0],
                  progress_bar=True,
                  experiment=exp,
                  checkpoint_callback=checkpoint_callback)

trainer.fit(model)

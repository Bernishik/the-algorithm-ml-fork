import datetime
import os
from typing import Callable, List, Optional, Tuple
import tensorflow as tf
import tree

import tml.common.checkpointing.snapshot as snapshot_lib
from tml.common.device import setup_and_get_device
from tml.core import config as tml_config_mod
import tml.core.custom_training_loop as ctl
from tml.core import debug_training_loop
from tml.core import losses
from tml.core.loss_type import LossType
from tml.model import maybe_shard_model
from tml.core.train_pipeline import TrainPipelineSparseDist
from tml.ml_logging.torch_logging import logging 
import tml.core.metrics as core_metrics
from tml.projects.twhin.metrics import create_metrics


import tml.projects.home.recap.data.dataset as ds
import tml.projects.home.recap.config as recap_config_mod
import tml.projects.home.recap.optimizer as optimizer_mod


# from tml.projects.home.recap import feature
import tml.projects.home.recap.model as model_mod
import torchmetrics as tm
import torch
import torch.distributed as dist
from torchrec.distributed.model_parallel import DistributedModelParallel
import torch.nn.functional as F

from absl import app, flags, logging


flags.DEFINE_string("config_path", './projects/home/recap/config/local_prod.yaml', "Path to hyperparameters for model.")
flags.DEFINE_bool("debug_loop", False, "Run with debug loop (slow)")
flags.DEFINE_boolean('standalone', False, 'Whether to run in standalone mode')
flags.DEFINE_integer('nnodes', 1, 'Number of nodes')
flags.DEFINE_integer('nproc_per_node', 1, 'Number of processes per node')

FLAGS = flags.FLAGS


def run(unused_argv: str, data_service_dispatcher: Optional[str] = None):
 os.environ['TML_BASE'] = os.getcwd()
 os.environ['LOCAL_RANK'] = "0"
 os.environ['WORLD_SIZE'] = "1"
 os.environ['LOCAL_WORLD_SIZE'] ="1"
 os.environ['MASTER_ADDR'] ="localhost"
 os.environ['MASTER_PORT'] = "29501"
 os.environ['RANK'] = "0"

 device = setup_and_get_device()
 # Always enable tensorfloat on supported devices.
 torch.backends.cuda.matmul.allow_tf32 = True
 torch.backends.cudnn.allow_tf32 = True

 config = tml_config_mod.load_config_from_yaml(recap_config_mod.RecapConfig, FLAGS.config_path)
 data_service_dispatcher = None

 train_dataset = ds.RecapDataset(
   data_config=config.train_data,
   dataset_service=data_service_dispatcher,
   mode=recap_config_mod.JobMode.TRAIN,
   compression=config.train_data.dataset_service_compression,
   vocab_mapper=None,
   repeat=True,
 )
 
 
 
 torch_element_spec = train_dataset.torch_element_spec
 
 config = tml_config_mod.load_config_from_yaml(recap_config_mod.RecapConfig, FLAGS.config_path)
 loss_fn = losses.build_multi_task_loss(
     loss_type=LossType.BCE_WITH_LOGITS,
     tasks=list(config.model.tasks.keys()),
     pos_weights=[task.pos_weight for task in config.model.tasks.values()],
   )
 
 model = model_mod.create_ranking_model(
     data_spec=torch_element_spec[0],
     config=config,
     loss_fn=loss_fn,
     device=device,
   )
 
 model = maybe_shard_model(model, device)
 
 optimizer, scheduler = optimizer_mod.build_optimizer(model, config.optimizer, None)
 
 optimizer.init_state()

 save_state = {
    "model": model,
    "optimizer": optimizer,
    "scaler":  torch.cuda.amp.GradScaler(enabled=False),
  }
 
 save_dir = '/home/bernish/tmp/runs/recap_local_debug'


 checkpoint_handler = snapshot_lib.Snapshot(
    save_dir=save_dir,
    state=save_state
  )
 
 chosen_checkpoint = snapshot_lib.get_checkpoint(save_dir=save_dir, missing_ok=True)
 
 checkpoint_handler.restore(chosen_checkpoint)

 train_iterator = iter(train_dataset.to_dataloader())
 
 eval_pipeline = TrainPipelineSparseDist(model, optimizer, device) 
 
 
 labels = list(config.model.tasks.keys())
 eval_steps = 1
 print('#' * 100)
 for _ in range(eval_steps):
    new_iterator =ctl.get_new_iterator(train_iterator)
    step_fn = ctl._get_step_fn(eval_pipeline, new_iterator, training=False)
    
    results = step_fn()
    probabilities = results['probabilities'][0].cpu()
    for key,val in enumerate(probabilities):
        print(labels[key] + ': ' + str(val.item()))
    print('#' * 100)
        





if __name__ == "__main__":
  app.run(run)
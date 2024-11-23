import datetime
import os
from typing import Callable, List, Optional, Tuple
import tensorflow as tf
import tree
import functools
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

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
from tml.projects.home.recap.data import preprocessors


import tml.projects.home.recap.data.dataset as ds
import tml.projects.home.recap.config as recap_config_mod
import tml.projects.home.recap.optimizer as optimizer_mod
from tml.projects.home.recap.data.tfe_parsing import get_seg_dense_parse_fn


# from tml.projects.home.recap import feature
import tml.projects.home.recap.model as model_mod
import torchmetrics as tm
import torch
import torch.distributed as dist
from torchrec.distributed.model_parallel import DistributedModelParallel
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import scipy.stats as stats

from absl import app, flags, logging


flags.DEFINE_string("config_path", './projects/home/recap/config/local_prod.yaml', "Path to hyperparameters for model.")
flags.DEFINE_bool("debug_loop", False, "Run with debug loop (slow)")
flags.DEFINE_boolean('standalone', False, 'Whether to run in standalone mode')
flags.DEFINE_integer('nnodes', 1, 'Number of nodes')
flags.DEFINE_integer('nproc_per_node', 1, 'Number of processes per node')
flags.DEFINE_integer("display_formula", 1, "Display Formula")

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

 data_config= config.test_data

 mode = recap_config_mod.JobMode.EVALUATE
 test_dataset = ds.RecapDataset(
   data_config=data_config,
   dataset_service=data_service_dispatcher,
   mode=mode,
   compression=config.train_data.dataset_service_compression,
   vocab_mapper=None,
   repeat=False,
 )
 _parse_fn = get_seg_dense_parse_fn(data_config)

 if data_config.preprocess:
      preprocessor = preprocessors.build_preprocess(data_config.preprocess, mode=mode)

 should_add_weights = any(
        [
          task_cfg.pos_downsampling_rate != 1.0 or task_cfg.neg_downsampling_rate != 1.0
          for task_cfg in data_config.tasks.values()
        ]
      )
 output_map_fn = ds._map_output_for_train_eval
 _output_map_fn = functools.partial(
      output_map_fn,
      tasks=data_config.tasks,
      preprocessor=preprocessor,
      add_weights=should_add_weights,
    )


 
  # Combine functions into one map call to reduce overhead.
 map_fn = functools.partial(
      ds._chain,
      f1=_parse_fn,
      f2=_output_map_fn,
    )
 
 glob = data_config.inputs
 filenames = sorted(tf.io.gfile.glob(glob))
 filenames_ds = (
      tf.data.Dataset.from_tensor_slices(filenames).shuffle(len(filenames))
      # Because of drop_remainder, if our dataset does not fill
      # up a batch, it will emit nothing without this repeat.
      # .repeat(0)
    )
 
 first_filename = next(iter(filenames_ds))
 file_ds = tf.data.TFRecordDataset([first_filename], compression_type="GZIP")
 
 file_ds = file_ds.batch(batch_size=data_config.global_batch_size, drop_remainder=True).map(
        map_fn,
        num_parallel_calls=data_config.map_num_parallel_calls
        or tf.data.experimental.AUTOTUNE,
      )

 def function_mapper(iterable, func):
    for item in iterable:
        yield func(item)

 file_ds_iterable = function_mapper(file_ds,ds.to_batch)

 torch_element_spec = test_dataset.torch_element_spec
 
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




 eval_pipeline = TrainPipelineSparseDist(model, optimizer, device) 

 scored_tweets_model_weight_fav= 0.5
 scored_tweets_model_weight_retweet= 1.0
 scored_tweets_model_weight_reply= 13.5
 scored_tweets_model_weight_good_profile_click= 12.0
 scored_tweets_model_weight_video_playback50= 0.005
 scored_tweets_model_weight_reply_engaged_by_author= 75.0
 scored_tweets_model_weight_good_click= 11.0
 scored_tweets_model_weight_good_click_v2= 10.0
 scored_tweets_model_weight_negative_feedback_v2= -74.0
 scored_tweets_model_weight_report= -369.0
 
 scored_weight = [
  scored_tweets_model_weight_fav,
  scored_tweets_model_weight_good_click,
  scored_tweets_model_weight_good_click_v2,
  scored_tweets_model_weight_negative_feedback_v2,
  scored_tweets_model_weight_good_profile_click,
  scored_tweets_model_weight_reply,
  scored_tweets_model_weight_reply_engaged_by_author,
  scored_tweets_model_weight_report,
  scored_tweets_model_weight_retweet,
  scored_tweets_model_weight_video_playback50,
  ]

 
 labels = list(config.model.tasks.keys())
 take_first_n_candidates = 1000
 half = take_first_n_candidates / 2
 print('#' * 100)

 res = []
 for _ in range(take_first_n_candidates):
    
    new_iterator =iter(file_ds_iterable)

    eval_pipeline._model.eval()
    outputs = eval_pipeline.progress(new_iterator)
  
    results = tree.map_structure(lambda elem: elem.detach(), outputs)
    probabilities = results['probabilities'][0].cpu()

    score = 0
    
    for key,val in enumerate(probabilities):
        if(FLAGS.display_formula):
          print(labels[key] + ': ' + str(val.item()))
    
    print("Score = ",end='')
    for key,val in enumerate(probabilities):
        if(FLAGS.display_formula):
          if(key != 0):
            print(" + ",end='')

          print(f"({scored_weight[key]} * {val.item()})",end='')
        score += scored_weight[key] * val.item()
    print(f"= {score}")
    res.append([score, _ > half])
   
    
    print('#' * 100)

 sorted_res = sorted(res, key=lambda x: x[0], reverse=True)

 positive_scores = [item[0] for item in sorted_res if item[1]]  
 negative_scores = [item[0] for item in sorted_res if not item[1]] 
 matplotlib.use('TkAgg')
  # Візуалізація розподілу
 plt.figure(figsize=(14, 6))
 
 # Гістограма розподілу
 sns.histplot(data=positive_scores, color="blue", label="Positive", kde=True, stat="density", bins=30, alpha=0.6)
 sns.histplot(data=negative_scores, color="red", label="Negative", kde=True, stat="density", bins=30, alpha=0.6)
 plt.title("Розподіл скорів для позитивних і негативних параметрів")
 plt.xlabel("Скор")
 plt.ylabel("Щільність")
 plt.legend()
 plt.show()

 # t-тест для незалежних вибірок
 t_stat, p_value = stats.ttest_ind(positive_scores, negative_scores)
 
 print(f"t-статистика: {t_stat}")
 print(f"p-значення: {p_value}")
 
 # Інтерпретація результатів
 if p_value < 0.05:
     print("Є статистично значуща різниця між середніми значеннями груп.")
 else:
     print("Немає статистично значущої різниці між середніми значеннями груп.")

#  positives_first = 0
#  negatives_first = 0
#  for k,r in enumerate(sorted_res):
#     print(r[0], str(r[1]))
#     if (k < half):
#      if(r[1] == True):
#         positives_first = positives_first + 1
#      else:
#         negatives_first = negatives_first + 1

#  print(positives_first)
#  print(negatives_first)
    





if __name__ == "__main__":
  app.run(run)
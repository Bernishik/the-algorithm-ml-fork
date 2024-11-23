import os
import json
from absl import app, flags, logging
import tensorflow as tf
from typing import Dict

from tml.projects.home.recap.data import tfe_parsing
from tml.core import config as tml_config_mod
import tml.projects.home.recap.config as recap_config_mod

flags.DEFINE_string("config_path", None, "Path to hyperparameters for model.")
flags.DEFINE_integer("n_examples", 1000 + 2, "Numer of examples to generate.")
flags.DEFINE_integer("first_n_positive", 500 + 2, "Numer of positive examples to generate.")
flags.DEFINE_integer("first_n_negative", 500, "Numer of negative examples to generate.")


FLAGS = flags.FLAGS


def _generate_random_example(
  tf_example_schema: Dict[str, tf.io.FixedLenFeature]
) -> Dict[str, tf.Tensor]:
  example = {}
  for feature_name, feature_spec in tf_example_schema.items():
    dtype = feature_spec.dtype
    if (dtype == tf.int64) or (dtype == tf.int32):
      x = tf.experimental.numpy.random.randint(0, high=10, size=feature_spec.shape, dtype=dtype)
    elif (dtype == tf.float32) or (dtype == tf.float64):
      x = tf.random.uniform(shape=[feature_spec.shape], dtype=dtype)
    else:
      raise NotImplementedError(f"Unknown type {dtype}")

    example[feature_name] = x

  return example

def _generate_random_example_v2(
    tf_example_schema: Dict[str, tf.io.FixedLenFeature],
    mode: str = "random",  # "random", "negative_zero", "positive_zero"
    embeding: str = "random",  # "random", "zero"
    positive_features: list[str] = None,
    negative_features: list[str] = None,
) -> Dict[str, tf.Tensor]:
    """
    Генерує приклад з урахуванням вибраного режиму.

    Args:
        tf_example_schema (Dict[str, tf.io.FixedLenFeature]): Схема прикладу.
        mode (str): Режим генерації ("random", "negative_zero", "positive_zero").
        negative_features (List[str]): Список характеристик, що погіршують скор.
        positive_features (List[str]): Список характеристик, що покращують скор.

    Returns:
        Dict[str, tf.Tensor]: Згенерований приклад.
    """
    example = {}

    for feature_name, feature_spec in tf_example_schema.items():
        dtype = feature_spec.dtype
        shape = feature_spec.shape

        if mode == "negative_zero" and feature_name in negative_features:
            # Характеристики, що погіршують скор, дорівнюють нулю
            x = tf.zeros(shape=shape, dtype=dtype)
        elif mode == "positive_zero" and feature_name in positive_features :
            # Характеристики, що покращують скор, дорівнюють нулю
            x = tf.zeros(shape=shape, dtype=dtype)
        else:
            if(embeding == "random"):
              if (dtype == tf.int64) or (dtype == tf.int32):
                x = tf.experimental.numpy.random.randint(0, high=10, size=shape, dtype=dtype)
              elif (dtype == tf.float32) or (dtype == tf.float64):
                x = tf.random.uniform(shape=[shape], dtype=dtype)
              else:
                raise NotImplementedError(f"Unknown type {dtype}")
            else: 
              if (dtype == tf.int64) or (dtype == tf.int32):
                 x = tf.zeros(shape=shape, dtype=dtype)
              elif (dtype == tf.float32) or (dtype == tf.float64):
                 x = tf.zeros(shape=[shape], dtype=dtype)
              else:
                raise NotImplementedError(f"Unknown type {dtype}")

        example[feature_name] = x

    return example


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _serialize_example(x: Dict[str, tf.Tensor]) -> bytes:
  feature = {}
  serializers = {tf.float32: _float_feature, tf.int64: _int64_feature}
  for feature_name, tensor in x.items():
    feature[feature_name] = serializers[tensor.dtype](tensor)

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


def generate_data(data_path: str, config: recap_config_mod.RecapConfig):
  with tf.io.gfile.GFile(config.train_data.seg_dense_schema.schema_path, "r") as f:
    seg_dense_schema = json.load(f)["schema"]

  tf_example_schema = tfe_parsing.create_tf_example_schema(
    config.train_data,
    seg_dense_schema,
  )

  record_filename = os.path.join(data_path, "random.tfrecord.gz")

  negative_features = [
    "recap.engagement.is_negative_feedback_v2",
    "recap.engagement.is_report_tweet_clicked",
  ]

  positive_features = [
    "recap.engagement.is_retweeted",
    "recap.engagement.is_favorited",
    "recap.engagement.is_shared",
    "recap.engagement.is_good_clicked_convo_desc_favorited_or_replied",
    "recap.engagement.is_tweet_detail_dwelled_15_sec"
]

  with tf.io.TFRecordWriter(record_filename, "GZIP") as writer:
    for n in range(FLAGS.n_examples):
      print(f'GENERATION IN PROGRESS WAIT {n+1}/{FLAGS.n_examples}')
      random_example = None
      if(n < FLAGS.first_n_positive):
        random_example = _generate_random_example_v2(tf_example_schema, 'negative_zero', "random", positive_features, negative_features)
      elif n < FLAGS.first_n_positive + FLAGS.first_n_negative:
        random_example = _generate_random_example_v2(tf_example_schema,'positive_zero', "ramdom", positive_features,negative_features)
      else:
        random_example = _generate_random_example_v2(tf_example_schema, positive_features,negative_features)
      serialized_example = _serialize_example(random_example)
      writer.write(serialized_example)


def _generate_data_main(unused_argv):
  config = tml_config_mod.load_config_from_yaml(recap_config_mod.RecapConfig, FLAGS.config_path)

  # Find the path where to put the data
  data_path = os.path.dirname(config.train_data.inputs)
  logging.info("Putting random data in %s", data_path)

  generate_data(data_path, config)
  logging.info("Generated random data in %s", data_path)



if __name__ == "__main__":
  app.run(_generate_data_main)

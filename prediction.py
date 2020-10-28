from tapas.utils import tf_example_utils
from tapas.protos import interaction_pb2
from tapas.utils import number_annotation_utils
from tapas.scripts import prediction_utils
from tapas.utils import task_utils
import subprocess
import tensorflow.compat.v1 as tf
import os 
import shutil
import csv
import pandas as pd
import IPython
import csv

max_seq_length = 512
vocab_file = "tapas_sqa_base/vocab.txt"
config = tf_example_utils.ClassifierConversionConfig(
    vocab_file=vocab_file,
    max_seq_length=max_seq_length,
    max_column_id=max_seq_length,
    max_row_id=max_seq_length,
    strip_column_names=False,
    add_aggregation_candidates=False,
)
converter = tf_example_utils.ToClassifierTensorflowExample(config)

def convert_interactions_to_examples(tables_and_queries):
  """Calls Tapas converter to convert interaction to example."""
  for idx, (table, queries) in enumerate(tables_and_queries):
    interaction = interaction_pb2.Interaction()
    for position, query in enumerate(queries):
      question = interaction.questions.add()
      question.original_text = query
      question.id = f"{idx}-0_{position}"
    for header in table[0]:
      interaction.table.columns.add().text = header
    for line in table[1:]:
      row = interaction.table.rows.add()
      for cell in line:
        row.cells.add().text = cell
    number_annotation_utils.add_numeric_values(interaction)
    for i in range(len(interaction.questions)):
      try:
        yield converter.convert(interaction, i)
      except ValueError as e:
        print(f"Can't convert interaction: {interaction.id} error: {e}")
        
def write_tf_example(filename, examples):
  with tf.io.TFRecordWriter(filename) as writer:
    for example in examples:
      writer.write(example.SerializeToString())

def predict(table_data, queries):
  examples = convert_interactions_to_examples([(table_data, queries)])
  examples
  write_tf_example("results/sqa/tf_examples/test.tfrecord", examples)
  write_tf_example("results/sqa/tf_examples/random-split-1-dev.tfrecord", [])
  subprocess.call(["bash", "get_checkpoint.sh",str(len(queries))])
  results_path = "results/sqa/model/test_sequence.tsv"
  all_coordinates = []
  with open(results_path) as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')
    for row in reader:
      coordinates = prediction_utils.parse_coordinates(row["answer_coordinates"])
      all_coordinates.append(coordinates)
      answers = ', '.join([table_data[row + 1][col] for row, col in coordinates])
      position = int(row['position'])
      print(">", queries[position])
      print(answers)
  return all_coordinates


file = []
with open("balancesheet.csv") as csvfile:
    reader = csv.reader(csvfile) # change contents to floats
    for row in reader: # each row is a list
        file.append(row)
with open('questions.txt') as querylist:
    queryarray = querylist.read().splitlines()

result=predict(file,queryarray)
print(result)
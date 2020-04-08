import os
from os.path import join as opjoin
import string
import random
import datetime
import json

MODEL_NAME = 'Model Name'
EPOCHS = 'Epochs'
LOSS = 'Loss'
LOSS_VALUE = 'Loss Value'
VAL_DS_LEN = 'Validation Dataset Length'
VAL_STEPS = 'Validation Steps'
GRID_SIZE = 'Grid Size'
CURRENT_TIME = 'Save Time'
RECALL = 'Recall'
SPEC = 'Specificity'
TRAIN_FINISH_TIMESTAMP = 'Training Finished at'
METRIC_EVAL_TIMESTAMP = 'Metrics Calculated at'
MODEL_FILE_NAME = 'Model Filename'
MODELS_JSON = 'models.json'


class TrainData:
    def __init__(self, model_name, epochs, loss, loss_value, val_ds_len, val_steps, grid_size, nn_models_dir,
                 train_finish_timestamp=None, metric_eval_timestamp=None, recall=None, spec=None, model_file_name=None):
        self.model_name = model_name
        self.epochs = epochs
        self.loss = loss
        self.loss_value = loss_value
        self.val_ds_len = val_ds_len
        self.val_steps = val_steps
        self.grid_size = grid_size
        self.nn_models_dir = nn_models_dir
        if train_finish_timestamp is None:
            self.train_finish_timestamp = datetime.datetime.now()
        else:
            self.train_finish_timestamp = train_finish_timestamp
        self.metric_eval_timestamp = metric_eval_timestamp
        self.recall = recall
        self.spec = spec
        self.models_json = opjoin(nn_models_dir, MODELS_JSON)
        self.model_file_name = model_file_name


    def save(self, model):
        random_string = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))
        model_file_name = '{}_{}_{}_{}.h5'.format(self.model_name, self.epochs, self.grid_size, random_string)
        self.model_file_name = model_file_name
        model.save(opjoin(self.nn_models_dir, model_file_name))

        if MODELS_JSON not in os.listdir(self.nn_models_dir):
            records = []
        else:
            with open(opjoin(self.nn_models_dir, MODELS_JSON), "r") as read_file:
                records = json.load(read_file)
        records.append(self.to_dict())
        with open(opjoin(self.nn_models_dir, MODELS_JSON), "w") as write_file:
            json.dump(records, write_file, indent=1)

    def add_metrics(self, recall=None, spec=None):
        self.recall = recall
        self.spec = spec
        self.metric_eval_timestamp = datetime.datetime.now()

    def to_dict(self):
        return {
            MODEL_NAME: self.model_name,
            EPOCHS: self.epochs,
            LOSS: self.loss,
            LOSS_VALUE: self.loss_value,
            VAL_DS_LEN: self.val_ds_len,
            VAL_STEPS: self.val_steps,
            GRID_SIZE: self.grid_size,
            RECALL: self.recall,
            SPEC: self.spec,
            TRAIN_FINISH_TIMESTAMP: str(self.train_finish_timestamp),
            METRIC_EVAL_TIMESTAMP: str(self.metric_eval_timestamp),
            MODEL_FILE_NAME: self.model_file_name
        }


def from_dict(d, nn_models_dir):
    return TrainData(d[MODEL_NAME], d[EPOCHS], d[LOSS], d[LOSS_VALUE], d[VAL_DS_LEN], d[VAL_STEPS], d[GRID_SIZE], nn_models_dir,
                     d[TRAIN_FINISH_TIMESTAMP], d[METRIC_EVAL_TIMESTAMP], d[RECALL], d[SPEC], d[MODEL_FILE_NAME])



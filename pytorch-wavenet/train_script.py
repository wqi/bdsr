import time
import torch

from model_logging import Logger
from audio_data import WavenetDataset
from data_loader import BDSRDataset
from scipy.io import wavfile
from wavenet_model import WaveNetModel, load_latest_model_from
from wavenet_training import WavenetTrainer, generate_audio


dtype = torch.FloatTensor
ltype = torch.LongTensor

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('use gpu')
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor

model = WaveNetModel(layers=11,
                     blocks=3,
                     dilation_channels=32,
                     residual_channels=32,
                     skip_channels=1024,
                     in_channels=1,
                     end_channels=512,
                     output_length=3070,
                     classes=1,
                     dtype=dtype,
                     bias=True)

model = load_latest_model_from('snapshots', use_cuda=True)
# model = torch.load('snapshots/some_model')

if use_cuda:
    print("move model to gpu")
    model.cuda()

print('model: ', model)
print('receptive field: ', model.receptive_field)
print('parameter count: ', model.parameter_count())

# data = WavenetDataset(dataset_file='train_samples/bach_chaconne/dataset.npz',
#                       item_length=model.receptive_field + model.output_length - 1,
#                       target_length=model.output_length,
#                       file_location='train_samples/bach_chaconne',
#                       test_stride=500)
data = BDSRDataset(lr_data_file='../data/music/music_train_lr.npy',
                   hr_data_file='../data/music/music_train_hr.npy',
                   item_length=model.receptive_field*2,
                   sample_rate=16000)
print('the dataset has ' + str(len(data)) + ' items')


def generate_and_log_samples(step):
    sample_length = 32000
    gen_model = load_latest_model_from('snapshots', use_cuda=False)
    print("start generating...")
    samples = generate_audio(gen_model,
                             length=sample_length,
                             temperatures=[0.5])
    tf_samples = tf.convert_to_tensor(samples, dtype=tf.float32)
    logger.audio_summary('temperature_0.5', tf_samples, step, sr=16000)

    samples = generate_audio(gen_model,
                             length=sample_length,
                             temperatures=[1.])
    tf_samples = tf.convert_to_tensor(samples, dtype=tf.float32)
    logger.audio_summary('temperature_1.0', tf_samples, step, sr=16000)
    print("audio clips generated")

logger = Logger(log_interval=20,
                validation_interval=400,
                generate_interval=1000)
'''
logger = TensorboardLogger(log_interval=40,
                           validation_interval=400,
                           log_dir="logs/chaconne_model")
'''

trainer = WavenetTrainer(model=model,
                         dataset=data,
                         lr=0.00001,
                         weight_decay=0.0,
                         snapshot_path='snapshots',
                         snapshot_name='diff_model',
                         snapshot_interval=100,
                         logger=logger,
                         dtype=dtype,
                         ltype=ltype)

print('start training...')
trainer.train(batch_size=5,
              epochs=200,
              continue_training_at_step=0)

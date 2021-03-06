import numpy as np
import scipy.io.wavfile as wav
import time
import torch
import torch.nn.functional as F
import wavenet_model as wm

from data_loader import BDSRDataset
from torch.autograd import Variable

dtype = torch.FloatTensor
ltype = torch.LongTensor

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('use gpu')
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor

model = wm.WaveNetModel(layers=10,
                        blocks=3,
                        dilation_channels=32,
                        residual_channels=32,
                        skip_channels=1024,
                        in_channels=1,
                        end_channels=512,
                        output_length=16,
                        classes=65536,
                        dtype=dtype,
                        bias=True)

model = wm.load_latest_model_from('snapshots')

data = BDSRDataset(lr_data_file='../data/music/music_test_lr.npy',
                   hr_data_file='../data/music/music_test_hr.npy',
                   item_length=18000,
                   sample_rate=16000)
dataloader = torch.utils.data.DataLoader(data,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=8,
                                         pin_memory=False)

print("")
tic = time.time()
for (lr, hr) in iter(dataloader):
    lr = Variable(lr.type(dtype))

    output = model(lr).squeeze(dim=1)
    output_length = output.size(1)
    output = output.view(output.size(0)*output.size(1), output.size(2))

    output = F.softmax(output, dim=1)
    _, output_values = output.data.topk(1)
    output_values = np.squeeze(output_values.cpu().numpy())

    hr = Variable(hr.type(ltype)) + (65536//2-1)
    hr = hr.squeeze(dim=1)[:, -output_length:].contiguous()
    hr = hr.view(hr.size(0)*hr.size(1))
    hr = np.squeeze(hr.cpu().data.numpy())

    lr = lr.squeeze(dim=1)[:, -output_length:].contiguous()
    lr = np.squeeze(lr.cpu().data.numpy()) * 256

    # Compute differences
    output_diff = output_values - hr
    lr_diff = lr - hr
    print("Mean BDSR Diff:", np.mean(np.abs(output_diff)))
    print("Mean Naive Diff:", np.mean(np.abs(lr_diff)))

    # Write WAV
    lr = lr - 32768
    hr = hr - 32768
    output_values = output_values - 32768
    print(hr)
    print(output_values)
    wav.write('out.wav', 16000, output_values.astype('int16'))
    wav.write('out_lr.wav', 16000, lr.astype('int16'))
    wav.write('out_hr.wav', 16000, hr.astype('int16'))

    break
toc = time.time()
print("Generation took", str((toc - tic) * 0.01), "seconds")

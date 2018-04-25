import numpy as np
import scipy.io.wavfile as wav
import torch

from data_loader import BDSRDataset

data = BDSRDataset(lr_data_file='../data/music/music_test_lr.npy',
                   hr_data_file='../data/music/music_test_hr.npy',
                   item_length=60000,
                   sample_rate=16000)
dataloader = torch.utils.data.DataLoader(data,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=8,
                                         pin_memory=False)

for (lr, hr) in iter(dataloader):
    hr = hr.squeeze(dim=1)
    hr = np.squeeze(hr.cpu().numpy())
    # Write WAV
    hr_rand = hr + np.random.randint(50, size=hr.size) - 5
    print(hr)
    print(hr_rand)
    print('----')
    wav.write('out.wav', 16000, hr_rand.astype('int16'))
    wav.write('out_hr.wav', 16000, hr.astype('int16'))

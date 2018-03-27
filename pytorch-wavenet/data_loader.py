import numpy as np
import torch
import torch.utils.data


class BDSRDataset(torch.utils.data.Dataset):
    def __init__(self, lr_data_file, hr_data_file, item_length, sample_rate=16000):
        self.lr_data_file = lr_data_file
        self.hr_data_file = hr_data_file
        self.item_length = item_length
        self.sample_rate = sample_rate

        self.lr_data = np.load(self.lr_data_file, mmap_mode='r')
        self.hr_data = np.load(self.hr_data_file, mmap_mode='r')
        assert self.lr_data.shape == self.hr_data.shape, "LR-HR dataset shapes don't match"

    def __len__(self):
        return self.hr_data.shape[0]

    def __getitem__(self, idx):
        lr_sample = self.lr_data[idx]
        hr_sample = self.hr_data[idx]

        # Sample LR and HR fragments from audio
        offset = np.random.randint(0, hr_sample.shape[0] - self.item_length)
        lr_fragment = lr_sample[offset:offset + self.item_length].tolist()
        hr_fragment = hr_sample[offset:offset + self.item_length].tolist()

        return lr_fragment, hr_fragment


# Test
dataset = BDSRDataset('../data/music/music_valid_lr.npy',
                      '../data/music/music_valid_hr.npy',
                      100)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=16,
                                         shuffle=True,
                                         num_workers=8,
                                         pin_memory=False)
for (lr, hr) in iter(dataloader):
    print(lr[0])
    print(hr[0])
    break

import numpy as np
import torch
import torch.utils.data


class BDSRDataset(torch.utils.data.Dataset):
    def __init__(self, lr_data_file, hr_data_file, item_length, sample_rate=16000, memmap=True):
        self.lr_data_file = lr_data_file
        self.hr_data_file = hr_data_file
        self.item_length = item_length
        self.sample_rate = sample_rate

        if memmap:
            self.lr_data = np.load(self.lr_data_file, mmap_mode='r')
            self.hr_data = np.load(self.hr_data_file, mmap_mode='r')
        else:
            self.lr_data = np.load(self.lr_data_file)
            self.hr_data = np.load(self.hr_data_file)
        assert self.lr_data.shape == self.hr_data.shape, "LR-HR dataset shapes don't match"

    def __len__(self):
        return self.hr_data.shape[0]

    def __getitem__(self, idx):
        lr_sample = self.lr_data[idx]
        hr_sample = self.hr_data[idx]

        # Sample LR and HR fragments from audio
        offset = np.random.randint(0, hr_sample.shape[0] - self.item_length)
        lr_fragment = lr_sample[offset:offset + self.item_length].astype(int)
        hr_fragment = hr_sample[offset:offset + self.item_length].astype(int)
        lr_tensor = torch.from_numpy(lr_fragment).unsqueeze(0)
        hr_tensor = torch.from_numpy(hr_fragment).unsqueeze(0)

        return lr_tensor, hr_tensor


# Test
# dataset = BDSRDataset('../data/music/music_valid_lr.npy',
#                       '../data/music/music_valid_hr.npy',
#                       100)
# dataloader = torch.utils.data.DataLoader(dataset,
#                                          batch_size=16,
#                                          shuffle=True,
#                                          num_workers=8,
#                                          pin_memory=False)
# for (lr, hr) in iter(dataloader):
#     print(lr)
#     print(hr)
#     break

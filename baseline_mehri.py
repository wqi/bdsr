import mehri_data_prep as data
import metrics
import numpy as np
import scipy.io.wavfile as wav

lr_samples = np.load('data/music/music_test_lr.npy')
hr_samples = np.load('data/music/music_test_hr.npy')


def naive_upscaling():
    total_psnr = 0

    for idx, lr_sample in enumerate(lr_samples):
        hr_sample = hr_samples[idx]
        psnr = metrics.bd_psnr_naive(lr_sample, hr_sample)
        total_psnr += psnr

    avg_psnr = total_psnr / len(lr_samples)
    return avg_psnr


def sox_dithering():
    total_psnr = 0

    for idx, lr_sample in enumerate(lr_samples):
        hr_sample = hr_samples[idx]
        wav.write('temp_source.wav', 16000, hr_sample.astype('int16'))
        data.resample_wav('temp_source.wav', 'temp_lr.wav', 8, True)
        data.resample_wav('temp_source.wav', 'temp_lr2.wav', 8, False)
        data.resample_wav('temp_lr.wav', 'temp_hr.wav', 16, True)
        hr_output = wav.read('temp_hr.wav')[1]

        psnr = metrics.bd_psnr_raw(hr_output, hr_sample)
        total_psnr += psnr

    avg_psnr = total_psnr / len(lr_samples)
    return avg_psnr


print(sox_dithering())
print(naive_upscaling())

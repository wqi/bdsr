import numpy as np
import sox

import scipy.io.wavfile as wav


# Downsamples WAV file to a given bit depth
def downsample_wav(file, bit_depth):
    tfm = sox.Transformer()
    tfm.convert(bitdepth=bit_depth)

    out_path = file.split('.wav')[0] + "_lr.wav"
    tfm.build(file, out_path)


# Downsamples NPZ file containing a set of WAVs
def downsample_mehri_audio_set(data_path, out_path, bit_depth):
    samples = np.load(data_path)
    lr_samples = np.empty_like(samples)

    # Downsample each audio sample in the NPZ
    for idx, sample in enumerate(samples):
        wav.write('temp.wav', 12800, sample.astype('float64'))
        downsample_wav('temp.wav', 8)
        lr_audio = wav.read('test_lr.wav')
        lr_samples[idx] = lr_audio[1]

    # Save downsampled audio in new NPZ
    print(lr_samples)
    np.save(out_path, lr_samples)


downsample_mehri_audio_set('data/music/music_train.npy', 'data/music/music_train_lr.npy', 8)

import numpy as np
import scipy.io.wavfile as wav
import sox


# Downsamples WAV file to a given bit depth
def resample_wav(file, out_path, bit_depth, dither):
    tfm = sox.Transformer()
    tfm.set_globals(dither=dither)
    tfm.convert(bitdepth=bit_depth)
    tfm.build(file, out_path)


# Downsamples NPZ file containing a set of WAVs
def downsample_mehri_set(data_path, out_path, bit_depth):
    samples = np.load(data_path)
    lr_samples = np.empty_like(samples)

    # Downsample each audio sample in the NPZ
    for idx, sample in enumerate(samples):
        wav.write('temp.wav', 16000, sample.astype('float64'))
        resample_wav('temp.wav', 'temp_lr.wav', bit_depth, False)
        lr_audio = wav.read('temp_lr.wav')
        lr_samples[idx] = lr_audio[1]

    # Save downsampled audio in new NPZ
    np.save(out_path, lr_samples)


# downsample_mehri_set('data/music_ref/music_test.npy', 'data/music/music_test_lr.npy', 8)

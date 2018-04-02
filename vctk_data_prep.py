import numpy as np
import os
import sox
import wavio


# Downsamples WAV file to a given bit depth with 16k sample rate
def downsample_wav(source_path, out_path, bit_depth):
    tfm = sox.Transformer()
    tfm.rate(16000)
    tfm.convert(bitdepth=bit_depth)

    tfm.build(source_path, out_path)


# Downsamples directory containing VCTK wav samples
def downsample_vctk_set(vctk_dir, out_path, bit_depth):
    out_samples = []

    # Get all VCTK file names in increasing order
    dir_files = os.listdir(vctk_dir)
    dir_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    # Downsample each audio sample in the directory
    for idx, file in enumerate(dir_files):
        source_path = os.path.join(vctk_dir, file)
        temp_path = 'temp.wav'

        downsample_wav(source_path, temp_path, bit_depth)
        out_audio = wavio.read(temp_path).data.squeeze()
        out_samples.append(out_audio)

    # Save downsampled audio in NPY file
    out_samples = np.array(out_samples)[-8:]
    print(out_samples.shape)
    np.save(out_path, out_samples)


downsample_vctk_set('data/vctk_ref/wav48/p225/', './data/vctk/vctk_test_lr.npy', 8)

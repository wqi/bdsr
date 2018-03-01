import numpy as np
import scipy.io.wavfile as wavfile


# Computes PSNR between two WAV files
def snr(ref_file, in_file):
    ref_wav = wavfile.read(ref_file)[1]
    in_wav = wavfile.read(in_file)[1]

    # Pad in_wav with 0s if ref_wav is slightly longer
    if (abs(in_wav.shape[0] - ref_wav.shape[0]) < 10):
        pad_width = ref_wav.shape[0] - in_wav.shape[0]
        in_wav = np.pad(in_wav, (0, pad_width), 'constant')
    else:
        print("Error: Reference WAV is of significantly different length from input WAV")
        return -1

    # Compute SNR
    norm_diff = np.square(np.linalg.norm(in_wav - ref_wav))
    if (norm_diff == 0):
        print("Error: Reference WAV is identical to input WAV")
        return -1

    ref_norm = np.square(np.linalg.norm(ref_wav))
    snr = 10 * np.log10(ref_norm / norm_diff)
    return snr


print(snr('./data/test/p225_001.wav', './data/test/p225_001.wav'))

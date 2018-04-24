import numpy as np
import scipy.io.wavfile as wav


def wav_snr(ref_file, in_file):
    """
    Compute SNR between two WAV files
    """
    ref_wav = wav.read(ref_file)[1]
    in_wav = wav.read(in_file)[1]

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


def bd_psnr_naive(lr, hr):
    """
    Compute PNSR between 8-bit PCM LR and 16-bit PCM HR inputs with naive upscaling
    """
    lr_scaled = (lr * 256)
    hr_scaled = (hr + 32768)

    mse = np.sum(np.square(hr_scaled - lr_scaled)) / lr.shape[0]
    psnr = 20 * np.log10(65535) - 10 * np.log10(mse)
    return psnr


def bd_psnr_raw(output, source):
    """
    Compute PNSR between 16-bit PCM upscaled output and 16-bit PCM source audio
    """
    mse = np.sum(np.square(source - output)) / output.shape[0]
    psnr = 20 * np.log10(65535) - 10 * np.log10(mse)
    return psnr

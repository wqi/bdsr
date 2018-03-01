import sox
import torchaudio


# IDs for SingleSpeaker test set from Kuleshov paper
test_speaker_id = 225
test_sample_ids = [355, 356, 357, 358, 359, 363, 365, 366]


# Loads tensor representation of VCTK sample
def load_vctk_wav(speaker_id, sample_id):
    vctk_root = './data/vctk/wav48'
    wav_path = vctk_root + '/p{}/p{}_{}.wav'.format(speaker_id, speaker_id, sample_id)

    return torchaudio.load(wav_path)


# Downsamples WAV file by a given factor
def downsample_wav(file, factor):
    hr_sample_rate = sox.file_info.sample_rate(file)
    lr_sample_rate = hr_sample_rate / factor

    tfm = sox.Transformer()
    tfm.rate(lr_sample_rate)

    out_path = file.split('.wav')[0] + "_lr.wav"
    tfm.build(file, out_path)


# Upsamples WAV file by a given factor using Sox's built-in method
def upsample_wav(file, factor):
    lr_sample_rate = sox.file_info.sample_rate(file)
    hr_sample_rate = lr_sample_rate * factor

    tfm = sox.Transformer()
    tfm.rate(hr_sample_rate)

    out_path = file.split('.wav')[0] + "_hr.wav"
    tfm.build(file, out_path)


upsample_wav('./data/test/p225_test_lr.wav', 4)

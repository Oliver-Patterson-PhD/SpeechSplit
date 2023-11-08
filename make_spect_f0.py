import os
import pickle
import numpy
import soundfile
import scipy
import librosa
import pysptk
import utils


mel_basis = librosa.filters.mel(sr=16000, n_fft=1024, fmin=90, fmax=7600, n_mels=80).T
min_level = numpy.exp(-100 / 20 * numpy.log(10))
b, a = utils.butter_highpass(30, 16000, order=5)

spk2gen = pickle.load(open('assets/spk2gen.pkl', "rb"))


# Modify as needed
rootDir = 'assets/wavs'
targetDir_f0 = 'assets/raptf0'
targetDir = 'assets/spmel'


dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

for subdir in sorted(subdirList):
    print(subdir)

    if not os.path.exists(os.path.join(targetDir, subdir)):
        os.makedirs(os.path.join(targetDir, subdir))
    if not os.path.exists(os.path.join(targetDir_f0, subdir)):
        os.makedirs(os.path.join(targetDir_f0, subdir))
    _, _, fileList = next(os.walk(os.path.join(dirName, subdir)))

    if spk2gen[subdir] == 'M':
        lo, hi = 50, 250
    elif spk2gen[subdir] == 'F':
        lo, hi = 100, 600
    else:
        raise ValueError

    prng = numpy.random.RandomState(int(subdir[1:]))
    for fileName in sorted(fileList):
        # read audio file
        x, fs = soundfile.read(os.path.join(dirName, subdir, fileName))
        assert fs == 16000
        if x.shape[0] % 256 == 0:
            x = numpy.concatenate((x, numpy.array([1e-06])), axis=0)
        y = scipy.signal.filtfilt(b, a, x)
        wav = y * 0.96 + (prng.rand(y.shape[0]) - 0.5) * 1e-06

        # compute spectrogram
        D = utils.pySTFT(wav).T
        D_mel = numpy.dot(D, mel_basis)
        D_db = 20 * numpy.log10(numpy.maximum(min_level, D_mel)) - 16
        S = (D_db + 100) / 100

        # extract f0
        f0_rapt = pysptk.sptk.rapt(wav.astype(numpy.float32) * 32768, fs, 256, min=lo, max=hi, otype=2)
        index_nonzero = (f0_rapt != -1e10)
        mean_f0, std_f0 = numpy.mean(f0_rapt[index_nonzero]), numpy.std(f0_rapt[index_nonzero])
        f0_norm = utils.speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)

        assert len(S) == len(f0_rapt)

        numpy.save(os.path.join(targetDir, subdir, fileName[:-4]),
                   S.astype(numpy.float32), allow_pickle=False)
        numpy.save(os.path.join(targetDir_f0, subdir, fileName[:-4]),
                   f0_norm.astype(numpy.float32), allow_pickle=False)

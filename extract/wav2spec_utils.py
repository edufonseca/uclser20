import os
import numpy as np
import scipy
import librosa
import soundfile
import matplotlib.pyplot as plt
import glob

#########################################################################
# Some of these functions have been inspired on the DCASE UTIL framework by Toni Heittola
# https://dcase-repo.github.io/dcase_util/
#########################################################################


def load_audio_file(file_path, input_fixed_length=0, pextract=None):

    try:
        data, source_fs = soundfile.read(file=file_path)
        data = data.T
    except:
        print('ERROR: Soundfile has crashed while reading {}'.format(file_path))
        print('Returning NONE and skipping this file. TO BE REMOVED FROM CSV.')
        return None

    # Resample if the source_fs is different from expected
    if pextract.get('fs') != source_fs:
        data = librosa.core.resample(data, source_fs, pextract.get('fs'))

    if len(data) > 0:
        data = get_normalized_audio(data)
    else:
        raise ValueError ('File corrupted. Could not open: %s' % file_path)
        # data = np.ones((input_fixed_length, 1))
        # print('File corrupted. Could not open: %s' % file_path)

    # careful with the shape
    data = np.reshape(data, [-1, 1])
    return data



def modify_file_variable_length(data=None, input_fixed_length=0, pextract=None):
    """
    data is the entire audio file loaded, with proper shape
    -depending on the loading mode (in pextract)
    --FIX: if sound is short, replicate sound to fill up to input_fixed_length
           if sound is too long, grab only a (random) slice of size  input_fixed_length
    --VARUP: short sounds get replicated to fill up to input_fixed_length
    --VARFULL: this function is a by pass (full length is considered, whatever that is)

    :return:
    """

    # deal with short sounds
    if len(data) <= input_fixed_length:
        if pextract.get('load_mode') == 'fix' or pextract.get('load_mode') == 'varup':
            if pextract.get('fill') == 'rep':
                # if file shorter than input_length, replicate the sound to reach the input_fixed_length
                nb_replicas = int(np.ceil(input_fixed_length / len(data)))
                # replicate according to column
                data_rep = np.tile(data, (nb_replicas, 1))
                data = data_rep[:input_fixed_length]

            elif pextract.get('fill') == 'zp':
                # if file shorter than input_length, zeropad the sound to reach the input_fixed_length
                data = librosa.util.fix_length(data, input_fixed_length)
            else:
                raise ValueError('unknown filling method')

    else:
        # deal with long sounds
        if pextract.get('load_mode') == 'fix':
            # if file longer than input_length, grab input_length from a random slice of the file
            max_offset = len(data) - input_fixed_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_fixed_length + offset)]

    return data

def get_normalized_audio(y, head_room=0.005):

    mean_value = np.mean(y)
    y -= mean_value

    max_value = max(abs(y)) + head_room
    return y / max_value


def get_mel_spectrogram(audio, pextract=None):
    """Mel-band energies

    Parameters
    ----------
    audio : numpy.ndarray
        Audio data.
    params : dict
        Parameters.

    Returns
    -------
    feature_matrix : numpy.ndarray
        (log-scaled) mel spectrogram energies per audio channel

    """
    # make sure rows are channels and columns the samples
    audio = audio.reshape([1, -1])
    window = scipy.signal.hamming(pextract.get('win_length_samples'), sym=False)

    mel_basis = librosa.filters.mel(sr=pextract.get('fs'),
                                    n_fft=pextract.get('n_fft'),
                                    n_mels=pextract.get('n_mels'),
                                    fmin=pextract.get('fmin'),
                                    fmax=pextract.get('fmax'),
                                    htk=pextract.get('htk'),
                                    norm=pextract.get('mel_basis_unit'))

    if pextract.get('normalize_mel_bands'):
        mel_basis /= np.max(mel_basis, axis=-1)[:, None]

    # init mel_spectrogram expressed as features: row x col = frames x mel_bands = 0 x mel_bands (to concatenate axis=0)
    feature_matrix = np.empty((0, pextract.get('n_mels')))
    for channel in range(0, audio.shape[0]):
        spectrogram = get_spectrogram(
            y=audio[channel, :],
            n_fft=pextract.get('n_fft'),
            win_length_samples=pextract.get('win_length_samples'),
            hop_length_samples=pextract.get('hop_length_samples'),
            spectrogram_type=pextract.get('spectrogram_type') if 'spectrogram_type' in pextract else 'magnitude',
            center=True,
            # center=False,
            window=window,
            pextract=pextract
        )

        mel_spectrogram = np.dot(mel_basis, spectrogram)
        mel_spectrogram = mel_spectrogram.T
        # at this point we have row x col = time x freq = frames x mel_bands

        if pextract.get('log'):
            mel_spectrogram = np.log10(mel_spectrogram + pextract.get('eps'))

        feature_matrix = np.append(feature_matrix, mel_spectrogram, axis=0)

    return feature_matrix



def get_spectrogram(y,
                    n_fft=1024,
                    win_length_samples=0.04,
                    hop_length_samples=0.02,
                    window=scipy.signal.hamming(1024, sym=False),
                    center=True,
                    spectrogram_type='magnitude',
                    pextract=None):

    """Spectrogram

    Parameters
    ----------
    y : numpy.ndarray
        Audio data
    n_fft : int
        FFT size
        Default value "1024"
    win_length_samples : float
        Window length in seconds
        Default value "0.04"
    hop_length_samples : float
        Hop length in seconds
        Default value "0.02"
    window : array
        Window function
        Default value "scipy.signal.hamming(1024, sym=False)"
    center : bool
        If true, input signal is padded so to the frame is centered at hop length
        Default value "True"
    spectrogram_type : str
        Type of spectrogram "magnitude" or "power"
        Default value "magnitude"

    Returns
    -------
    np.ndarray [shape=(1 + n_fft/2, t), dtype=dtype]
        STFT matrix

    """

    if spectrogram_type == 'magnitude':
        return np.abs(librosa.stft(y + pextract.get('eps'),
                                   n_fft=n_fft,
                                   win_length=win_length_samples,
                                   hop_length=hop_length_samples,
                                   center=center,
                                   window=window))
    elif spectrogram_type == 'power':
        return np.abs(librosa.stft(y + pextract.get('eps'),
                                   n_fft=n_fft,
                                   win_length=win_length_samples,
                                   hop_length=hop_length_samples,
                                   center=center,
                                   window=window)) ** 2
    else:
        message = 'Unknown spectrum type [{spectrogram_type}]'.format(
            spectrogram_type=spectrogram_type
        )
        raise ValueError(message)


def get_mel_spectrogram_lib(audio, pextract=None):

    audio = audio.reshape([1, -1])

    window = scipy.signal.hamming(pextract.get('win_length_samples'), sym=False)

    mel_spectrogram = librosa.feature.melspectrogram(y=audio[0, :],
                                                     sr=pextract.get('fs'),
                                                     win_length=pextract.get('win_length_samples'),
                                                     hop_length=pextract.get('hop_length_samples'),
                                                     window=window,
                                                     n_fft=pextract.get('n_fft'),
                                                     n_mels=pextract.get('n_mels'),
                                                     center=True,
                                                     # center=False,
                                                     power=2
                                                     ).T

    if pextract.get('log'):
        mel_spectrogram = np.log10(mel_spectrogram + pextract.get('eps'))

    return mel_spectrogram


def make_sure_isdir(_path):
    """
    make sure the a directory at the end of pre_path exists. Else create it
    :param pre_path:
    :param args:
    :return:
    """
    # full_path = os.path.join(pre_path, _out_file)
    if not os.path.exists(_path):
        os.makedirs(_path)
    return _path
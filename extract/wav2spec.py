import os
import argparse
import numpy as np
import pprint
import yaml
from wav2spec_utils import load_audio_file, modify_file_variable_length, get_mel_spectrogram, make_sure_isdir
import h5py
import random


def compute_tf_rep(pctrl=None, pextract=None, mode='train'):
    random.seed(1)

    pextract['audio_len_samples'] = int(pextract.get('fs') * pextract.get('audio_len_s'))

    if mode == 'train' or mode=="val":
        csv_info = [l.strip('\n') for l in open(os.path.join(pctrl.get('root'), 'train.csv')).readlines()]
        names = np.asarray([str.split(k, ",")[0] for k in csv_info[1:]])
        labels_ = np.asarray([str.split(k, ",")[1] for k in csv_info[1:]])
        belongs2clean = np.asarray([str.split(k, ",")[3] for k in csv_info[1:]])
        idx_validation = belongs2clean.astype(int)
        audio_path = np.asarray([pctrl.get('root_data_train') + str.split(k, ",")[0] for k in csv_info[1:]])

    elif mode =='test':
        csv_info = [l.strip('\n') for l in open(os.path.join(pctrl.get('root'), 'test.csv')).readlines()]
        names = np.asarray([str.split(k, ",")[0] for k in csv_info[1:]])
        labels_ = np.asarray([str.split(k, ",")[1] for k in csv_info[1:]])
        audio_path = np.asarray([pctrl.get('root_data_test') + str.split(k, ",")[0] for k in csv_info[1:]])

    label_names = np.unique(labels_)
    labels = []
    for k in labels_:
        labels.append(np.argwhere(label_names == k)[0][0])

    labels = np.asarray(labels)


    _ = make_sure_isdir(pctrl.get('hdf5_path'))
    hdf5_file = h5py.File(pctrl.get('hdf5_path') + 'FSDnoisy18k_' + mode +'.hdf5', mode='w')

    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    dt2 = h5py.special_dtype(vlen=str)

    if mode=="train":
        set_len = len(audio_path)-idx_validation.sum()
    elif mode=="val":
        set_len = idx_validation.sum()
    elif mode=="test":
        set_len = len(audio_path)

    dset = hdf5_file.create_dataset('binary_data', (set_len,), dtype=dt)
    lb = hdf5_file.create_dataset("labels", (set_len,), dtype=dt2)
    cont = 0

    for i, path in enumerate(audio_path):

        if mode=="train" and idx_validation[i]==0:
            # load entire audio file and modify variable length, if needed
            y = load_audio_file(path, input_fixed_length=pextract['audio_len_samples'], pextract=pextract)
            y = modify_file_variable_length(data=y, input_fixed_length=pextract['audio_len_samples'],
                                            pextract=pextract)
            # compute log-mel spec. row x col = time x freq
            mel_spectrogram = get_mel_spectrogram(audio=y, pextract=pextract)

            dset[cont] = np.frombuffer(mel_spectrogram.tobytes(), dtype='uint8')
            lb[cont] = int(labels[i])
            cont += 1

        elif mode == "val" and idx_validation[i] == 1:
            # load entire audio file and modify variable length, if needed
            y = load_audio_file(path, input_fixed_length=pextract['audio_len_samples'], pextract=pextract)
            y = modify_file_variable_length(data=y, input_fixed_length=pextract['audio_len_samples'],
                                            pextract=pextract)
            # compute log-scaled mel spec. row x col = time x freq
            mel_spectrogram = get_mel_spectrogram(audio=y, pextract=pextract)

            dset[cont] = np.frombuffer(mel_spectrogram.tobytes(), dtype='uint8')
            lb[cont] = int(labels[i])
            cont += 1

        elif mode=="test":
            # load entire audio file and modify variable length, if needed
            y = load_audio_file(path, input_fixed_length=pextract['audio_len_samples'], pextract=pextract)
            y = modify_file_variable_length(data=y, input_fixed_length=pextract['audio_len_samples'],
                                            pextract=pextract)
            # compute log-scaled mel spec. row x col = time x freq
            mel_spectrogram = get_mel_spectrogram(audio=y, pextract=pextract)

            dset[cont] = np.frombuffer(mel_spectrogram.tobytes(), dtype='uint8')
            lb[cont] = int(labels[i])
            cont += 1

    print('Files processed in mode {}: {}'.format(mode, cont))

if __name__ == '__main__':

    # ==================================================================== ARGUMENTS
    parser = argparse.ArgumentParser(description='Compute mel spectrogram features')
    parser.add_argument('-m', '--mode',
                        help="Either 'test' or 'train' or 'val', to define the set over which compute features.",
                        dest='mode',
                        action='store',
                        required=True,
                        type=str)
    parser.add_argument('-s', '--setting',
                        help="path to yaml with feature extraction parameters, eg, ../config/params_v1.yaml.",
                        dest='setting',
                        action='store',
                        required=True,
                        type=str)

    args = parser.parse_args()

    # settings for computing features
    audio_len_ss = [1]  # patch length in seconds
    fills = ['rep']     # clips shorter than 1s are replicated til 1s
    bandss = [96]       # number of mel bands

    params_yaml = args.setting

    print('\nYaml file with parameters defining the experiment: {}\n'.format(str(params_yaml)))
    params = yaml.load(open(params_yaml), yaml.FullLoader)
    pctrl = params['ctrl']
    pextract = params['extract']

    print('\nExperimental setup:')
    print('pctrl=')
    pprint.pprint(pctrl, width=1, indent=4)
    print('pextract=')
    pprint.pprint(pextract, width=1, indent=4)

    for audio_len_s in audio_len_ss:
        for fill in fills:
            for bands in bandss:
                pextract['audio_len_s']=audio_len_s
                pextract['fill']=fill
                pextract['n_mels']=bands

                print('===Extracting features with patch duration {}, filling {}'.format(audio_len_s, fill))
                compute_tf_rep(pctrl=pctrl, pextract=pextract, mode=args.mode)

    print('End of job.')

import numpy as np
import h5py
from torch.utils.data import Dataset


def fetch_file_2_tensor(mel_spec, label_original, patch_hop, patch_len):
    """
    Given a mel_spec, perform slicing into T-F patches, and store them in a list. Create a list of labels of same shape
    inheriting clip-level labels.
    :param mel_spec:
    :return: two lists of patches and labels of same shape.
    """

    idx = 0
    start = 0
    im_TF_patches = []
    labels = []

    while (start + patch_len) <= mel_spec.shape[0]:
        im_TF_patches.append(mel_spec[start: start + patch_len])
        labels.append(label_original)
        # update indexes
        start += patch_hop
        idx += 1

    return im_TF_patches, labels


def make_dataset(args, mode, hdf5_path, idxVal=[]):

    np.random.seed(271828)
    if mode == "train" and args['learn']['train_on_clean'] == 1:
        # in one downstream task, we train on the train_clean set (which is saved in FSDnoisy18k_val.hdf5)
        # we'll pick 85% only for training later on
        data_hdf5 = h5py.File(hdf5_path + 'FSDnoisy18k_' + "val" + '.hdf5', mode='r')
    else:
        # by default we train on train_noisy, validate on train_clean, and evaluate on the test set
        data_hdf5 = h5py.File(hdf5_path + 'FSDnoisy18k_' + mode + '.hdf5', mode='r')

    if len(data_hdf5['labels']) == 0:
        raise (RuntimeError('Found 0 patches, please check the data set'))


    labels_name = np.array(data_hdf5['labels'])

    # create dicts such that key: value is as follows
    # label: int
    # int: label
    list_labels = np.unique(labels_name)
    label_to_int = {k: v for v, k in enumerate(list_labels)}
    int_to_label = {v: k for k, v in label_to_int.items()}

    labels_original = []
    for k in labels_name:
        labels_original.append(label_to_int[k])
    labels_original = np.asarray(labels_original)


    if mode == "train" and args['learn']['train_mode']=="static_slice":
        # save all patches as train data to be returned

        per_class_samples_accum = np.zeros((args['learn']['num_classes'],))
        per_class_samples_num = [[] for i in range(args['learn']['num_classes'])]
        per_class_samples_idx = [[] for i in range(args['learn']['num_classes'])]

        val_sum = 0.0
        num_values = 0.0

        ## First loop to calculate data shape
        for i in range(len(data_hdf5['binary_data'])):

            im = np.ascontiguousarray(np.rec.array(np.frombuffer(data_hdf5['binary_data'][i])).reshape(-1, args['extract']['n_mels']))
            val_sum += im.sum()
            num_values += im.size
            file_frames = float(im.shape[0])  # number of time frames
            # number of patches within clip
            nb_inst = np.maximum(1, int(np.ceil((file_frames - args['extract']['patch_len']) / args['extract']['patch_hop'])))
            per_class_samples_accum[labels_original[i]] += nb_inst
            per_class_samples_num[labels_original[i]].append(nb_inst) #number of TF patches for each class and spectogram
            per_class_samples_idx[labels_original[i]].append(i) #overall indexes per-class

        spec_mean = val_sum / num_values
        val_std = 0.0

        train_size = int(per_class_samples_accum.sum())
        # initialize data and labels to be returned
        data = np.zeros((train_size, args['extract']['patch_len'], args['extract']['n_mels'])).astype('float32')
        labels = np.zeros((train_size,)).astype(int)
        train_count = 0
        for i in range(len(data_hdf5['binary_data'])):
            im = np.ascontiguousarray(np.rec.array(np.frombuffer(data_hdf5['binary_data'][i])).reshape(-1, args['extract']['n_mels']))
            val_std += ((im - spec_mean) ** 2).sum()
            im_TF_patches, labels_TF_patches = fetch_file_2_tensor(im, labels_original[i], args['extract']['patch_hop'], args['extract']['patch_len'])
            # save all patches in a clip as train data to be returned
            data[train_count:train_count + len(im_TF_patches)] = np.asarray(im_TF_patches)
            labels[train_count:train_count + len(im_TF_patches)] = np.asarray(labels_TF_patches).astype(int)
            train_count += len(labels_TF_patches)

        spec_std = np.sqrt(val_std) / num_values

    elif mode=='train' and args['learn']['train_mode'] == "dynamic_slice":
        # instead of slicing TF patches and returning all of them exhaustively, we will return a random patch per clip
        # here, we simply save the entire clip spectrogram. Then in get_item() we'll select a random patch (different in each epoch)
        # initialize data and labels to be returned
        data = []
        labels = []
        val_sum = 0.0
        num_values = 0.0

        for i in range(len(data_hdf5['binary_data'])):
            im = np.ascontiguousarray(np.rec.array(np.frombuffer(data_hdf5['binary_data'][i])).reshape(-1, args['extract']['n_mels']))
            val_sum += im.sum()
            num_values += im.size
            # append one item (entire clip spectrogram) and the corresponding label
            data.append(np.asarray(im).astype('float32'))
            labels.append(np.asarray(labels_original[i]))
        labels = np.asarray(labels).astype(int)

        if args['learn']['train_on_clean'] == 1:
            # in one downstream task we train on 85% of train_clean and we use 15% of train_clean set for validation
            data_val, data_train = [], []
            labels_val, labels_train = [], []

            np.random.seed(args['learn']['seed_initialization'])
            # shuffle indexes for all train clips
            idxes = np.random.permutation(len(labels))
            # amount of clips for val set
            val_num = round(float(len(idxes))*0.15)
            # select 15% of clips for validation and the remaining 85% for training
            idxVal = idxes[0:val_num]
            idxTrain = idxes[val_num:]

            ##Recompute sum of values to get average value for normalization, ONLY FOR TRAIN CLIPS
            val_sum = 0.0
            num_values = 0.0
            for i in range(len(labels)):
                # consider only the training clips (ignore those for validation)
                if np.sum(i==idxTrain)>0:
                    data_train.append(data[idxes[i]])
                    labels_train.append(labels[idxes[i]])
                    val_sum += data[idxes[i]].sum()
                    num_values += data[idxes[i]].size

            data = data_train
            labels = labels_train
            print("Keep some data for validation")

        spec_mean = val_sum/num_values
        val_std = 0.0
        for i in range(len(data_hdf5['binary_data'])):
            im = np.ascontiguousarray(np.rec.array(np.frombuffer(data_hdf5['binary_data'][i])).reshape(-1, args['extract']['n_mels']))
            val_std += ((im-spec_mean)**2).sum()
        spec_std = np.sqrt(val_std)/num_values

    elif mode=='val' or mode=='test':
        data = []
        labels = []
        if args['learn']['val_mode'] !="balanced":
            # args['learn']['val_mode'] = 'unbalanced' this way we take advantage of all data available for evaluation purposes
            for i in range(len(data_hdf5['binary_data'])):
                im = np.ascontiguousarray(np.rec.array(np.frombuffer(data_hdf5['binary_data'][i])).reshape(-1, args['extract']['n_mels']))
                im_TF_patches, labels_TF_patches = fetch_file_2_tensor(im, labels_original[i], args['extract']['patch_hop'], args['extract']['patch_len'])
                if mode=="val" and args['learn']['train_on_clean'] == 1 and np.sum(idxVal == i) > 0:
                    # we only return the clips of train_clean designated for validation, as per idxVal
                    data.append(np.asarray(im_TF_patches).astype('float32'))
                    labels.append(np.asarray(labels_TF_patches))
                else:
                    data.append(np.asarray(im_TF_patches).astype('float32'))
                    labels.append(np.asarray(labels_TF_patches))
        else:
            # balanced val_mode
            # set constant number of items per class to the minimum per class number of items
            num_samples_per_class = np.bincount(labels_original).min()
            per_class_counter = np.zeros((args['learn']['num_classes']))
            for i in range(len(data_hdf5['binary_data'])):
                if per_class_counter[labels_original[i]] < num_samples_per_class:
                    im = np.ascontiguousarray(np.rec.array(np.frombuffer(data_hdf5['binary_data'][i])).reshape(-1, args['extract']['n_mels']))
                    im_TF_patches, labels_TF_patches = fetch_file_2_tensor(im, labels_original[i], args['extract']['patch_hop'], args['extract']['patch_len'])
                    data.append(np.asarray(im_TF_patches).astype('float32'))
                    labels.append(np.asarray(labels_TF_patches))
                    per_class_counter[labels_original[i]] += 1


    if mode=="train" and args['learn']['train_on_clean']==1:
        # in one downstream task we train on 85% of train_clean and we use 15% of train_clean set for validation
        # we return 85% of train_clean in data. And idxVal: the indexes of clips that we defined for validation within train_clean
        return data, labels, label_to_int, int_to_label, spec_mean, spec_std, idxVal

    if mode=="train":
        # we return entire train_noisy in data
        return data, labels, label_to_int, int_to_label, spec_mean, spec_std

    else:
        # we return entire train_clean (default for val), or
        # only the 15% of train_clean as per idxVal (when val and ['train_on_clean']==1), or
        # the entire test set (default for test)
        return data, labels, label_to_int, int_to_label


class FSDnoisy18k(Dataset):
    def __init__(self, args, mode="train", hdf5_path="./FSDnoisy18k/data_hdf5/", idxVal=[]):

        if mode == "train":
            if args['learn']['train_on_clean'] == 1:
                # in one downstream task we train on 85% of train_clean and we use 15% of train_clean set for validation
                # idxVal contains indexes for the 15% of train_clean as validation clips
                data, labels, label_to_int, int_to_label, spec_mean, spec_std, idxVal = make_dataset(args, mode, hdf5_path)
            else:
                # default for train
                data, labels, label_to_int, int_to_label, spec_mean, spec_std = make_dataset(args, mode, hdf5_path)

            self.data_mean = spec_mean
            self.data_std = spec_std
            self.idxVal = idxVal

        elif mode == "val" and args['learn']['train_on_clean']==1:
            # we pass idxVal to select only the 15% of clips in train_clean
            data, labels, label_to_int, int_to_label = make_dataset(args, mode, hdf5_path, idxVal)
        else:
            # default for val or test
            data, labels, label_to_int, int_to_label = make_dataset(args, mode, hdf5_path)

        # data and labels to be batched and feed net
        self.data = data
        self.labels = labels
        self.label_to_int = label_to_int
        self.int_to_label = int_to_label

        self.mode = mode  # train, val, test

        self.args = args
        self.num_classes = args['learn']['num_classes']
        self._count = 1

    def __getitem__(self, index):
        # given an index, return patch, labels

        if self.mode == "train":
            img, labels = self.data[index], self.labels[index]
        elif self.mode == "test" or self.mode == "val":
            img, labels = self.data[index], self.labels[index]


        if self.mode == "train" and self.args['learn']['train_mode'] == "dynamic_slice":
            # Get dynamic slice from spectrogram
            if (img.shape[0]-self.args['extract']['patch_len']) == 0:
                # If image has size equal to path_len, no crop to do
                img1 = img.copy()
            else:
                start = np.random.randint(0, img.shape[0] - self.args['extract']['patch_len'])
                img1 = img[start: start + self.args['extract']['patch_len']]

            if self.args['learn']['method'] == "Contrastive":
                # what is the second patch starting point?
                if self.args['learn']['CL_positives'] == 'same':
                    # get the same patch (worst case)
                    img2 = img1.copy()
                elif self.args['learn']['CL_positives'] == 'within_clip':
                    # randomly get a second TF patch within the clip - no overlap
                    if (img.shape[0] - self.args['extract']['patch_len']) == 0:
                        # If image has size equal to path_len, no crop to do
                        img2 = img.copy()
                    else:
                        start2 = np.random.randint(0, img.shape[0] - self.args['extract']['patch_len'])
                        img2 = img[start2: start2 + self.args['extract']['patch_len']]

        else:
            # in static_slice, we are already loading TF patches of 101x96
            img1 = img.copy()

        # Apply mix-back==============================
        if self.mode == "train" and self.args['learn']['method'] == "Contrastive" and self.args['learn']['CL_pos_mix'] in ['mix_within_clip', 'mix_out_clip']:
            # here we have img1, img2
            # mix each patch with a background patch to further do DA and mitigate potential shortcuts
            # NOTE: we mix here so that, if we apply transforms, they are applied to the mixture foreground + background

            # 1) get two background spectrograms, back1 and back2
            if self.args['learn']['CL_pos_mix'] == 'mix_within_clip':
                # background patches are taken within clip, randomly
                back1 = np.zeros_like(img1)
                back2 = np.zeros_like(img2)
                if img.shape[0] >= 4 * self.args['extract']['patch_len']:
                    # requires a min length of 4 seconds, to have variability
                    start_back1 = np.random.randint(0, img.shape[0] - self.args['extract']['patch_len'])
                    start_back2 = np.random.randint(0, img.shape[0] - self.args['extract']['patch_len'])
                    back1 = img[start_back1: start_back1 + self.args['extract']['patch_len']]
                    back2 = img[start_back2: start_back2 + self.args['extract']['patch_len']]

            elif self.args['learn']['CL_pos_mix'] == 'mix_out_clip':
                # select two background clips, no constraints (may be within batch, but rarely due to train size)
                idx_back1 = np.random.randint(0, len(self.data))
                while idx_back1 == index:
                    idx_back1 = np.random.randint(0, len(self.data))
                idx_back2 = np.random.randint(0, len(self.data))
                while idx_back2 == index or idx_back2 == idx_back1:
                    idx_back2 = np.random.randint(0, len(self.data))
                back1_full = self.data[idx_back1]
                back2_full = self.data[idx_back2]

                if back1_full.shape[0] > self.args['extract']['patch_len']:
                    start_back1 = np.random.randint(0, back1_full.shape[0] - self.args['extract']['patch_len'])
                    back1 = back1_full[start_back1: start_back1 + self.args['extract']['patch_len']]
                else:
                    back1 = back1_full
                if back2_full.shape[0] > self.args['extract']['patch_len']:
                    start_back2 = np.random.randint(0, back2_full.shape[0] - self.args['extract']['patch_len'])
                    back2 = back2_full[start_back2: start_back2 + self.args['extract']['patch_len']]
                else:
                    back2 = back2_full

            # 2) mix img1 + back1, and img2 + back2
            # convex combination of foreground patch with a background patch
            # pay special attention to the energy difference between imgx and backx.
            # if we ensure imgx is dominant, we avoid dramatically corrupting foreground, eg, if img is almost silence and back has strong signal

            alpha_mix = np.random.uniform(low=0, high=self.args['learn']['CL_pos_mix_alpha'])

            def mix_energy(_patch, _back, _alpha_mix):

                # adjust foreground
                _patch_lin = 10 ** _patch
                # compute energy now in the linear domain
                _e_patch = np.sum(_patch_lin)
                _patch_lin_adj = _patch_lin * (1-_alpha_mix)

                # adjust background patch energy with respect to that of fore
                _back_lin = 10 ** _back
                # compute energy now in the linear domain
                _e_back = np.sum(_back_lin)

                # true multiplier factor for background depends on the relative amplitude differences
                # to ensure that the back energy is always much lower than that of foreground
                _e_back_target = _alpha_mix * _e_patch
                _beta_mix = _e_back_target / _e_back

                _back_lin_adj = _back_lin * _beta_mix

                # mix in the linear domain, and log
                out_patch = np.log10(_patch_lin_adj + _back_lin_adj)
                return out_patch

            img1 = mix_energy(img1, back1, alpha_mix)
            img2 = mix_energy(img2, back2, alpha_mix)
            # end of mix-back========================

        # imgx can be only the patch, or the patch already mixed with another background patch and the transform applies to both
        if self.transform is not None:
            img1 = self.transform(img1)

            if self.mode == "train" and self.args['learn']['method'] == "Contrastive":
                # Apply a different augmentation to the 2nd TF patch
                img2 = self.transform(img2)

            if self.mode == "test" or self.mode == "val":
                img1 = img1.permute((1, 2, 0))

        # return
        if self.mode == "train":
            if self.args['learn']['method'] == "Contrastive":
                return img1, img2, labels, index
            else:
                return img1, labels, index

        elif self.mode == "test" or self.mode == "val":
            return img1, labels, index

    def __len__(self):
        return len(self.data)


def get_dataset(args):
    # prepare subsets of FSDnoisy18k

    #################################### Train set #############################################
    dataset_train = FSDnoisy18k(args, mode="train", hdf5_path=args['ctrl']['hdf5_path'])
    #################################### Validation set ########################################
    if args['learn']['train_on_clean'] == 1:
        # in one downstream task we train on 85% of train_clean and we use 15% of train_clean set for validation
        dataset_val = FSDnoisy18k(args, mode="val", hdf5_path=args['ctrl']['hdf5_path'], idxVal=dataset_train.idxVal)
    else:
        # in the rest of cases, we use the entire train_clean set for validation
        dataset_val = FSDnoisy18k(args, mode="val", hdf5_path=args['ctrl']['hdf5_path'])
    #################################### Test set ##############################################
    dataset_test = FSDnoisy18k(args, mode="test", hdf5_path=args['ctrl']['hdf5_path'])

    return dataset_train, dataset_val, dataset_test


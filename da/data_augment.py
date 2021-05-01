import numpy as np
import copy
import random
import cv2


# specAugment inspired from
# https://github.com/DemisEom/SpecAugment/blob/master/SpecAugment/spec_augment_pytorch.py
# https://github.com/zcaceres/spec_augment/blob/master/SpecAugment.ipynb

def freq_mask(_spec, F=30, n_masks=1, mask_val='zero', reduce_mask_range=0):
    """
    differences from specAugment:
    - reduce_mask_range: float [0:1] to restrict the freq range where we apply masking. both bw and f0 are affected.
                if None, ignored. Typical value: 0.5 to focus on the lowest half of the spectrum
    - mask_val allows for several mask values

    :param _spec:
    :param F:
    :param n_masks:
    :param mask_val:
    :return:
    """
    n_frames = _spec.shape[0]
    n_bands = _spec.shape[1]
    for i in range(n_masks):
        # bw drawn from a uniform distribution from 0 to parameter F
        # f0 drawn from [0, n_bands-bw]
        if reduce_mask_range == 0:
            # 0 means we do nothing
            bw = int(np.random.uniform(low=0.0, high=F))
            f0 = random.randint(0, n_bands - bw)
        else:
            # but, do we want to reduce the mask bw accordingly (mild DA)?
            # bw = int(np.random.uniform(low=0.0, high=int(F*reduce_mask_range)))

            # or keep the original mask bw in a reduced range (strong DA)?
            bw = int(np.random.uniform(low=0.0, high=F))

            # the range where the mask is applied is always restricted to LF
            f0 = random.randint(0, int(n_bands*reduce_mask_range) - bw)

        if mask_val == 'zero':
            _spec[:, f0: f0 + bw] = 0
        elif mask_val == 'min':
            _spec[:, f0: f0 + bw] = _spec.min()
        elif mask_val == 'mean':
            _spec[:, f0: f0 + bw] = _spec.mean()
        elif mask_val == 'max':
            _spec[:, f0: f0 + bw] = _spec.max()
        elif mask_val == 'noise':
            _spec[:, f0: f0 + bw] = np.random.normal(_spec.mean(), _spec.std(), size=(n_frames, bw))
    return _spec


def time_mask(_spec, T=40, n_masks=1, mask_val='zero'):
    """
    differences from specAugment:
    - mask_val allows for several mask values
    - we ignore the parameter p that controls the maximum deltat for simplicity.

    :param _spec:
    :param T:
    :param n_masks:
    :param mask_val:
    :return:
    """

    n_frames = _spec.shape[0]
    n_bands = _spec.shape[1]
    for i in range(n_masks):
        # deltat drawn from a uniform distribution from 0 to parameter T
        # t0 drawn from [0, n_frames-deltat]
        deltat = int(np.random.uniform(low=0.0, high=T))
        t0 = random.randint(0, n_frames - deltat)

        if mask_val == 'zero':
            _spec[t0: t0 + deltat, :] = 0
        elif mask_val == 'min':
            _spec[t0: t0 + deltat, :] = _spec.min()
        elif mask_val == 'mean':
            _spec[t0: t0 + deltat, :] = _spec.mean()
        elif mask_val == 'max':
            _spec[t0: t0 + deltat, :] = _spec.max()
        elif mask_val == 'noise':
            _spec[t0: t0 + deltat, :] = np.random.normal(_spec.mean(), _spec.std(), size=(deltat, n_bands))
    return _spec


def spec_augment(_spec,
                 do_time_warp=True,
                 W=40,
                 do_freq_mask=True,
                 F=20,
                 m_f=1,
                 reduce_mask_range=0,
                 do_time_mask=True,
                 T=20,
                 m_t=1,
                 mask_val='zero'):
    """Spec augmentation Calculation Function as variant of specAugment.

    # Arguments:
     _spec (numpy array): rows=T, cols=F
      W: time_warping_para(float):
      F: frequency mask parameter
      m_f: number of frequency masks
      T: time mask parameter
      m_t: number of time masks
      - reduce_mask_range: float [0:1] to restrict the freq range where we apply masking. both bw and f0 are affected.
                if None, ignored. Typical value: 0.5 to focus on the lowest half of the spectrum
      - mask_val allows for several mask values

    # Returns
      _spec(numpy array): warped and masked mel spectrogram.
    """

    _spec_aug = copy.deepcopy(_spec)
    # Step 1 : Time warping, not used

    # Step 2 : Frequency masking
    # hard wideband suppression of bw bands starting from f0
    # bw drawn from a uniform distribution from 0 to parameter F
    # f0 drawn from [0, n_bands-bw]
    if do_freq_mask:
        _spec_aug = freq_mask(_spec_aug, F=F, n_masks=m_f, mask_val=mask_val, reduce_mask_range=reduce_mask_range)

    # Step 3 : Time masking
    # sound mute of deltat consecutive frames, starting at t0
    # deltat drawn from a uniform distribution from 0 to parameter T
    # t0 drawn from [0, n_frames-deltat]
    if do_time_mask:
        _spec_aug = time_mask(_spec_aug, T=T, n_masks=m_t, mask_val=mask_val)

    return _spec_aug


class SpecAugment:
    def __init__(self,
                 do_time_warp=False,
                 W=40,
                 do_freq_mask=True,
                 F=20,
                 m_f=1,
                 reduce_mask_range=0,
                 do_time_mask=False,
                 T=20,
                 m_t=1,
                 mask_val='zero'):
        self.do_time_warp = do_time_warp
        self.W = W
        self.do_freq_mask = do_freq_mask
        self.F = F
        self.m_f = m_f
        self.reduce_mask_range = reduce_mask_range
        self.do_time_mask = do_time_mask
        self.T = T
        self.m_t = m_t
        self.mask_val = mask_val

    def __call__(self, spec):
        return spec_augment(spec,
                            self.do_time_warp,
                            self.W,
                            self.do_freq_mask,
                            self.F,
                            self.m_f,
                            self.reduce_mask_range,
                            self.do_time_mask,
                            self.T,
                            self.m_t,
                            self.mask_val)


def random_time_shift(_spec, Tshift=50):
    """
    random delay (effect of circular convolution)
    :param _spec:
    :param Tshift: max number of delay frames. delay is drawn from normal distribution (1, Tshift)
    :return:
    """

    n_frames = _spec.shape[0]

    # deltat drawn from a uniform distribution from 1 to parameter Tshift
    # minimum shift is one frame (avoids crash)
    deltat = int(np.random.uniform(low=1.0, high=Tshift))

    # allocate
    _spec_out = np.zeros_like(_spec)

    # delay shift
    # end
    _spec_out[deltat:, :] = _spec[:n_frames-deltat, :]
    # begin
    _spec_out[:deltat, :] = _spec[-deltat:, :]
    return _spec_out


class RandTimeShift:
    def __init__(self,
                 do_rand_time_shift=True,
                 Tshift=50):
        self.do_rand_time_shift = do_rand_time_shift
        self.Tshift = Tshift

    def __call__(self, spec):
        if self.do_rand_time_shift:
            return random_time_shift(spec, self.Tshift)
        else:
            return spec


def random_freq_shift(_spec, Fshift=20):
    """
    random upwards shift in frequency: low band energies increase pitch. Acoustic aberration
    Fshift must be not very large.

    :param _spec:
    :param Fshift:
    :return:
    """
    n_bands = _spec.shape[1]

    # deltaf drawn from a uniform distribution from 1 to parameter Fshift
    # minimum shift is one band (avoids crash)
    deltaf = int(np.random.uniform(low=1.0, high=Fshift))

    # allocate
    _spec_out = np.zeros_like(_spec)

    # upwards shift in frequency: low band energies increase pitch
    # new high band
    _spec_out[:, deltaf:] = _spec[:, :n_bands-deltaf]
    # new low band
    _spec_out[:, :deltaf] = _spec[:, -deltaf:]
    return _spec_out


class RandFreqShift:
    def __init__(self,
                 do_rand_freq_shift=True,
                 Fshift=20):
        self.do_rand_freq_shift = do_rand_freq_shift
        self.Fshift = Fshift

    def __call__(self, spec):
        if self.do_rand_freq_shift:
            return random_freq_shift(spec, self.Fshift)
        else:
            return spec


def compander(_spec, comp_alpha=0.7):
    """
    simplest compander (compressor/expander) via spectrogram image contrast adjustment
    compression/expansion factor is drawn from a normal distribution limited to comp_alpha

    :param _spec:
    :param alpha:
    :return:
    """

    if comp_alpha < 1:
        # compress amplitude
        adjust = np.random.uniform(low=comp_alpha, high=1.0)
    elif comp_alpha > 1:
        # expand amplitude
        adjust = np.random.uniform(low=1.0, high=comp_alpha)
    else:
        # bypass
        adjust = 1

    # apply compander
    _spec_out = _spec * adjust

    return _spec_out


class Compander:
    def __init__(self,
                 do_compansion=True,
                 comp_alpha=0.7):
        self.do_compansion = do_compansion
        self.comp_alpha = comp_alpha

    def __call__(self, spec):
        if self.do_compansion:
            return compander(spec, self.comp_alpha)
        else:
            return spec


def gauss_noise(_spec, stdev):
    """
    add WGN of 0 mean and given standard deviation
    :param _spec:
    :param sigma_sq:
    :return:
    """
    t, f = _spec.shape
    awgn = np.random.normal(0, stdev, (t, f))
    # gauss = gauss.reshape(t, f)
    _spec_noise = _spec + awgn
    return _spec_noise


class GaussNoise:
    """
    add WGN
    instead of fixing the noise by parameter, we want to have variability
    stdev_gen is the stdev to generate the stdev of the noise
    """

    def __init__(self, stdev_gen=1e-3):
        self.stdev_gen = stdev_gen

    def __call__(self, spec):
        if self.stdev_gen > 0.0:
            stdev = np.random.uniform(0, self.stdev_gen)
            spec = gauss_noise(spec, stdev)
            spec = np.float32(spec)
        return spec


class RandomGaussianBlur:
    '''Apply Gaussian blur with random kernel size
    Args:
        max_ksize (int): maximal size of a kernel to apply, should be odd
        stdev_x (int): Standard deviation
    '''

    def __init__(self, do_blur=True, max_ksize=5, stdev_x=20):
        assert max_ksize % 2 == 1, "max_ksize should be odd"
        self.do_blur = do_blur
        self.max_ksize = max_ksize // 2 + 1
        self.stdev_x = stdev_x

    def __call__(self, spec):
        if self.do_blur:
            # define kernel_size randomly
            kernel_size = tuple(2 * np.random.randint(0, self.max_ksize, 2) + 1)
            # Gaussian blurring of the spectrogram
            blured_spec = cv2.GaussianBlur(spec, kernel_size, self.stdev_x)
            return blured_spec
        else:
            return spec


class TimeReversal:
    def __init__(self, do_time_reversal=True):
        self.do_time_reversal = do_time_reversal
        # to flip horizontally, meaning time-reversal in our setting
        self.flip_code = 0

    def __call__(self, spec):
        if self.do_time_reversal:
            spec = cv2.flip(spec, self.flip_code)
        return spec


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

EMSPy - HDMI ELECTROMAGNETIC EAVESDROPPING
@author: Louis de Benoist


It is a well known consequence of Maxwell's equations that electromagnetic waves propagate through space
whenever an electric current flows through a wire. HDMI cables, which transmit video signals, are no exception.
The predictable way in which pixels are transmitted leaves them susceptible to so-called Van Eck phreaking attacks,
the possibility of which has been known by governments since the 1960s. The purpose of this Python module is to regroup
some functions that will help you to design your own system capable of interpreting the waves emitted by an HDMI cable
and reproducing the transmitted images.

"""


import numpy as np
from scipy.signal import *
import matplotlib.pyplot as plt
import scipy.io, scipy.stats, scipy.linalg
import requests, io
from scipy import ndimage
from PIL import Image
from scipy.io.wavfile import write
from matplotlib.colors import hsv_to_rgb
np.random.seed(0)

########################################################################################################

# Hyperparameters

########################################################################################################

link = "https://www.cl.cam.ac.uk/teaching/2021/L314/assignment4a/"
filename_325 = "scene3-640x480-60-325M-64M-40M.dat"
filename_400 = "scene3-640x480-60-400M-64M-40M.dat"
filename_425 = "scene3-640x480-60-425M-64M-40M.dat"

noise_425 = "noise-640x480-60-425M-64M-40M.dat"

# dimensions
xt = 800
yt = 525

# center frequency of transmitter
Fc_400 = 400e6
Fc_425 = 425e6

Fp = 25.17e6 # baseline pixel-clock frequency
Fs = 64e6 # sampling frequency

########################################################################################################

# Signal Demodulation

########################################################################################################

def loadZ(num_frames=None, filename = filename_425):

    """

    load the IQ data at the specified file name

    """

    if num_frames != None:
        data = np.fromfile(filename, dtype='<f4', count = int(2*(num_frames + 1)/60 * Fs))
    else:
        data = np.fromfile(filename, dtype='<f4')

    I = data[0::2]
    Q = data[1::2]

    return I + 1j*Q


def getNumSamples(Z, Fs, Fr):

    """

    return the number of samples you get when resampling Z (which was originally sampled at Fs) at Fr

    """

    return int(len(Z) * (Fr/Fs))


def to_image(x, xt, yt):

    """

    return multiple image frames contained in x split depending on
    the values of xt and yt

    """

    video = []
    image = []
    line = []

    for i in range(len(x)):

        # if the image has 525 rows, then it is filled up and we move on to the next frame
        if len(image) % yt == 0 and len(image) !=0:
            video.append(image)
            image = []

        # if we have gone through 800 pixels, then we move on to the next line
        if len(line) % xt == 0 and len(line) !=0:
            image.append(line)
            line = []
        line.append(x[i])

    if len(video) == 0:
        video.append(image)

    return np.array(video)



def demodulate(Z):

    """

    demodulate the IQ signal

    """

    return np.abs(Z)

def shift_425(Z, shift = 250*xt+450):

    """

    shift for the 425 MHz test file

    """

    return Z[shift:]

def average_frames(Z, xt, yt, num_frames=3):

    """

    return the average of the frames in Z

    """

    frame_1 = get_frame(Z, frame_num=1, Fs=Fs, Fr=Fr, Fv=60, resample_Z=True)
    average = frame_1

    for i in range(2, num_frames+1):
        frame = get_frame(Z, frame_num=i, Fs=Fs, Fr=Fr, Fv=60, resample_Z=True)
        average += frame

    return average / num_frames


def combine_images(im1, im2):

    """

    Combine the top half of one image with the
    bottom half of another

    """

    images = np.array([im1, im2])
    return np.vstack(images)


########################################################################################################

# Estimating parameters via Autocorrelation and CrossCorrelation

########################################################################################################


def get_autocorrelation(signal, Fs, cutoff = None, demodulate = demodulate, Fv = 0, plot_mode = "num_samples", plot = True, k = 1):

    """

    returns autocorrelation of IQ signal.
    If the signal is too big, k can be used to use every k
    sample in the signal to save space.

    cutoff specifies the maximum number of samples to use for the autoccorelation:
    - None --> as much as the signal --> peaks will correspond to 1/frame rate
    - less than a frame --> peaks will correspond to 1/line rate

    """

    if Fv != 0:
        N = int(len(Z) / Fv)
    else:
        N = len(signal)

    signal = [signal[i] for i in range(0, N, k)]
    autocorr = plt.acorr(demodulate(signal), maxlags=len(signal)-1)[1]

    # normalize
    autocorr = np.array(autocorr) / (len(autocorr) * np.ones(len(autocorr)) - np.arange(len(autocorr)))
    autocorr = autocorr[:-len(autocorr)//5]

    if cutoff != None:
        cutoff = cutoff // k
        autocorr_mag = demodulate(autocorr)[len(signal): len(signal) + cutoff]
    else:
        autocorr_mag = demodulate(autocorr)[len(signal):]

    if plot:
        plt.figure(figsize=(20,4))

        if plot_mode == "time":
            x = [(i*k/Fs) for i in range(len(autocorr_mag))]
            plt.xlabel("Time (s)")

        if plot_mode == "num_samples":
            x = [(i*k) for i in range(len(autocorr_mag))]
            plt.xlabel("Number of samples")

        plt.plot(x, autocorr_mag)
        plt.ticklabel_format(style='plain')
        plt.show()

    return autocorr_mag


def get_frame_rate(Z, Fs, k, interval = [5000, 10000], plot = False):

    """

    returns frame rate estimate from autoccorrelation
    the two succesive peaks must be a distance that falls within the specified interval

    """

    autocorr_mag = get_autocorrelation(Z, Fs, plot = plot, k = k)


    # find the first two maxima that are at least delta samples apart
    maxima = np.argsort(-autocorr_mag)

    max_1 = maxima[0]
    max_2 = maxima[1]

    i = 2
    while k*np.abs(max_2-max_1) < interval[0] or k*np.abs(max_2-max_1) > interval[1]:
        max_2 = maxima[i]
        i+=1

    frame_sampled_length = k * np.abs(max_2 - max_1)

    Fv = 1/((frame_sampled_length)/Fs)
    return Fv


def get_line_rate(Z, Fs, Fv, k=1, interval = [500, 1000], delta = 500, plot = False):

    """

    returns line rate estimate from autoccorrelation
    the two succesive peaks must be less than delta samples away

    """

    autocorr_mag = get_autocorrelation(Z, Fs, Fv = Fv, plot = plot)

    # find the first two maxima that are at least delta samples apart
    maxima = np.argsort(-autocorr_mag)

    max_1 = maxima[0]
    max_2 = maxima[1]

    i = 2
    while k*np.abs(max_2-max_1) < interval[0] or k*np.abs(max_2-max_1) > interval[1]:
        max_2 = maxima[i]
        i+=1

    line_sampled_length = k * np.abs(max_2 - max_1)

    Fh = 1/((line_sampled_length)/Fs)

    return Fh


def get_correlation(signal1, signal2, Fs, demodulate = demodulate, cutoff = None, plot_mode = "num_samples", plot = True):

    """

    returns correlation between two IQ signals

    """

    correlated = correlate(demodulate(signal1), demodulate(signal2))[len(signal1):]

    if cutoff != None:
        correlated = correlated[:cutoff]

    if plot:
        plt.figure(figsize=(20,4))

        if plot_mode == "num_samples":
            x = [i for i in range(len(correlated))]
            plt.xlabel("Number of samples")

        if plot_mode == "time":
            x = [(i/Fs) for i in range(len(correlated))]
            plt.xlabel("Time (s)")

        plt.ticklabel_format(style='plain')
        plt.plot(x, correlated)
        plt.show()

    return correlated

def get_frame(Z, frame_num, Fs, shift_fun=shift_425, resample_Z=False, Fr = Fr, Fv=60):

    """

    returns the requested frame in Z. This will depend on the frame rate Fv and the sampling rate Fs.
    If resample_Z is true, then we also resample Z and interpolate using a parzen window. We then apply the
    requested shift so that the eavesdropped screen is in the top left corner of the resulting image.

    """

    if resample_Z:
        num_samples = getNumSamples(Z, Fs, Fr)
        Zr = resample(Z, num_samples, window = "parzen")
        Zr = shift_fun(Zr)
        start = int((Fr/Fv) * (frame_num - 1))
        end = int(start + (Fr/Fv))
        return Zr[start:end]

    else:
        Z = shift_fun(Z)
        start = int((Fs/Fv) * (frame_num - 1))
        end = int(start + (Fs/Fv))
        return Z[start:end]

def get_dimensions(Z, Fs, Fr, eps=80, interval1=[1000000, 2000000], interval2 = [1000, 2700], k=100):

    """

    Determines xt and yt by fintuning Fr = Fp
    First, it uses the peaks of the autocorrelation of Z to determine a guess for yt
    Then, after resampling, it tries to minimize the error for the new yt estimate (given some error
    threshold eps).
    It then returns the final xt estimate, the corresponding yt estimate, and Fr

    """

    # guess yt using autocorrelation peak
    Fv = get_frame_rate(Z, Fs, interval=interval1, k=k, plot=False)
    Fh = get_line_rate(Z, Fs, interval=interval2, k=k, Fv=Fv, plot=False)
    yt = Fh/Fv
    print("yt reference value:", yt)

    # initial estimate for xt based on the base-value of Fr=Fp
    xt_estimate = Fr/Fh
    print("initial xt guess:", xt_estimate)

    # resample and get new Fh estimate based on resampled
    num_samples = getNumSamples(Z, Fs, Fr)
    Zr = resample(Z, num_samples, window = "parzen")
    Fh = get_line_rate(Zr, Fr, interval=[500, 1000], k=k, Fv=Fv, plot=False)
    yt_estimate = Fh/Fv
    print("yt estimate:", yt_estimate)

    while np.abs(yt_estimate - yt) > eps:
        # update Fr to some random value close to the original
        Fr = Fr + 5*np.random.random()/100 * Fr
        num_samples = getNumSamples(Z, Fs, Fr)
        Zr = resample(Z, num_samples, window = "parzen")
        Fh = get_line_rate(Zr, Fr, interval=[500, 1000], k=k, Fv=Fv, plot=False)
        yt_estimate = Fh/Fv

    xt_estimate = Fr / Fh
    print("final xt guess", Fr/Fh)

    return xt_estimate, yt_estimate, Fr


def testing_example():

    """

    Example that displays the automatic image dimension deduction and Fr tuning

    """

    Z = loadZ(num_frames=4)
    return get_dimensions(Z, Fs, Fr)



########################################################################################################

# Enabling Pixels to Maintain Their Phase

########################################################################################################


def get_hsv(frame, demodulate=demodulate, filename = "HSV.jpg"):

    """

    make an HSV image from a given frame and return it as an RGB image (for display purposes)

    """

    mag = demodulate(frame)
    mag *= 1/mag.max()

    angle = np.pi + np.angle(frame)
    angle *= 1/angle.max()

    saturation = np.ones(angle.shape)

    hsv_image = np.dstack((angle, saturation, mag))
    image = hsv_to_rgb(hsv_image) # convert to rgb to display

    plt.figure(figsize=(20,10))
    plt.imshow(image, cmap="gray")
    plt.show()
    plt.imsave(filename, image, cmap="gray")
    return image


def spectrogram(Z, Fs):

    """

    plot a spectrogram of Z

    """

    plt.figure(figsize=(20,5))
    plt.specgram(Z, Fs=Fs)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.show()


def HSV_example():

    """

    Example that shifts Z with a complex phasor
    and displays the resulting HSV image

    """

    Z = loadZ(num_frames=4, filename=filename_425)

    spectrogram(Z, Fs)

    Fu = 2.2e7 - 200000 - 1550

    phasor = np.exp(-1.0j*2.0*np.pi*np.arange(0, len(Z))*(-Fu/Fs))
    Z = Z*phasor

    spectrogram(Z, Fs)

    frame = get_frame(Z=Z, frame_num=1, Fs=Fs, Fr=Fr, shift_fun=shift_425, resample_Z=True)
    image = to_image(frame, xt, yt)[0]
    hsv = get_hsv(image)


########################################################################################################

# Using different center frequencies

########################################################################################################


def example_400_center_frequency(Fu_425):

    """

    Example where we find the phasor shift for a center frequency of 400 using the
    phasor for a center frequency of 425 and then get the image as usual.

    """

    Z_400 = loadZ(num_frames=4, filename=filename_400)
    spectrogram(Z_400, Fs)
    Fu_400 = 25e06  - Fu_425
    phasor_400 = np.exp(-1.0j*2.0*np.pi*np.arange(0, len(Z))*(Fu_400/Fs))
    Z_400 = Z_400 * phasor_400
    spectrogram(Z_400, Fs)
    frame = get_frame(Z=Z_400, frame_num=1, Fs=Fs, Fr=Fr, shift_fun=lambda x:x, resample_Z=True)
    image = to_image(frame, xt, yt)[0]
    hsv = get_hsv(image)


########################################################################################################

# Simulating Longer Distances

########################################################################################################

def add_noise(Z, sigma=0.00001):

    """

    add Gaussian noise with variance sigma to Z

    """

    N = len(Z)
    eps = np.random.normal(loc=0, scale=sigma, size=(N, 1)) + 1j * np.random.normal(loc=0, scale=sigma, size=(N, 1))

    for i in range(N):
        Z[i] += eps[i]
    return Z

def estimate_noise_std(noise_filename = noise_425):

    """

    returns the sample standard deviation of the noise

    """

    noise = loadZ(filename=noise_filename)
    frame = get_frame(Z=noise, frame_num=1, Fs=Fs, Fr=Fr, shift_fun=lambda x: x, resample_Z=True)
    image = to_image(frame, xt, yt)[0]

    # average of real and complex std
    std = np.std(image)
    return  std

def test_noise_reconstruction(sig=3*1.2277526e-05):

    """

    test the reconstruction of Z after adding some complex noise

    """

    Z = loadZ(num_frames=4)
    Fu = Fp - 3400175 - 1550

    phasor = np.exp(-1.0j*2.0*np.pi*np.arange(0, len(Z))*(-Fu/Fs))
    Z = Z*phasor

    frame = get_frame(Z=Z, frame_num=1, Fs=Fs, Fr=Fr, shift_fun=shift_425, resample_Z=True)
    frame = add_noise(frame, sigma=sig)
    image = to_image(frame, xt, yt)[0]
    get_hsv(image)

"""
Custom Sound Library by Ryan Peters
    built for manual spectrogram processing

TOC:

"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import argparse
import sys 
from tqdm import tqdm

import numpy as np 
from numpy.lib.stride_tricks import as_strided
from scipy.io import wavfile
from scipy import signal, ndimage

import cv2

def splice_sound(sig,fs,duration=2.0):
    """Splice long audio files in equally sized sections of duration (seconds) long
    """
    splice_len = int(fs * duration)
    split_indices = np.arange(start=1,stop=sig.size,step=splice_len)
    return np.array_split(sig,split_indices)[1:]

def timeseries(x,fs):
    #Returns timeseries of x
    return np.linspace(0,len(x)/fs,num=len(x))

def display_spectrogram(Sxx):
    plt.imshow(Sxx,origin="lower")
    plt.show()

def gaussian_kernel(r,c,sigma=1):
    g = np.ones((r,c))

    for i in range(r):
        dist_y = np.square(int(r/2) - i)
        for j in range(c): 
            dist_x = np.square(int(c/2) - j)
            g[i,j] = (1 / (np.sqrt(2 * np.pi * sigma))) * np.exp(-((dist_x + dist_y)/(2 * np.square(sigma))))

    g = g / np.sum(g)

    return g

def DoG(r,c,sigma1=3,sigma2=5):
    return gaussian_kernel(r,c,sigma=sigma2) - gaussian_kernel(r,c,sigma=sigma1)

#Shift array by num (positive=right) padding with fill_value
def shift_array(arr, num, fill_value=0):
    num = int(num)
    #Source: https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def calculate_spectrogram(x,fs,window_type="hanning",log10=True):
    print(x.shape)
    NFFT = int(0.01 * fs)
    noverlap = int(0.005 * fs)
    nperseg = int((NFFT + noverlap) / 2)
    if nperseg > NFFT:
        nperseg = NFFT
    
    #print("Spectrogram Parameters:\nNFFT={}\nnOverlap={}\nnperseg={}\n".format(NFFT,noverlap,nperseg))

    #Returns tuple (t,f,Sxx) where:
    #t = time bins
    #f = frequency bins 
    #Sxx = intensity of a frequency at a time (Sxx[t,f])
    t,f,Sxx = signal.spectrogram(x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=NFFT,
        window=signal.get_window(window_type, nperseg),
        detrend=False)
    
    if log10:
        Sxx = np.log10(Sxx)

    return t,f,Sxx

def butter_lowpass_filter(data, cutoff, fs, order=5):
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def blur1D(x,ksize=3,sigma=1):
    g = np.zeros(ksize)
    for i in range(ksize):
        g[i] = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((int(ksize/2) - i) ** 2) / (2 * sigma**2))

    #Normalize between [0,1)
    g = g / g.sum()

    x = np.convolve(x,g,mode="same")

    return x

def blur2D(Sxx,ksize=3,sigma=1):
    g = np.zeros(ksize)
    for i in range(ksize):
        g[i] = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((int(ksize/2) - i) ** 2) / (2 * sigma**2))

    #Normalize between [0,1)
    g = g / g.sum()

    #Blur by convolution
    x = cv2.filter2D(Sxx,-1,g)
    x = cv2.filter2D(Sxx,-1,g.T)

    return x

def smooth(x,window=9):
    if window % 2 == 0:
        print(f"Smooth window must be odd. Reducing to {window-1:}")
        window -= 1

    x = np.pad(np.copy(x), window//2, mode="edge")
    x = np.convolve(x, np.ones(window), mode="valid") / window
    return x

def vertical_edge_detection(img):
    img = blur2D(img,ksize=5,sigma=2)

    sobel_y_2order = np.array([1,4,6,4,1,0,0,0,0,0,-2,-8,-12,-8,-2,0,0,0,0,0,1,4,6,4,1]).reshape(5,5)
    vertical_edges = cv2.filter2D(img, -1, sobel_y_2order)
    
    return vertical_edges

def create_vowel_template(vowel_type="centroid",duration=1.0, height=32, nblurs=3, ksize=5, sigma=1, normalize=True, enhance=True, maxfreq=5000, display=False):
    """
    Create a generic spectrogram of a vowel used for cross-correlation in template matching
    There are several types of vowels that can be created

    Vowel_type:
        centroid: use a schwa
        extremes: use 4 vowels at the extremes of the vowel space [i,u,ae,a]
        extremes_combined: combine the extremes into one vowel
    """
    
    soundfiles_path = "soundfiles/"
    f,((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3)
    
    #Determine vowel template based on vowel_type
    template = []
    if vowel_type == "centroid":
        #Get basic vowel (shcwa)
        fs,schwa = wavfile.read(soundfiles_path + "schwa.wav")
        template.append(schwa)
    elif vowel_type == "extremes" or vowel_type == "extremes_combined":
        #fs=44.1kHz per vowel
        fs,i = wavfile.read(soundfiles_path + "i.wav")
        _,a = wavfile.read(soundfiles_path + "a.wav")
        _,u = wavfile.read(soundfiles_path + "u.wav")
        _,ae = wavfile.read(soundfiles_path + "ae.wav")
        template.append(i)
        template.append(ae)
        template.append(u)
        template.append(a)
    

    #How much of vowels to use as template (float = percent, int = num points)
    if isinstance(duration, float):
        if 0 < duration <= 1:
            half_dur = int(duration * len(template[0]) / 2)
        else:
            print("Duration (float) must be greater than 0 and less than 1")
            exit()
    elif isinstance(duration,int):
        if duration > 0:
            half_dur = duration // 2
        else:
            print("Duration (int) must be greater than 0")
            exit()
    else:
        print("Duration must be an int or float")
        exit()

    for i,vowel in enumerate(template):
        middle = int(len(vowel)/2)
        
        template[i] = vowel[middle-half_dur:middle+half_dur]
    


    #Calculate schwa spectrogram
    for i,vowel in enumerate(template):
        f,t,vowel = calculate_spectrogram(vowel,fs,window_type="hanning", log10=True)

        #Crop to 0-5kHz
        vowel = crop_spectrogram(vowel,f[-1],spectrogram_max_freq=maxfreq)
        ax1.imshow(vowel,origin="lower")

        #Blur
        for _ in range(nblurs):
            vowel = blur2D(vowel,ksize=ksize,sigma=sigma)
        ax2.imshow(vowel,origin="lower")

        if enhance:
            vowel = vertical_edge_detection(vowel)
        ax3.imshow(vowel,origin="lower")

        #Resize to 32 height
        vowel = cv2.resize(vowel, (vowel.shape[1],height),interpolation=cv2.INTER_LINEAR)
        ax4.imshow(vowel,origin="lower")
        if normalize:
            vowel = normalize_array(vowel)

        ax5.imshow(vowel,origin="lower")

        if enhance: #Edge detection "flips" values, so invert them
            vowel = vowel.max() - vowel 
        ax6.imshow(vowel,origin="lower")

        if display:
            plt.show()
        else:
            plt.close()
        
        template[i] = vowel

    if vowel_type == "extremes_combined":
        combined_vowel = np.zeros_like(template[0])
        for vowel in template:
            combined_vowel += vowel
        combined_vowel = normalize_array(combined_vowel)
        template = [combined_vowel]

        if display:
            plt.imshow(combined_vowel,origin="lower")
            plt.show()

    return template

def crop_spectrogram(Sxx, max_freq, spectrogram_max_freq=5000):
    crop_height = int((spectrogram_max_freq / max_freq) * Sxx.shape[0])
    Sxx = Sxx[0:crop_height,:]
    return Sxx

def normalize_array(x,zero_center=False):
    #Unity-Based Normalization (ie set range to [0,1])
    if x.size != 0:
        x = x - x.min()
        x  = x / x.max()

    return x

def threshold(x,t):
    x = np.copy(x)
    x[x<t] = 0
    return x

def threshold_around_peak(r,window_len=20):
    #Take response with peaks and threshold around the largest peak
    #Could be useful one day

    #Take 1st derivative --> absolute value --> 2 x smooth --> normalize --> square --> threshold
    d_r = np.abs(np.diff(r)).flatten()
    d_r = smooth(d_r, window=19)
    d_r = normalize_array(d_r)
    d_r = np.square(d_r)
    mask = create_mask(d_r, 0.05, comparison="greater")
    return mask

def get_mask_regions(m):
    #Returns 2d-array of [start,end] for chunks in the mask
    regions = []
    start = end = 0

    for i in range(len(m)-1):
        if m[i] == 0 and m[i+1] == 1: #Begin of chunk
            start = i+1
        elif m[i] == 1 and m[i+1] == 0: #End of a chunk
            end = i+1   #makes exclusive indexing easier
            regions.append([start,end])
            start = end = 0

    if start: #If ending in region
        regions.append([start,len(m)])


    regions = np.array(regions)

    return regions


def create_mask(x,t,comparison="greater"):
    mask = np.copy(x)
    if comparison.lower() == "greater":
        mask[mask > t] = 1
    elif comparison.lower() == "less":
        mask[mask < t] = 1

    mask[mask != 1] = 0
    return mask

def ssd(input_image, template, valid_mask=None):
    #https://stackoverflow.com/questions/17881489/faster-way-to-calculate-sum-of-squared-difference-between-an-image-m-n-and-a

    
    if valid_mask is None:
        valid_mask = np.ones_like(template)
    total_weight = valid_mask.sum()
    window_size = template.shape

    # Create a 4-D array y, such that y[i,j,:,:] is the 2-D window
    #     input_image[i:i+window_size[0], j:j+window_size[1]]
    y = as_strided(input_image,
                    shape=(input_image.shape[0] - window_size[0] + 1,
                           input_image.shape[1] - window_size[1] + 1,) +
                          window_size,
                    strides=input_image.strides * 2)

    # Compute the sum of squared differences using broadcasting.
    ssd = ((y - template) ** 2 * valid_mask).sum(axis=-1).sum(axis=-1)

    #ssd = ssd.flatten()[template.shape[1]:-template.shape[1]]
    return ssd

def intensity(sig,fs,display=False):
    """Calculate intensity of waveform sig and returns derivative of it
    Parameters:
        sig: waveform to get intensity of 
        fs: sampling frequency of sig
        display: if True, display the intensity and derivative against waveform
    """
    window_len = int(fs * 0.1)
    window_len += (1 if window_len % 2 == 0 else 0)
    
    #Take absolute value then integrate
    sig_intensity = np.pad(np.abs(sig), window_len // 2, mode="edge")
    sig_intensity = np.convolve(sig_intensity, np.ones(window_len), mode="valid")
    sig_intensity = normalize_array(sig_intensity)

    #Smooth
    sig_intensity = smooth(sig_intensity, window=window_len)
    d_sig_intensity = smooth(np.abs(np.diff(sig_intensity, append=sig_intensity[-1])),window=window_len)
    d_sig_intensity = normalize_array(d_sig_intensity)    

    if display:
        f, (ax1,ax2) = plt.subplots(2,1)
        ax1.plot(timeseries(sig,fs),sig)
        ax2.plot(sig_intensity)
        ax2.plot(d_sig_intensity)
        plt.show()

    return sig_intensity,d_sig_intensity

def zero_crossing_rate(sig,fs,unit=0.1, normalize_range=True, display=True):
    """Determines the zero crossing rate of the signal 
    unit: crossings per unit of time (0.1 seconds default)
    """

    window_len = int(fs * 0.1)
    zcr = np.zeros_like(sig)
    zcr[np.where(np.diff(np.sign(sig)))[0]] = 1 #set zero crossings to one
    zcr = np.convolve(zcr, np.ones(window_len), mode="same") #window to integrate

    #smooth for peak detection
    smooth_window = int(fs * 0.05)
    print(smooth_window)
    for _ in range(2):
        zcr = np.convolve(zcr, np.ones(smooth_window), mode="same") / smooth_window

    if normalize_range:
        zcr= normalize_array(zcr)

    d_zcr = np.diff(zcr,append=zcr[-1])
    d_zcr = np.abs(d_zcr)
    d_zcr = np.convolve(d_zcr, np.ones(10),mode="same") / 10 #smooth
    d_zcr = normalize_array(d_zcr)
    dr_peaks,_ = signal.find_peaks(d_zcr,
        height=0.05,
        prominence=0.075)

    if display:
        f,(ax1,ax2) = plt.subplots(2,1)
        ax1.plot(timeseries(sig,fs),sig)
        ax1.set_title("Signal Waveform")
        ax2.plot(zcr)
        ax2.plot(d_zcr)
        ax2.plot(dr_peaks,d_zcr[dr_peaks],"x")
        ax2.set_title("Zero-Crossing Rate per {} Seconds".format(unit))
        plt.show()

    return zcr, d_zcr


def correlate(a,b,normalize=True,normalize_range=True, square_response=True, convolve=False, use_ssd=False, is_image=False):
    """
    Custom correlation wrapper for signal's own correlate function
    Guarantees that the 2 input matrices are the same height so correlation coefficient (r) is 1-d
    Pads matrix a on LR edges with itself so r at the end is flat

    Parameters:
        a,b: correlate matrix b against matrix a
        normalize: if True, normalize a and b by subtracting mean and dividing by std deviation then setting to range [0,1]
        normalize_range: if True, set r to [0,1]
        square_response: if True, square r
        convolve: if True, flip b by UP-DOWN and LEFT-RIGHT before correlation
        use_ssd: if True, use sum of square differences instead of correlation for a,b 
        is_image: ignore same height requirements and pad UD as well
    """
    if a.shape[0] != b.shape[0]:
        if not is_image:
            sys.exit(f"Matrix heights of {a.shape[0]:} and {b.shape[0]:} must be the same.")
    
    if a.shape[1] < b.shape[1]:
        sys.exit("Matrix a {} must have a greater width than Matrix b {}.".format(a.shape,b.shape))

    if normalize: 
        a = (a - a.mean()) / a.std()
        b= (b - b.mean()) / b.std()
        a = normalize_array(a)
        b = normalize_array(b)

    if convolve:
        b = np.flipud(np.fliplr(b))

    padding = b.shape[1] // 2 
    if is_image:
        a = np.pad(a,((padding,padding),(padding,padding)),mode="edge")
    else:
        a = np.pad(a,((0,0),(padding,padding)),mode="edge")
    
    if use_ssd:
        r = ssd(a,b)
    else:
        #Correlate
        r = signal.correlate2d(a,b,mode="valid")

    if not is_image:
        r = r.flatten()

    if normalize_range:
        r = normalize_array(r)

    if square_response:
        r = np.square(r)

    return r

def peak_envelope(x,prominence=0.01):
    envelope = np.copy(x)

    #Find peaks
    envelope = normalize_array(envelope).flatten()
    peaks,_ = signal.find_peaks(envelope,prominence=prominence)# 0.025    

    #Envelope the peaks
    envelope = np.interp(np.arange(envelope.size),peaks,envelope[peaks]) 

    #Smooth envelope
    avg_peak_distance = int(np.diff(peaks).mean())
    envelope = smooth(envelope,window=avg_peak_distance)

    return envelope

def peak_expansion(x,fs,duration=0.1,threshold=0.5,prominence=0.25):
    peaks, properties = signal.find_peaks(x,
            height=threshold,
            distance=int(fs*0.015),
            prominence=prominence)

    half_duration = int(fs * duration/2)
    x_pe = np.zeros_like(x)
    for p in peaks:
        x_pe[p-half_duration:p+half_duration] = 1

    return x_pe


def extract_vowels(sig,template,fs,lp_cutoff=4500,maxfreq=5000, normalize=True,display=False):
    """
    Perform cross-correlation between spectrograms of signal and template
    Note: template is an array of vowels (1 or 4 in length)
    """ 
    #Convert signal to spectogram
    f, t, Sxx = calculate_spectrogram(sig,fs,
        window_type="hanning",
        log10=True)

    if normalize:
        Sxx = np.square(normalize_array(Sxx))


    #reduce freqs displayed to 0-5000Hz
    Sxx = crop_spectrogram(Sxx, f[-1], spectrogram_max_freq=maxfreq)
    
    #Resize 
    Sxx = blur2D(Sxx,ksize=5,sigma=1)
    Sxx = cv2.resize(Sxx, (Sxx.shape[1]*2,template[0].shape[0]),interpolation=cv2.INTER_LINEAR)


    #Correlate signal with template to get response
    r = np.zeros((len(template),Sxx.shape[1]))
    for i,vowel in enumerate(template):
        r[i] = correlate(Sxx, vowel, normalize=normalize, square_response=True)    
    r = np.sum(r,axis=0).flatten()
    

    #Find peaks of r and build regions around them
    distance = int(0.10 * fs * len(r) / len(sig))
    peaks, properties = signal.find_peaks(r,
            height=0.3,
            distance=distance,
            prominence=0.075)  

    print("Peaks found: {}\nPeak X-Coordinates: {}\n".format(len(peaks),peaks))

    #Display peaks
    if display:
        print("Displaying Spectrogram of Signal")
        fig,(ax1,ax2,ax3) = plt.subplots(3,1)
        fig.tight_layout(h_pad=1.1)

        ax1.plot(timeseries(sig,fs),sig)
        ax1.set_title("Waveform")
        ax1.set_xlim(0,len(sig)/fs)
        ax2.imshow(Sxx,origin="lower",aspect="auto",interpolation="nearest")
        ax2.set_title("Spectrogram")
        ax3.plot(r,color="b")
        ax3.plot(peaks,r[peaks],"x")
        ax3.set_title("Response from Template Matching")
        ax3.set_xlim(0,len(r))
        plt.show()


    #Loop through the peaks, and perform partial-autocorrelation in a window around peak 
    #Dropoffs in response indicate vowel boundaries 
    #A slow dropoff may indicate diphthong or sonorant-vowel boundary
    region_length = int(0.4 * (Sxx.shape[1] / len(sig)) * fs) #smaller region length = faster but possible to miss an extended vowel
    peak_window_len = 5
    regions = [[p-region_length,p+region_length] for p in peaks]
    vowels = []

    
    for (start,end),p in zip(regions,peaks):

        #Select windows of spectrogram around peak ]
        region_middle_index = ((start+end) // 2 ) - max(0,start)#used to denote position of original peak within region_Sxx
        region_Sxx = np.copy(Sxx[:,max(0,start):min(end,Sxx.shape[1])])
        peak_Sxx = np.copy(Sxx[:,max(0,p-peak_window_len):min(p+peak_window_len,Sxx.shape[1])])
        print("\nWindowing peak {} from {} to {}".format(p,start,end))
        print("Region width: {}. Peak location: {}".format(Sxx.shape[1],region_middle_index))

        
        #Note that both spectrograms already blurred from earlier
        #"Partial" (not in the general sense) autocorrelation of vowel against its surrounding spectrogram
        #looking for dropoffs to denote vowel boundary
        r_acf = correlate(region_Sxx, peak_Sxx, 
            normalize=normalize,
            normalize_range=False,
            square_response=True,
            convolve=False,
            use_ssd=False) #should probably use SSD 
        

        #TODO: changed, check results
        r_acf = smooth(r_acf,window=19)
        #r_acf = np.convolve(r_acf,np.ones(20),mode="same") / 20 #smooth 

        #Set original peak (from r) to 1 (array is no longer set from [0,1] but results in better thresholding)
        r_acf /= r_acf[region_middle_index]


        #Threshold the response and create a mask
        m = create_mask(r_acf,0.6,comparison="greater")


        #Find the endpoints of the thresholded region
        vowel_start = len(m)
        vowel_end = 0
        counter = 0
        m_zero = np.where(m==0)[0]
        while not (vowel_start < region_middle_index < vowel_end): #Find region where peak exists
            vowel_start = m_zero[np.where(np.diff(m_zero) > 1)][counter] #Find where region goes from 0 to 1
            vowel_end = m_zero[np.where(m_zero == vowel_start)[0] + 1][0]
            counter += 1
        
        endpoints = np.array([vowel_start,vowel_end])
        print("Endpoints from Correlation: {}".format(endpoints))

        #Cross-check with segmentation derived by gradient of r_acf to see if segmentation occurs within the region
        #if so, update the region endpoints accordingly
        #TODO: take multiple autocorrelations in the vowel peak in case peak is weird
        #TODO: either that or just use segment_speech to determine vowel regions
        r_acf = smooth(r_acf,window=9)
        dr = np.diff(r_acf,append=r_acf[-1])
        dr = np.abs(dr)
        dr = smooth(dr,window=9)
        #dr[dr < 0] = 0
        dr = normalize_array(dr)
        dr_peaks,properties = signal.find_peaks(dr,
            height=0.05,
            prominence=0.1)


        for dr_peak in dr_peaks:
            if endpoints[0] < dr_peak < endpoints[1]:
                temp = np.copy(endpoints)
                if dr_peak < region_middle_index:
                    endpoints[0] = dr_peak
                    print("Segmentation changed lower bound from {} to {}".format(temp[0],dr_peak))
                elif dr_peak > region_middle_index:
                    endpoints[1] = dr_peak
                    print("Segmentation changed upper bound from {} to {}".format(temp[1],dr_peak))

        #Create mask for display purposes only
        vowel_selected = np.zeros_like(r_acf)
        vowel_selected[endpoints[0]:endpoints[1]] = 1

        #Interpolate to original signal
        endpoints += max(0,start)
        endpoints *= len(sig)
        endpoints = (endpoints / Sxx.shape[1]).astype(np.int64)

        vowel_minimum_duration = int(0.025 * fs)
        if (endpoints[1] - endpoints[0]) > vowel_minimum_duration:
            vowels.append(endpoints.astype(np.int32))
        else:
            print("Vowel region [{},{}] of length {} not added due to minimum duration {} required".format(endpoints[0],endpoints[1],np.diff(endpoints)[0],vowel_minimum_duration))

        
        


        #plot results for each peak
        if display:
            fig,(ax0,ax1,ax2,ax3) = plt.subplots(4,1)
            fig.tight_layout(h_pad=1.1)
            ax0.imshow(peak_Sxx,origin="lower")
            ax0.set_title("Vowel Peak Template for Autocorrelation")
            ax1.imshow(region_Sxx,origin="lower",aspect="auto",interpolation="nearest")
            ax1.set_title("Vowel Surrounding Region")
            ax2.plot(r_acf)
            ax2.plot(m,color="m")
            ax2.set_xlim(0,len(r_acf))
            ax2.set_title("Autocorrelation Coefficient of Peak Template correlated to its Region it resides in")
            ax3.plot(dr,color="m",label="")
            ax3.plot(dr_peaks,dr[dr_peaks],"x",color="y")
            ax3.set_xlim(0,len(dr))
            ax3.set_title("Segmentation from Running Difference")
            ax3.plot(vowel_selected,color="b",label="Final Vowel Region")
            ax3.legend()
            plt.show()

    for i,(s,e) in enumerate(vowels):
        vowels[i] = [s+peak_window_len,e-peak_window_len]

    return vowels


def voicing_detection(sig,fs,f=None,spectrogram_max_freq=5000,height=32,window_len=5):
    #Convert to spectrogram
    f,t,Sxx = calculate_spectrogram(sig,fs,log10=False) #log10 must be false
   
    #Blur --> normalize [0,1) --> crop spectrogram
    Sxx = crop_spectrogram(normalize_array(blur2D(Sxx,ksize=5)),f[-1],spectrogram_max_freq=spectrogram_max_freq)

    #Resize to height
    Sxx = cv2.resize(Sxx,(Sxx.shape[1],height),cv2.INTER_LINEAR)

    #Expand a sobel horizontal edge detection kernel to size of (spectrogram height,window_len)
    sobel_x = np.array([[1,0,-2,0,1],
        [4,0,-8,0,4,],
        [6,0,-12,0,6],
        [4,0,-8,0,4],
        [1,0,-2,0,1]]).astype(np.float32)
    edge_kernel = cv2.resize(sobel_x,(window_len,Sxx.shape[0]),interpolation=cv2.INTER_LINEAR)

    #Edge detection: (1d response since heights are the same)
    padding = edge_kernel.shape[1] // 2
    r = signal.convolve(np.pad(Sxx,((0,0),(padding,padding))),edge_kernel,mode="valid")


    #CREATE THE VOICING MASK

    #Find peaks in response
    r = normalize_array(r).flatten()
    peaks,_ = signal.find_peaks(r,prominence=0.005) 

    #Create mask
    voicing_mask = np.zeros_like(r) #Assume everything is voiceless

    #Use peak spacing to determine whether space between peaks is voiced or not
    #Since vocal folds vibrate at the same rate, we can reliably use this method
    avg_peak_distance = np.diff(peaks).mean()
    peak_dist_threshold = avg_peak_distance * 2.5
    
    #If distance between peaks is small (compared to average spacing), mark it as voiced
    for i in range(len(peaks)-1): 
        if peaks[i+1] - peaks[i] < peak_dist_threshold:
            voicing_mask[peaks[i]:peaks[i+1]] = 1

    #Chunk mask into regions in relation to original sig length
    regions = get_mask_regions(voicing_mask)

    #Remove small impulse regions (like plosive bursts)
    for i,(s,e) in enumerate(regions):
        if e-s < peak_dist_threshold:
            regions[i] = [0,0]
            #regions = np.delete(regions, i, 0)
            voicing_mask[s:e] = 0
            print("Removing region [{}:{}] with length of {}".format(s,e,e-s))
    

    #Interpolate regions start and ends to original signal length
    scale = len(sig) / len(voicing_mask) 
    regions = (regions * scale).astype(np.int32)

    #Plot outputs
    f,(ax1,ax2) = plt.subplots(2,1)
    f,t,Sxx = calculate_spectrogram(sig,fs,log10=False)
    ax1.pcolormesh(t, f, Sxx, cmap=cm.gray_r, shading='gouraud', norm=LogNorm())
    ax1.set(xlabel='Time [sec]', ylabel='Frequency [Hz]')
    ax1.set_ylim(0,5000)

    ax2.plot(r,color="g")
    ax2.plot(peaks,r[peaks],"x")
    ax2.set_xlim(0,len(r))
    ax2.plot(voicing_mask,color="m")
    plt.show()
 

    voicing = np.zeros_like(sig)
    for s,e in regions:
        voicing[s:e] = sig[s:e]
    
    f,(ax1,ax2) = plt.subplots(2,1)
    f,t,Sxx = calculate_spectrogram(voicing,fs,log10=False)
    ax1.pcolormesh(t, f, Sxx, cmap=cm.gray_r, shading='gouraud', norm=LogNorm())
    ax1.set(xlabel='Time [sec]', ylabel='Frequency [Hz]')
    ax1.set_ylim(0,5000)

    ax2.plot(voicing,color="g")
    ax2.set_xlim(0,len(voicing))
    plt.show()


    

    return regions

def segment_speech(sig, fs, stride_size, spectrogram_height=32, settings="short window",display=True):
    """
    Segment sound using stable regions on a spectrogram to determine where a new sound begins

    """
    #TODO: find stable regions with singular pass then clean up boundaries with correlations ONLY on stable regions
    #       or else edges will be boundaried
    #       Possibility: may lose impulses like plosives? --> threshold derivative lower?
    settings_dict = {"long window": {"window len":25, "prominence":0.01, "peak height":0.0, "peak width":0.0},
        "short window": {"window len":5,"prominence":0.0001,"peak height":0.0001, "peak width":0.0},
        "testing": {"window len":3, "prominence":0.001, "peak height":0.0001, "peak width":0.0}
    }

    #Grab dictionary of settings
    settings = settings_dict[settings]
    print("Settings Used: {}".format(settings))


    #Get spectrogram of signal, set to range [0,1] and square to highlight frequencies
    f,t,Sxx = calculate_spectrogram(sig,fs,
        log10=True)

    #TODO: something weird here occuring
    #Sxx = np.square(normalize_array(Sxx))
    #Sxx = np.square(Sxx)
    Sxx = normalize_array(Sxx)

    

    #Square --> Blur --> crop spectrogram to 5k Hz --> range to [0,1]
    Sxx = normalize_array(crop_spectrogram(blur2D(np.square(Sxx),ksize=5),f[-1],spectrogram_max_freq=5000))


    #Resize to input height parameter
    Sxx = cv2.resize(Sxx, (Sxx.shape[1],spectrogram_height), interpolation=cv2.INTER_LINEAR)

    #Create array of stride indexes for correlation window origins
    half_window = settings["window len"] // 2
    strides = np.arange(start=half_window,stop=Sxx.shape[1]-half_window,step=stride_size)
    half_window = settings["window len"] // 2
    
    #Allocate array of correlations
    correlations = np.zeros((strides.size,Sxx.shape[1]))

    for correlations_index,stride_index in enumerate(tqdm(strides[:-1],desc="Correlations:")):
        
        region = Sxx[:,stride_index-half_window:stride_index+half_window+1]
        #Correlate region against entire spectrogram
        r = correlate(Sxx, region, 
            normalize=True, 
            normalize_range=False, 
            square_response=True,
            convolve=False,
            use_ssd=True,
            is_image=False)
            

        #Store in convolutions array
        correlations[correlations_index,:] = smooth(r,window=9)

    #Sum inidividual convolutions into a single array
    correlation_sum = np.sum(correlations, axis=0)

    #Generalize by region window size and number of correlations 
    correlation_sum /= (region.size * correlations.shape[0])

    #Smooth correlation_sum to remove impulse-peaks
    correlation_sum = smooth(correlation_sum, window=9)


    #Use derivative to find boundaries (peak in derivative = large change = boundary)
    dr = np.diff(correlation_sum,append=correlation_sum[-1])
    dr = np.abs(dr)
    dr = smooth(dr,window=9)

    #dr = normalize_array(dr)
    dr_peaks,properties = signal.find_peaks(dr,
        height=settings["peak height"],
        prominence=settings["prominence"],
        width=settings["peak width"])

    #Use intensity to determine where peaks can exist (only when signal is present from 0-5000Hz)
    sxx_intensity = peak_envelope(np.sum(Sxx,axis=0)).flatten() 
    sxx_intensity = smooth(sxx_intensity,window=9)
    sxx_intensity /= Sxx.shape[0]
    intensity_mask = create_mask(sxx_intensity,t=0.005,comparison="greater")

    dr_peaks[np.where(intensity_mask[dr_peaks] == 0)] = 0
    #dr_peaks = dr_peaks[dr_peaks != 0]

    #Add segmentation boundaries at mask region boundaries
    new_boundaries = np.where(np.diff(intensity_mask))
    dr_peaks = np.sort(np.append(dr_peaks,new_boundaries))
    

    if display:

        #print
        print(f"Boundaries: {len(dr_peaks):}")
        for i,(drp,d,prom,w) in enumerate(zip(dr_peaks, np.round(dr[dr_peaks],4),np.round(properties["prominences"],4),properties["right_ips"]-properties["left_ips"])):
            print(f"Peak: {i+1:<4}  | Peak X Loc: {drp:<4}   | Height: {d:>2.5f}  | Prominence: {prom:>2.5f}  | Width: {w:>2.5f}")
        print()

        #plot
        fig,(ax0,ax1,ax2,ax3) = plt.subplots(4,1)
        fig.tight_layout(h_pad=1.1)
        
        for p in dr_peaks:
            ax0.axvline(x=p,color="w")
        ax0.set_title("Spectrogram from 0 to 5000 Hz with height of {}".format(Sxx.shape[0]))
        ax0.set_xlim(0,Sxx.shape[1])

        divider = make_axes_locatable(ax0)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax0.imshow(Sxx,origin="lower",cmap=cm.viridis)
        fig.colorbar(im, cax=cax, orientation='vertical')
        
        ax1.set_title("Correlation by SSD")
        ax1.set_xlim(0,Sxx.shape[1])
        ax1.plot(correlation_sum)

        ax2.set_title("Correlation Derivative")
        ax2.set_xlim(0,Sxx.shape[1])
        ax2.plot(dr)
        ax2.plot(dr_peaks, dr[dr_peaks],"x")

        ax3.set_title("Spectrogram Intensity and Mask for Peak Location Thresholding")
        ax3.set_xlim(0,Sxx.shape[1])
        ax3.plot(sxx_intensity,color="m",label="Sxx Intensity")
        ax3.plot(intensity_mask*sxx_intensity.max(), color="b", label="Intensity Mask")
        ax3.legend(loc="upper right")

        plt.show()

    #Interpolate dr_peaks to signal length
    dr_peaks = np.multiply(dr_peaks, sig.shape[0] / Sxx.shape[1])


    return dr_peaks.astype(np.int64)

def match_filter(sig,fs,template,duration,height=32):
    """A match filter algorithm that ATTEMPTS TO account for length/speed difference in speech
    by using segmentation to make regions the same length

    Currently does not work :(. 
    """
    #TODO: Fix segmentation algorithm per its todo
    #TODO: Make sure template used is one region long (theoretically could be 2)

    f,t,Sxx = calculate_spectrogram(sig,fs,log10=True)
    f2,t2,Sxx_template = calculate_spectrogram(template,fs,log10=True)

    
    Sxx = crop_spectrogram(normalize_array(blur2D(Sxx,ksize=5)),f[-1],spectrogram_max_freq=5000)
    Sxx_template = crop_spectrogram(normalize_array(blur2D(Sxx_template,ksize=5)),f2[-1],spectrogram_max_freq=5000)

    #make distinct regions of sig and template to be same duration
    boundaries_sig = segment_speech(sig,fs,200,
            spectrogram_height=height,
            window_len=5,
            peak_prominence=0.02,
            display=False)
    
    boundaries_sig  = np.multiply(boundaries_sig, Sxx.shape[1] / sig.size).astype(np.int64) + 1

    sections_sig = list(np.hsplit(Sxx,boundaries_sig))
    section_size = int(fs * 0.05 * Sxx.shape[1] // sig.size)
    
    for i,sect in enumerate(sections_sig):
        if sect.size != 0:
            sections_sig[i] = cv2.resize(np.atleast_2d(sect),(section_size,Sxx.shape[0]),cv2.INTER_LINEAR)

    Sxx_same_length = np.hstack(sections_sig)


    boundaries_template = segment_speech(template,fs,200,
            spectrogram_height=height,
            window_len=5,
            peak_prominence=0.02,
            display=False)

    boundaries_template  = np.multiply(boundaries_template, Sxx_template.shape[1] / template.size).astype(np.int64) + 1
    sections_template = list(np.hsplit(Sxx_template,boundaries_template))
    
    for i,sect in enumerate(sections_template):
        if sect.shape[1] != 0:
            sections_template[i] = cv2.resize(np.atleast_2d(sect),(section_size,Sxx.shape[0]),cv2.INTER_LINEAR)

    
    Sxx_template_same_length = np.hstack(sections_template)
    

    #Correlate
    r = correlate(Sxx_same_length, Sxx_template_same_length, 
            normalize=True, 
            normalize_range=False, 
            square_response=True,
            convolve=False)

       

    #Display

    ff,(ax1,ax2,ax3) = plt.subplots(3,1)
    
    ax1.imshow(Sxx_same_length,origin="lower")
    ax1.set_ylim([0,50])
    ax2.imshow(Sxx_template_same_length,origin="lower")
    ax2.set_ylim([0,50])
    ax3.plot(r)
    ax3.set_xlim([0,len(r)])
    plt.show()

def segment_image(img, window_size=3, stride_size=1, display=True):
    #Remove noise from image

    smooth_kernel = np.ones((5,5),np.float32)/25
    sharpen_kernel = np.array([-1,-1,-1,-1,9,-1,-1,-1,-1],np.float32).reshape(3,3)
    img = cv2.bilateralFilter(img,5,75,75)
    #img = cv2.filter2D(img,-1,smooth_kernel) #try bilateral filter cv2.bilateralfilter
    img = cv2.resize(img,(100,100),interpolation=cv2.INTER_LINEAR)
    #img = cv2.filter2D(img,-1,sharpen_kernel)


    half_window = window_size // 2
    strides_horizontal = np.arange(start=half_window,stop=img.shape[1]-half_window,step=stride_size)
    strides_vertical = np.arange(start=half_window,stop=img.shape[0]-half_window,step=stride_size)

    num_correlations = strides_vertical.size + strides_horizontal.size
    window_total_size = window_size ** 2

    #Allocate correlations matrices
    correlation_channel_sums = np.zeros((img.shape[0],img.shape[1],img.shape[2]),dtype=np.int64)
    r = np.zeros((img.shape[0],img.shape[1]),dtype=np.float32)

    def image_derivative(img):
        padded = np.pad(img.astype(np.float32),((0,1),(0,1)),mode="edge")
        dx = np.abs(np.diff(padded[:-1,:]))
        dy = np.abs(np.diff(padded.T[:-1,:]))
        d = dx + dy.T
        d = cv2.filter2D(normalize_array(d) * 255,-1,smooth_kernel)
        return d
    

    for channel in range(img.shape[2]):
        print(f"Begining Channel {channel:}:")
        correlation_index_counter = 0
        for s_v in strides_vertical:
            print(f"Starting Row {s_v:}")
            for s_h in strides_horizontal:
                region = img[s_v-half_window:s_v+half_window+1,s_h-half_window:s_h+half_window+1,channel]
                r = correlate(img[:,:,channel], region, 
                    normalize=False, 
                    normalize_range=False, 
                    square_response=True,
                    convolve=False,
                    use_ssd=True,
                    is_image=True)
         
                #Set range to [0,256] so r can be cast to np.uint8 for processing
                r = (normalize_array(r) * 255).astype(np.uint8)
                r = cv2.filter2D(r,-1,smooth_kernel)
                r = (normalize_array(r) * 255).astype(np.uint8)
                r = cv2.medianBlur(r,5)
                r = cv2.bilateralFilter(r,15,75,75)

                #Save into correlation_channel_sums
                correlation_channel_sums[:,:,channel] += r

                print(f"Coor: [{s_v:},{s_h:}] Channel {channel:} : {correlation_channel_sums[:,:,0].max():} Channel 1: {correlation_channel_sums[:,:,1].max():} Channel 2: {correlation_channel_sums[:,:,2].max():}")
                if s_h == 0:
                    print(s_h,s_v)
                    


                    #r = cv2.medianBlur(r,5)
                    #r = cv2.bilateralFilter(r,15,75,75)
                    #median = cv2.medianBlur(cv2.medianBlur(cv2.medianBlur(r.astype(np.uint8),5),7),9)
                    #median = cv2.filter2D(median,-1,smooth_kernel)

                    edge2 = cv2.Canny(r,100,200)

                    #Different way to do edges
                    padded = np.pad(r.astype(np.float32),((0,1),(0,1)),mode="edge")
                    dx = np.abs(np.diff(padded[:-1,:]))
                    dy = np.abs(np.diff(padded.T[:-1,:]))
                    d = dx + dy.T
                    d = cv2.filter2D(normalize_array(d) * 255,-1,smooth_kernel)
                    
                    #Sobel edges
                    gx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
                    gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
                    dx2 = signal.convolve2d(np.pad(r,((1,1),(1,1)),mode="edge"),gx,mode="valid")
                    dy2 = signal.convolve2d(np.pad(r,((1,1),(1,1)),mode="edge"),gy,mode="valid")
                    edges = dx2 + dy2


                    f,((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3)

                    ax1.imshow(img[:,:,channel],cmap="gray", vmin=0, vmax=255)
                    ax2.imshow(padded,cmap="gray", vmin=0, vmax=255)
                    ax3.imshow((normalize_array(correlation_channel_sums[:,:,channel]) * 255).astype(np.uint8), cmap="gray", vmin=0, vmax=255)
                    ax4.imshow(d,cmap="gray",vmin=0,vmax=255)
                    ax5.imshow(region,cmap="gray")
                    ax6.imshow((dy.T).astype(np.uint8),cmap="gray")
                    plt.show()
                    #exit()

    def make_img_ready(img):
        return (normalize_array(img) * 255).astype(np.uint8)

    #Set to [0,256] --> median --> smooth
    unranged = np.copy(correlation_channel_sums)
    for channel in range(img.shape[2]):
        correlation_channel_sums[:,:,channel] = cv2.filter2D(cv2.medianBlur(make_img_ready(correlation_channel_sums[:,:,channel]),9),-1,smooth_kernel)

        
    

    id1 = image_derivative(unranged[:,:,0])
    id2 = image_derivative(unranged[:,:,1])
    id3 = image_derivative(unranged[:,:,2])
    imgd = id1 + id2 + id3

    test = (normalize_array(np.sum(unranged,axis=2)/3) * 255).astype(np.uint8)    
    
    #Display
    f,((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8)) = plt.subplots(4,2)

    ax1.imshow(correlation_channel_sums[:,:,0],cmap='gray', vmin=0, vmax=255)
    ax1.set_ylabel("Channel 0")
    ax1.set_xlabel("Correlation")
    ax1.xaxis.set_label_position('top') 
    ax3.imshow(correlation_channel_sums[:,:,1],cmap='gray', vmin=0, vmax=255)
    ax3.set_ylabel("Channel 1")
    ax5.imshow(correlation_channel_sums[:,:,2],cmap='gray', vmin=0, vmax=255)
    ax5.set_ylabel("Channel 2")
    ax7.imshow(np.sum(correlation_channel_sums,axis=2)/3,cmap='gray', vmin=0, vmax=255)
    ax7.set_ylabel("Corr. Sums")

    id1 = image_derivative(correlation_channel_sums[:,:,0])
    id2 = image_derivative(correlation_channel_sums[:,:,1])
    id3 = image_derivative(correlation_channel_sums[:,:,2])

    ax2.imshow(id1,cmap='gray', vmin=0, vmax=255)
    ax2.set_xlabel("Derivative")
    ax2.xaxis.set_label_position('top') 
    ax4.imshow(id2,cmap='gray', vmin=0, vmax=255)
    ax6.imshow(id3,cmap='gray', vmin=0, vmax=255)
    ax8.imshow(imgd,cmap='gray', vmin=0, vmax=255)
    plt.show()


    

    


    


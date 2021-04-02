from SoundLibPeters import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm


import numpy as np 
from scipy.io import wavfile
from scipy import signal

import argparse

if __name__ == "__main__":
    #Initialize arg parse for command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--file",dest="file",action="store",default="test.wav",
                        help="Image in soundfiles folder to load into program")
    parser.add_argument("--duration",dest="duration",action="store",type=float, default=0.2)
    parser.add_argument("--height",dest="height",action="store",type=int, default=32)
    parser.add_argument("--normalize",dest="normalize",action="store_true")
    parser.add_argument("--enhance",dest="enhance",action="store_true")
    parser.add_argument("--maxfreq",dest="maxfreq",action="store",type=int,default=5000)
    parser.add_argument("--blursize",dest="blurksize",action="store",type=int,default=5)
    parser.add_argument("--threshold",dest="threshold",action="store",type=float,default=0.9)
    parser.add_argument("--analysis",dest="analysis",action="store",type=str,default="threshold")
    parser.add_argument("--vowel_length",dest="vowel_length",action="store",type=float,default=0.1)
    parser.add_argument("--extraction",dest="extraction",action="store",type=str,default="vowels")
    parser.add_argument("--template_name", dest="template_name",action="store",type=str,default="ryan")
    parser.add_argument("--stride_size",dest="stride_size",action="store",type=int,default=2)
    parser.add_argument("--output_directory",dest="output_directory",action="store",type=str,default="Outputs/")

    args = parser.parse_args()

    #Load sound file
    soundfiles_dir = "soundfiles/"

    if False:
        #Currently here for a testing phase
        segment_image(cv2.imread("imgs/gradients.png",cv2.IMREAD_COLOR),
            window_size=3,
            stride_size=5,
            display=True)
        exit()

    fs, sig = wavfile.read(soundfiles_dir + args.file)
    ts = timeseries(sig,fs)

    spliced_sound = splice_sound(sig,fs,duration=2.5)

    for sig in spliced_sound:
        #Run extraction method
        if args.extraction == "voicing":
            grad = voicing_detection(sig,fs,
                spectrogram_max_freq=250,
                height=32,
                window_len=5)

        elif args.extraction == "vowels":
            #Create generalized vowel template
            template = create_vowel_template(
                vowel_type="centroid",
                duration=args.duration,
                height=args.height, 
                nblurs=10,
                ksize=args.blurksize,
                sigma=1,
                normalize=args.normalize, 
                enhance=args.enhance,
                maxfreq=args.maxfreq,
                display=False)

            #Use template and signal to extract vowels through 2-step correlation
            vowel_regions = extract_vowels(sig,template,fs,
                lp_cutoff=5000,
                maxfreq=args.maxfreq,
                normalize=args.normalize,
                display=True)

            print("Vowel Regions: {}".format(vowel_regions))

            v_mask = np.zeros_like(sig)
            for i,(s,e) in enumerate(vowel_regions):
                v_mask[s:e] = 1
                sig_vowel = sig[s:e]

                output_directory = args.output_directory
                output_filename = "vowel_output_{}.wav".format(np.random.randint(20))
                wavfile.write(output_directory + output_filename, fs, sig_vowel.astype(np.int16))
                print("Vowel {} written to {}".format(i,output_filename))

            

            f,(ax1,ax2,ax3) = plt.subplots(3,1)
            f,t,Sxx = calculate_spectrogram(sig,fs,window_type="hanning",log10=False)
            ax1.plot(timeseries(sig,fs),sig)
            ax1.set_xlim(0,len(sig)/fs)
            ax2.pcolormesh(t, f, Sxx, cmap=cm.gray_r, shading='gouraud', norm=LogNorm())
            ax2.set(xlabel='Time [sec]', ylabel='Frequency [Hz]')
            ax2.set_ylim(0,5000)
            ax3.plot(v_mask)
            ax3.set_xlim(0,len(v_mask))
            plt.show()

        elif args.extraction == "segmentation":
            boundaries = segment_speech(sig,fs,args.stride_size,
                spectrogram_height=args.height,
                settings="testing",
                display=True)

            #build segmented speech sound file with each segment being 
            total_segment_duration = 1.0
            segment_length = int(fs * total_segment_duration)

            sections = np.split(sig,boundaries)
            for i,s in enumerate(sections):
                sections[i] = np.atleast_2d(np.append(s, np.zeros(segment_length - s.shape[0])))
                
            segmented_speech = np.stack(sections)

            #save output
            output_directory = args.output_directory
            output_filename = "segmented_speech_output_{}.wav".format(np.random.randint(5))
            wavfile.write(output_directory + output_filename, fs, segmented_speech.astype(np.int16))
            print("Segmented Speech written to {}".format(output_filename))

        elif args.extraction == "zcr":
            inten,d_inten = intensity(sig,fs)
            zcr, d_zcr = zero_crossing_rate(sig,fs,unit=0.01)

            f, t, Sxx = calculate_spectrogram(sig,fs,
                window_type="hanning")

            Sxx = np.log10(Sxx)
            
            Sxx = crop_spectrogram(normalize_array(blur2D(Sxx,ksize=5)),f[-1],spectrogram_max_freq=5000)

            f,(ax1,ax2,ax3) = plt.subplots(3,1)
            ax1.plot(timeseries(sig,fs),sig)    
            ax1.set_xlim(0,len(sig)/fs)
            ax2.imshow(Sxx,origin="lower",aspect="auto",interpolation="nearest")
            ax3.plot(inten,color="b", label="Intensity")
            ax3.plot(d_inten,color="r",label="Intensity Derivative")
            ax3.plot(zcr,color="g", label="Zero-Crossing Rate")
            ax3.plot(d_zcr,color="y", label="Zero Crossing Rate Derivative")
            ax3.set_xlim(0,len(inten))
            plt.show()
        elif args.extraction == "matchfilter":
            fs2, template = wavfile.read(soundfiles_dir + args.template_name + ".wav")

            match_filter(sig,fs,template,
                duration=int(fs * 0.1),
                height=args.height)

            

            
            

        else:
            print("Extraction method not set up")
            exit()

    
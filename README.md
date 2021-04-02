# Custom Sound Processing Library

A custom sound library by Ryan Peters used for Sound Segmentation, Voicing Detection, and Vowel Extraction from speech.

## Sound Files
Name of soundfiles in /soundfiles indicates the transcribed speech in it (non-IPA)


## Command Line Arguments
Most of the command line arguments are there for testing besides:
    **--file** which is used for loading the different .wav files
    **--extraction** which determines what analysis you want to perform on the speech

Other arguments' default values are what I have found to be the currently optimal setting/value.


## Extraction/Analysis Methods

### Voicing Detection

Run horizontal sobel edge detection on speech to find vocal fold vibrations. Masks out unvoiced regions

```bash
python LibTesting.py --extraction voicing --file wessen.wav 
```

*wessen.wav is the made-up word "wessen"*


### Vowel Extraction

Use template-matching of a generalized vowel against the spectrogram of the input file to determine the location of vowels within speech
Cleans up boundaries of vowels using a form of autocorrelation/template matching

Needs work to apply a lot of the methods from segmentation to clean up matching and thresholding

```bash
python LibTesting.py --extraction vowels --file nest.wav  --normalize
```


### Sound Segmentation

Segments sounds by its similar/stable regions according to the spectrogram

```bash
python LibTesting.py --extraction segmentation --file not_butter.wav --normalize
```

*not_butter.wav is the sentence "I can't believe it's not butter"*


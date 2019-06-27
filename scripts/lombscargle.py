import numpy as np
import pandas as pd
import gc
from read_train import *
from astropy.stats import LombScargle
from scipy.signal import find_peaks_cwt, find_peaks

info = init_reading()

def cluster_peaks(x, peaks, diffs, threshold):
    i = 0
    best_peaks = []
    diffs = np.append(diffs, 0)
    
    while(i < len(diffs)):
        cluster = []
        flag = 0
        t = threshold
        while(t > 0):
            cluster.append(peaks[i])
            t -= diffs[i]
            i += 1
            if(i >= len(peaks)):
                break;

        mx = x[cluster[0]]
        ind = cluster[0]
        for c in range(1,len(cluster)):
            if (x[cluster[c]] > mx):
                mx = x[cluster[c]]
                ind = cluster[c]
        best_peaks.append(ind)
        
       
    return best_peaks

def find_best(Pxx_den, peaks, no_of_best):
    best = []
    for i in peaks:
        best.append([Pxx_den[i], i])
    best = sorted(best, key=lambda tup: tup[0])
    peaks = []
    for i in range(no_of_best):
        peaks.append(best[-(i+1)][1])
    return peaks


def arr_dist(arr, sep, n=5):
    output = []
    for i,x in enumerate(arr):
        keep=True
        for y in output:
            if abs(y-x)<sep:
                keep=False
                break
        if(keep):
            output.append(i)
            if len(output)==n:
                return(np.asarray(output))
            
def LSP_freq(df, signal_col, time_col, nrows, min_freq, max_freq, freq_sep, threshold):
    print('Lomb-Scargle Periodogram analysis commencing.')
    print('Minimum detection frequency: {}Hz'.format(MIN_FREQ))
    print('Manual maximum frequency cutoff: {}Hz'.format(MAX_FREQ))
    print('Number of segments: ', round(len(df)/ROWS_PER_SEGMENT))
    #initialise empty arrays for frequency outputs to be concatenated to DataFrame 
    freq_1 = np.zeros(len(df))
    freq_2 = np.zeros(len(df))
    freq_3 = np.zeros(len(df))
    freq_4 = np.zeros(len(df))
    freq_5 = np.zeros(len(df))
    freq_0 = np.zeros(len(df))
    amps_0 = np.zeros(len(df))
    segment_num = np.zeros(len(df))
    #loop through input DataFrame in chunks of length=nrows
    init_id = 0
    segment_id =1
    while init_id < len(df):
        if segment_id==1:
            print('Processing segment {:d}...'.format(segment_id))
        if segment_id%25==0:
            print('Processing segment {:d}...'.format(segment_id))
        end_id = min(init_id + nrows, len(df))
        ids = range(init_id, end_id)
        df_chunk = df.iloc[ids]
        #np arrays of amplitude and time columns
        signal = df_chunk[signal_col].values
        ttf = df_chunk[time_col].values
        #clear memory
        del df_chunk
        gc.collect()
        #calulate Lomb-Scargle periodograms for spectral analysis
        freq, power = LombScargle(ttf, signal).autopower(nyquist_factor=2)
        freq_df = pd.DataFrame({'freq': freq.round(),
                               'amp': power})
        
        #obtain frequencies sorted by highest amplitude as np.array

        top_freqs = freq_df.loc[(freq_df.freq > min_freq) & (freq_df.freq < max_freq)].values
        #obtain top 5 values that do not lie within 3kHz of eachother
        dist = 3000
        peaks = []
        while(True):
            peaks, _ = find_peaks(top_freqs[:,0], distance = dist)
            if(len(peaks) >= 5):
                break;
            else:
                dist = dist/ 2
        #peaks  = cluster_peaks(top_freqs[:,1], peaks, np.diff(peaks),3000)

        peaks = find_best(top_freqs[:,0], peaks, 5)
        freqs = np.sort(top_freqs[peaks,1])
    
        #freq_df = freq_df.loc[freq_df['amp'] >= threshold] 
      
        periodic_freq= freq_df.loc[(freq_df.freq > min_freq) & (freq_df.freq < max_freq)].sort_values('amp', ascending=False).freq.values
        periodic_amp = freq_df.loc[(freq_df.freq > min_freq) & (freq_df.freq < max_freq)].sort_values('amp', ascending=False).amp.values

        #update main frequency component arrays
        freq_1[ids] = freqs[4]
        freq_2[ids] = freqs[3]
        freq_3[ids] = freqs[2]
        freq_4[ids] = freqs[1]
        freq_5[ids] = freqs[0]
        try:
            freq_0[ids] = periodic_freq[0]
            amps_0[ids] = periodic_amp[0]
        except:
            freq_0[ids] = 0
        del freq_df, top_freqs, freq, power, peaks
        gc.collect()
        segment_num[ids] = segment_id
        #del top_freqs
        init_id += nrows
        segment_id += 1
    print('...Done. Adding main component frequencies as DataFrame columns...')
    df['Freq_1'] = freq_1
    df['Freq_2'] = freq_2
    df['Freq_3'] = freq_3
    df['Freq_4'] = freq_4
    df['Freq_5'] = freq_5
    df['Freq_periodic'] = freq_0
    df['Amps_periodic'] = amps_0
    df['Segment'] = segment_num
    df['Freq_MinMax'] = df['Freq_1'] - df['Freq_5']
    print('...Done.')
    

for i in range(10,17):
    train = read_object_info(info, i)
    ROWS_PER_SEGMENT = 10000
    TIME_PER_SEGMENT = train.iloc[0, 1] - train.iloc[ROWS_PER_SEGMENT, 1]
    MIN_FREQ = round(1/TIME_PER_SEGMENT)
    MAX_FREQ = 1e8
    FREQ_SEP = 0.2e6
    LSP_freq(train, signal_col = 'acoustic_data', time_col = 'time_to_failure', nrows = ROWS_PER_SEGMENT, 
             min_freq = MIN_FREQ, max_freq = MAX_FREQ, freq_sep = FREQ_SEP, threshold = 0.1)
    train.to_csv("../features_freqs_" +str(i) + ".csv")
    

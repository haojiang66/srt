# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 19:37:04 2023

@author: hybth
"""
'''
#Calculates the frequency, duration and intensity of droughts
#Default threshold is 15th percentile
#Code always returns duration, optional outputs include timing (index of dry days), magnitude and intensity
# and the magnitude/intensity of droughts of lengths specified in 'count'


#Options
# spei_vec:      input data vector
# severity:     threshold for determing drought months (one of 'moderate', 'severe' or 'extreme' (see code below for more details))
# scale:        scale for calculating SPI/SPEI
# subset:       can calculate threshold based on a subset of data
# monthly:      calculates a separate threshold for each month
# add_metrics:  additional metrics to be returned by code (timing: index of drought months)
# count:        drought lengths for which to return frequency, magnitude and intensity
'''

import netCDF4 as nc
import numpy as np


def movingaverage(data, window_size):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
    ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    return ma_vec


def Drought_metrics(spei_vec, lib_path, moving_average_window_size=3):
    severity = 'moderate'
    bins=np.arange(1,12+1)
    
    spei_vec = movingaverage(spei_vec, moving_average_window_size) # 滑动平均，避免单个湿润月份中断干旱
    spei_vec = np.insert(spei_vec,0,[0]*(moving_average_window_size-1)) 
    #在开始插入两个0，避免因为滑动平均导致的数据长度缩短，同时spei=0表示正常情况，不影响干旱判断
    
    metric = drought_metrics_SPEI(spei_vec=spei_vec, 
                                 lib_path=lib_path, severity=severity,
                                 add_metrics=(['timing', 'duration','spei_magnitude','spei_intensity']),
                                 scale=3, count = bins, miss_val=-32768, use_min_threshold=False)
   
    return metric


def drought_metrics_SPEI(spei_vec, lib_path, severity, scale,
                        add_metrics=(['timing', 'magnitude', 'intensity', 'rel_intensity',
                        'count_duration', 'count_magnitude', 'count_intensity', 'count_rel_intensity',
                        'spei_magnitude', 'spei_count_magnitude',
                        'spei_intensity', 'spei_count_intensity']),
                        count=[ 1,  2,  3,  4,  5,  6], miss_val=float('nan'),
                        use_min_threshold=False):


    # source packages
    import sys
    import os
    import numpy as np

    #Set paths and import drought functions
    sys.path.append(os.path.abspath(lib_path))
    from find_consec import find_consec
    from find_end import find_end



    #########################
    ### Find drought days ###
    #########################


    # Set SPI threshold values according to selected severity
    # Criteria from http://www.wamis.org/agm/pubs/SPI/WMO_1090_EN.pdf

    if severity == 'moderate':
        threshold = -1.0
        min_threshold = -1.5
    elif severity == 'severe':
        threshold = -1.5
        min_threshold = -2.0
    elif severity == 'extreme':
        threshold = -2.0
        min_threshold = -100.0 #picked a random no.
    else:
        sys.exit("Check drought severity input, not defined correctly !!")



    #Suppress warning message for below command (which produces a warning when NaNs present in spei_vec)
    #A bit dangerous, should probably fix later
    np.seterr(invalid='ignore')


    ### Find dry months ###
    if use_min_threshold==False:
    	dry_days = np.where(spei_vec <= threshold)[0]
    else:
    	dry_days = np.where((spei_vec <= threshold) & (spei_vec > min_threshold))[0]

    #sort ascending
    dry_days = np.sort(dry_days)

    # breakpoint()

    #####################
    ### Find duration ###
    #####################

    #Filter out consecutive days to find 1st drought day

    if len(dry_days) > 0:

        consec = find_consec(dry_days)

        start = dry_days[np.where(consec == 0)[0]]

        end = find_end(consec=consec, start_days=start, dry_days=dry_days)

        #Calculate duration (difference of end and start days PLUS scale-1)
        duration = (end - start +1)

    else:
        
        start = 0
        
        end = 0
        
        duration = 0


    ### Timing ###

    timing = np.nan

    if 'timing' in add_metrics:

        timing = np.zeros(len(spei_vec))

        if len(dry_days) > 0:
            timing[dry_days]=1



    
    #Count number of drought events of certain length (as defined in "count")

    count_duration = np.nan

    if 'count_duration' in add_metrics:

        count_duration = np.zeros(len(count))

        for ic, c in enumerate(count):
            count_duration[ic] = len(np.where(duration==c)[0])
    
    # breakpoint()



    ##########################
    ### Find SPI magnitude ###
    ##########################

    #Calculates magnitude of drought-month SPI values
    #Don't need to use scale, already reflected in monthly SPI values

    #Initialise here if not wanted outputs, to avoid errors creating output dict
    spei_magnitude = miss_val
    spei_count_magnitude = miss_val

    if 'spei_magnitude' in add_metrics:

        #If found dry days
        if len(dry_days) > 0:

             #initialise
             spei_magnitude = np.zeros(len(start)) + miss_val

             for k in range(len(start)):

                 #More than one consec day
                 if end[k] - start[k] > 0:
                     spei_magnitude[k] = abs(sum(spei_vec[start[k]:(end[k]+1)] ))  #Need to add 1 to end day because of python indexing!!
                 #One consec day only
                 else:
                     spei_magnitude[k] =  abs(spei_vec[start[k]])

        #If no dry days
        else:
            spei_magnitude = 0.



        #Calculate mean magnitude of drought events of certain length (as defined in "count")
        if 'spei_count_magnitude' in add_metrics:

            spei_count_magnitude = np.zeros(len(count)) + miss_val

            #If found dry days
            if len(dry_days) > 0:

                for ic, c in enumerate(count):

                    #find indices for events of duration 'c'
                    ind = np.where(duration==c)[0]

                    if len(ind) > 0:
                        spei_count_magnitude[ic] = np.mean(spei_magnitude[ind])




    ##########################
    ### Find SPI intensity ###
    ##########################

    #Calculates intensity of drought-month SPI values
    #Don't need to use scale, already reflected in monthly SPI values

    #Initialise here if not wanted outputs, to avoid errors creating output dict
    spei_intensity = miss_val
    spei_count_intensity = miss_val

    if 'spei_intensity' in add_metrics:

        #If found dry days
        if len(dry_days) > 0:

             #initialise
             spei_intensity = np.zeros(len(start)) + miss_val

             for k in range(len(start)):

                 #More than one consec day
                 if end[k] - start[k] > 0:
                     spei_intensity[k] = abs(np.mean(spei_vec[start[k]:(end[k]+1)] ))  #Need to add 1 to end day because of python indexing!!
                 #One consec day only
                 else:
                     spei_intensity[k] =  abs(spei_vec[start[k]])

        #If no dry days
        else:
            spei_intensity = 0.



        #Calculate mean magnitude of drought events of certain length (as defined in "count")
        if 'spei_count_intensity' in add_metrics:

            spei_count_intensity = np.zeros(len(count)) + miss_val

            #If found dry days
            if len(dry_days) > 0:

                for ic, c in enumerate(count):

                    #find indices for events of duration 'c'
                    ind = np.where(duration==c)[0]

                    if len(ind) > 0:
                        spei_count_intensity[ic] = np.mean(spei_intensity[ind])


    ### List outputs ###
    outs = {'start': start, 'end': end,'duration': duration,'timing': timing,
            'spei_magnitude': spei_magnitude, 'spei_intensity': spei_intensity,}


    return outs;


if __name__ =='__main__':
    spei_path = r'F:\backup E\SPEI\spei_1960_2019_03_month.nc'
    f = nc.Dataset(spei_path)
    times = f.variables["time"][:]
    spei = np.array(f.variables['SPEI'][:])
    cord = {'s':{'lat':105,'lon':3},'t':{'lat':9,'lon':193}}
    gtype = 's'
    spei_vec = spei[:,cord[gtype]['lat'],cord[gtype]['lon']]
    spei_vec[np.isnan(spei_vec)]=0
    spei_vec = spei_vec[264:]
    lib_path = r'F:\backup E\SRT\Third_talk\functions'
    metric = Drought_metrics(spei_vec, lib_path, moving_average_window_size=3)
    start = metric['start']
    end = metric['end']
    duration = metric['duration']
    timing = metric['timing']
    magnitude = metric['spei_magnitude']
    intensity = metric['spei_intensity']

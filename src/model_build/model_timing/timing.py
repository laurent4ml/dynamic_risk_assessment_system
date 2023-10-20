import os
import timeit

def ingestion_timing():
    '''
    ingestion process timing

    Output 
        timing (float): time spend for ingestion process 
    '''
    starttime = timeit.default_timer()
    os.system('python3 ingestion.py')
    timing=timeit.default_timer() - starttime
    return timing

def training_timing():
    '''
    training process timing

    Output 
        timing (float): time spend for training process 
    '''
    starttime = timeit.default_timer()
    os.system('python3 training.py')
    timing=timeit.default_timer() - starttime
    return timing

def execution_time():
    '''
    calculate timing of training.py and ingestion.py

    Output
        timings (list): timing for training and ingestion tasks
    '''
    timings = {}
    timings['ingestion'] = ingestion_timing()
    timings['training'] = training_timing()
    return timings
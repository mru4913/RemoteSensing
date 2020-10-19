#!/usr/local/bin/python

import os 
import time
import logging

class timer:
    '''
    Timer
    '''
    def __init__(self):
        self.start()
        
    def start(self):
        self.start_time = time.time()
        
    def stop(self, verbose=True):
        self.end_time = time.time()
        if verbose:
            self.time_elapsed = self.end_time - self.start_time 
            print('The process takes',
                  time.strftime('%H:%M:%S', time.gmtime(self.time_elapsed)))
            
class Accumulator:
    '''
    Sum a list of numbers over time
    '''
    def __init__(self, n):
        self.data = [0.0] * n
        
    def add(self, *args):
        self.data = [a+b for a, b in zip(self.data, args)]
        
    def reset(self):
        self.data = [0] * len(self.data)
        
    def __getitem__(self, i):
        return self.data[i]

class datalogging:
    '''
    Data Logging
    '''
    def __init__(self, file_name, *args):
        self.file_name = file_name
        # initialize column names 
        with open(self.file_name, 'a') as f:
            if os.stat(self.file_name).st_size == 0:
                datalogging.write(f, *args)
                
    def record(self, *args):
        with open(self.file_name, 'a') as f:
            datalogging.write(f, *args)
     
    @staticmethod
    def write(f, *args):
        args = (str(i) for i in args)
        string = ','.join(args) + '\n'
        f.write(string)
        f.flush()
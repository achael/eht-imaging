#parloop.py
#Andrew Chael, December 3, 2018
#Wraps up some helper functions for parallel loops

import subprocess
import sys
import os
from multiprocessing import cpu_count
from multiprocessing import Pool
from multiprocessing import Process, Value, Lock

TIMEOUT = 31536000

class Parloop(object):
    """a simple parallel loop with a counter"""

    def __init__(self, func):
        """initialize the loop"""

        self.func = func

    def run_loop(self, arglist, processes=-1):
        """run the loop on the list of arguments with multiple processes"""

        processes = processes
        n = len(arglist)

        if not type(arglist[0]) is list:
            arglist = [[arg] for arg in arglist]

        if processes > 0:
            print("Set up loop with %d Processes" % processes)
        elif processes == 0:  #maximum number of processes -- different argument?
            processes = int(cpu_count())
            print("Set up loop with all available (%d) Processes" % processes)
        else:
            print("Set up loop with no multiprocessing")

        out = -1
        if processes > 0: # run on multiple cores with multiprocessing
            counter = Counter(initval=0, maxval=n)
            pool = Pool(processes=processes, initializer=self._initcount, initargs=(counter,))
            try:
                print('Running the loop')
                self.prog_msg(0, n, 0)
                out = pool.map_async(self,  arglist).get(TIMEOUT)
                pool.close()
            except KeyboardInterrupt:
                print('\ngot ^C while pool mapping, terminating')
                pool.terminate()
                print('pool terminated')
            except Exception as e:
                print('\ngot exception: %r, terminating' % (e,))
                pool.terminate()
                print('pool terminated')
            finally:
                pool.join()

        else: # run on a single core
            out = []
            for i in range(n):
                self.prog_msg(i, n, i-1)
                args = arglist[i]
                out.append(self.func(*args))

        return out

    def _initcount(self,x):
        """initialize the counter"""
        global counter
        counter = x

    def __call__(self,args):
        """call the loop function"""
        try:
            outval = self.func(*args)
            #global counter
            counter.increment()
            self.prog_msg(counter.value(), counter.maxval, counter.value()-1)
            return outval
        except KeyboardInterrupt:
            raise KeyboardInterruptError()


    def prog_msg(self, i, n, i_last=0):
        """print a progress bar"""

        complete_percent_last = int(100*float(i_last)/float(n))
        complete_percent = int(100*float(i)/float(n))
        ndigit = str(len(str(n)))

        bar_width = 30
        progress = int(bar_width * complete_percent/float(100))
        barparams = (i, n, ("-"*progress) + (" " * (bar_width-progress)),complete_percent)

        printstr = "\rProcessed %0"+ndigit+"i/%i : [%s]%i%%"
        sys.stdout.write(printstr % barparams)
        sys.stdout.flush()

class Counter(object):
    """counter object for sharing among multiprocessing jobs"""

    def __init__(self,initval=0,maxval=0):
        self.val = Value('i',initval)
        self.maxval = maxval
        self.lock = Lock()
    def increment(self):
        with self.lock:
            self.val.value += 1
    def value(self):
        with self.lock:
            return self.val.value

class HiddenPrints:
    """suppresses printing from the loop function"""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout




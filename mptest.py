"""from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Process, Queue

import multiprocessing
import numpy as np
import os

print multiprocessing.cpu_count()

def query(i):
    # print multiprocessing.current_process().name
    pass

NUMOBJECTS = 20
pool_size = 16
pool_array = { i:np.array(NUMOBJECTS) for i in range(1,17) }
pool = Pool(pool_size)
# for i in range(NUMOBJECTS):
    # pool.apply_async(query, (i,))
# pool.map(query, np.arange(NUMOBJECTS))

# pool.close()
# pool.join()"""
from multiprocessing import Pool
import multiprocessing

import time

work = [["A", 5], ["B", 2], ["C", 5], ["D", 2], ["A", 5], ["B", 2], ["C", 5], ["D", 2], ["A", 5], ["B", 2], ["C", 5], ["D", 2], ["A", 5], ["B", 2], ["C", 5], ["D", 2]]


def work_log(work_data):
    print multiprocessing.current_process()._identity[0]
    print(" Process %s waiting %s seconds" % (work_data[0], work_data[1]))
    time.sleep(int(work_data[1]))
    print(" Process %s Finished." % work_data[0])
    xx=[multiprocessing.current_process()._identity[0]]
    # pool_array[multiprocessing.current_process()._identity[0]][qres]]=1


def pool_handler():
    p = Pool(6)
    p.map(work_log, work)


if __name__ == '__main__':
    pool_handler()
    print 'DONE'
'''
8
Pool= 0:00:00.203813
mapp= 0:00:00.908389
close= 0:00:00.000025
join= 0:00:00.153066

16
Pool= 0:00:00.418962
mapp= 0:00:00.591470
close= 0:00:00.000017
join= 0:00:00.062467

32
Pool= 0:00:00.830354
mapp= 0:00:00.592138
close= 0:00:00.000012
join= 0:00:00.081253'''
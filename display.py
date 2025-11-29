import matplotlib.pyplot as plt


import numpy as np


def main():

    N_Steps = 512

    env_indx = np.arange(501)[1::20]

    files = ['runtime_doors_async.txt','runtime_doors_sync.txt','runtime_doors_jax_forloop.txt','runtime_doors_jax_full.txt']
    labels = ['async', 'sync', 'jax-step', 'jax-step-loop']

    for i,filename in enumerate(files):
        data = []
        with open(filename,'r') as f:

            lines = f.readlines()

            for line in lines:
                data.append(float(line.split(':')[-1]))
        
        plt.plot(env_indx,data,label=f'{labels[i]} - final: {(data[-1]):.2f} s',linewidth=2)

    plt.xlabel('Number of Environments')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime for Gym vectorization methods and Jax acceleration of stepping and looping (512 step/episode)')
    plt.legend()
    plt.show()

if __name__=='__main__':

    main()
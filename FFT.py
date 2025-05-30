import numpy as np
import cmath
import matplotlib.pyplot as plt
import time

######## MINOR FUNCTIONS
def rounding_number_to_power_2(number_points):
    return 2**int(np.ceil(np.log2(number_points))) 

def evaluating_function_on_space_grid(real_space_function,domain_of_existence,number_of_real_space_points):
    indices=np.arange(number_of_real_space_points)
    return real_space_function(domain_of_existence[0]+indices*(domain_of_existence[1]-domain_of_existence[0])/number_of_real_space_points)

def shifting_and_normalizing_FFT(transformed_function,domain_of_existence,number_of_real_space_points):
    indices=np.arange(number_of_real_space_points)
    #shifting the Fourier transform by e^{-i2*np.pi*k*existence_domain[0]} and multiplying it by dx/(np.sqrt(2*np.pi)) (due to the discretization of the integral into a sum and the definition of FT)
    transformed_function*=np.exp(-2j*np.pi*indices*domain_of_existence[0]/np.abs(domain_of_existence[1]-domain_of_existence[0]))*\
        (np.abs(domain_of_existence[1]-domain_of_existence[0])/(2*np.pi*number_of_real_space_points))
    return transformed_function

def reordering_frequencies(transformed_function):
    number_of_real_space_points=len(transformed_function)
    return np.concatenate((transformed_function[int(number_of_real_space_points/2):],transformed_function[:int(number_of_real_space_points/2)]))
########

######## FFT RECURSIVE IMPLEMENTATION
def FFT_recursive(function):
    N = len(function)
    if N == 1:
        return function
    else:
        function_even=FFT_recursive(function[::2])
        function_odd=FFT_recursive(function[1::2])
        factor = \
          np.exp(-2j*np.pi*np.arange(N)/ N)
        # here using the periodicity of DFT
        transformed_function = np.concatenate(\
            [function_even+factor[:int(N/2)]*function_odd,
            function_even+factor[int(N/2):]*function_odd])
        return transformed_function
########

######## FFT NOT RECURSIVE IMPLEMENTATION
def FFT_not_recursive(function):
    number_of_real_space_points = len(function)
    number_of_bits = int(np.ceil(np.log2(number_of_real_space_points))) 
    # defining a unique matrix to save the values
    skjl=np.zeros((number_of_bits+1,number_of_real_space_points,number_of_real_space_points),dtype=complex)
    # the discrete values of the real space function are renamed considering the mirrored representation of the binary index
    new_indices=[int((bin(i)[2:].zfill(number_of_bits))[::-1],2) for i in range(0,number_of_real_space_points)]
    skjl[0,np.arange(number_of_real_space_points)[:, None],new_indices]=function

    #calculating the different Fourier entries
    for k in range(1,number_of_bits+1):
        skjl[k,np.arange(2**k)[:,None],np.arange(2**(number_of_bits-k))]=skjl[(k-1),np.arange(2**k)[:,None],2*np.arange(2**(number_of_bits-k))]+np.exp(-2j*np.pi*np.arange(2**k)[:,None]/2**k)*skjl[(k-1),np.arange(2**k)[:,None],2*np.arange(2**(number_of_bits-k))+np.ones(2**(number_of_bits-k),dtype=int)]
        skjl[k,np.arange(2**k,number_of_real_space_points)[:,None],np.arange(2**(number_of_bits-k))]=skjl[k,np.arange(number_of_real_space_points-2**k)[:,None],np.arange(2**(number_of_bits-k))]
    
    # the different Fourier entries are ordered with the positive frequencies first and the negative frequencies later
    # frequencies are k_j = j*number_of_real_space_points/(np.abs(domain_entries[1]-domain_entries[0])) where j= 0, 1,..., N/2, -N/2, -N/2+1, -N/2+2, ..., -1 
    return skjl[number_of_bits,:,0]
########

######## STANDARD FOURIER TRANSFORM
def DFT(function):
    number_of_real_space_points = len(function)
    omega=np.exp(-2j*np.pi*np.arange(number_of_real_space_points)[:, None]*np.arange(number_of_real_space_points)/number_of_real_space_points)
    transformed_function=omega@function
    # the different Fourier entries are ordered with the positive frequencies first and the negative frequencies later
    # frequencies are k = j/(np.abs(domain_entries[1]-domain_entries[0])) j= 0, 1,..., N/2, -N/2, -N/2+1, -N/2+2, ..., -1 
    return transformed_function
########

######## PLOTTING FUNCTION
def plot_functions(function1,freq1,name1,function2,freq2,name2,function3,freq3,name3):    
    plt.plot(freq1,np.abs(function1), '.b', alpha=1, label = name1,markersize=12)
    plt.plot(freq2,np.abs(function2), '.r', alpha=1, label = name2,markersize=6)
    plt.plot(freq3,np.abs(function3), '.g', alpha=1, label = name3,markersize=2)
    plt.legend()
    plt.yscale("linear")
    plt.xscale("linear")
    plt.xlabel("k [1/[L]]")
    plt.ylabel("")
    plt.show()

def main(part):
    ############ Implementation FFT (not recursive) and comparison with other routines and to analytical solution
    if part==0:
        # parameters
        number_of_real_space_points=2**7
        number_of_real_space_points=rounding_number_to_power_2(number_of_real_space_points)
        domain_of_existence=[-10,10]
        # function to transform
        def real_space_function(x):
            return np.exp(-x**2/2)
        # function to transform on the real space grid
        function=evaluating_function_on_space_grid(real_space_function,domain_of_existence,number_of_real_space_points)
        
        # naming for plot
        name1="numpy FFT"
        name2="FFT"
        name3="analytical FT"

        # 1) FFT Numpy routine (applying at the end a normalization)
        #numpy_frequencies=np.fft.fftfreq(number_of_real_space_points,np.abs(domain_of_existence[1]-domain_of_existence[0])/number_of_real_space_points)
        start_time_1 = time.time()
        alternative_transformed_function=np.fft.fft(function)
        end_time_1 = time.time()
        alternative_transformed_function*=np.abs(domain_of_existence[1]-domain_of_existence[0])/(2*np.pi*number_of_real_space_points)
        time_1 = end_time_1 - start_time_1
        
        # 3) FT analytical solution
        transformed_domain_of_existence=[-number_of_real_space_points/(2*np.abs(domain_of_existence[1]-domain_of_existence[0])),number_of_real_space_points/(2*np.abs(domain_of_existence[1]-domain_of_existence[0]))]
        def analytical_fourier_transform(k):
            return np.exp(-2*np.pi**2*k**2)/np.sqrt(2*np.pi)
        # function to transform on the real space grid
        analytical_transformed_function=evaluating_function_on_space_grid(analytical_fourier_transform,transformed_domain_of_existence,number_of_real_space_points)
    elif part==1:
        # parameters
        number_of_real_space_points=2**10
        number_of_real_space_points=rounding_number_to_power_2(number_of_real_space_points)
        domain_of_existence=[-np.pi,np.pi]
        # function to transform
        def real_space_function(x):
            return np.where(x<0, -1.0, 1.0)
        # function to transform on the real space grid
        function=evaluating_function_on_space_grid(real_space_function,domain_of_existence,number_of_real_space_points)
        function
        
        # naming for plot
        name1="numpy FFT"
        name2="FFT"
        name3="analytical FT"

        # 1) FFT Numpy routine (applying at the end a normalization)
        #numpy_frequencies=np.fft.fftfreq(number_of_real_space_points,np.abs(domain_of_existence[1]-domain_of_existence[0])/number_of_real_space_points)
        start_time_1 = time.time()
        alternative_transformed_function=np.fft.fft(function)
        end_time_1 = time.time()
        alternative_transformed_function*=np.abs(domain_of_existence[1]-domain_of_existence[0])/(2*np.pi*number_of_real_space_points)
        time_1 = end_time_1 - start_time_1
        
        # 3) FT analytical solution
        transformed_domain_of_existence=[-number_of_real_space_points/(2*np.abs(domain_of_existence[1]-domain_of_existence[0])),number_of_real_space_points/(2*np.abs(domain_of_existence[1]-domain_of_existence[0]))]
        def analytical_fourier_transform(k):
            if np.isscalar(k):
                return (1-np.cos(2*np.pi**2*k))/(2*np.pi**2*k) if k!=0 else 0
            else:
                return np.array([(1-np.cos(2*np.pi**2 * k[i]))/(2*np.pi**2* k[i]) if k[i]!=0 else 0 for i in range(len(k))])
        
        # function to transform on the real space grid
        analytical_transformed_function=evaluating_function_on_space_grid(analytical_fourier_transform,transformed_domain_of_existence,number_of_real_space_points)
    else:
        # parameters
        number_of_real_space_points=2**14
        number_of_real_space_points=rounding_number_to_power_2(number_of_real_space_points)
        domain_of_existence=[-np.pi,np.pi]
        # function to transform
        def real_space_function(x):
            return np.where(((x<=np.pi/2)& (x>-np.pi/2)), -1.0, 1.0)
        # function to transform on the real space grid
        function=evaluating_function_on_space_grid(real_space_function,domain_of_existence,number_of_real_space_points)
        
        # naming for plot
        name1="numpy FFT"
        name2="FFT"
        name3="analytical FT"

        # 1) FFT Numpy routine (applying at the end a normalization)
        #numpy_frequencies=np.fft.fftfreq(number_of_real_space_points,np.abs(domain_of_existence[1]-domain_of_existence[0])/number_of_real_space_points)
        start_time_1 = time.time()
        alternative_transformed_function=np.fft.fft(function)
        end_time_1 = time.time()
        alternative_transformed_function*=np.abs(domain_of_existence[1]-domain_of_existence[0])/(2*np.pi*number_of_real_space_points)
        time_1 = end_time_1 - start_time_1
        
        # 3) FT analytical solution
        transformed_domain_of_existence=[-number_of_real_space_points/(2*np.abs(domain_of_existence[1]-domain_of_existence[0])),number_of_real_space_points/(2*np.abs(domain_of_existence[1]-domain_of_existence[0]))]
        def analytical_fourier_transform(k):
            if np.isscalar(k):
                return -1j*(1-np.cos(2*np.pi**2*k))/(2*np.pi**2*k) if k!=0 else 0
            else:
                return np.array([-1j*(1-np.cos(2*np.pi**2 * k[i]))/(2*np.pi**2* k[i]) if k[i]!=0 else 0 for i in range(len(k))])
        
        # function to transform on the real space grid
        analytical_transformed_function=evaluating_function_on_space_grid(analytical_fourier_transform,transformed_domain_of_existence,number_of_real_space_points)
    

    # 2) FFT routine
    start_time_2 = time.time()
    transformed_function=FFT_not_recursive(function)
    end_time_2 = time.time()
    time_2 = end_time_2 - start_time_2
    transformed_function=shifting_and_normalizing_FFT(transformed_function,domain_of_existence,number_of_real_space_points)
    
    # ordering of the frequencies (for the numpy routine and the FFT-not-recursive routine)
    frequencies=np.concatenate((np.arange(0,number_of_real_space_points/2),np.arange(-number_of_real_space_points/2,0)))
    frequencies=frequencies/(np.abs(domain_of_existence[1]-domain_of_existence[0]))
    ordered_frequencies=np.concatenate((frequencies[int(number_of_real_space_points/2):],frequencies[:int(number_of_real_space_points/2)]))

    transformed_ordered_function=reordering_frequencies(transformed_function)
    alternative_transformed_function=reordering_frequencies(alternative_transformed_function)

    # plot the two routines and the analytical solution
    plot_functions(alternative_transformed_function,ordered_frequencies,name1,transformed_ordered_function,ordered_frequencies,name2,analytical_transformed_function,ordered_frequencies,name3)
   
    print("timing procedure 1:",time_1,"timing procedure 2:",time_2,"time_1 is",((time_2-time_1)/time_2)*100,"% faster than time_2")
        

if __name__ == "__main__":
    main(1)
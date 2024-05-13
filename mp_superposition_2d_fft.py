import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import os
import shutil
import pdb
from scipy.optimize import curve_fit
import scipy
import logging
import multiprocessing as mp
import time


# Create log
#logging.basicConfig(format='%(asctime)s %(message)s',filename='superposition.log', encoding='utf-8', level=logging.INFO)

def read_data(read_ttg_data):
    # Create data arrays for time and amplitude. Convert data for time to ns.
    time_grid = read_ttg_data[:,0]
    ttg_grid = read_ttg_data[:,1]

    #interpolate for t=0
    ttg_grid[0] = 1.
    # Create bsplines for interpolation
    return CubicSpline(time_grid, ttg_grid)

# Function to sum up contribution from all grating wavelengths with given initial condition for a given position x and time t
def ttg_integrate(t, ttg_data, initial_fft, initial_k1, initial_k2):
    # Create a k-grid
    k = np.array(list(ttg_data.keys()))
    k = np.sort(k)

    # Create an array for Fourier spectrum in k space for a given t
    ttg_points = np.array([ttg_data[kk](t) for kk in k]) 
    # Add a point for k=0
    k = np.insert(k, 0, 0)
    ttg_points = np.insert(ttg_points, 0, 1)
    
    # Create an interpolated function for the Fourier spectrum
    ttg_k = CubicSpline(k, ttg_points)

    # Create an array to store convoluted spectrum
    temp_field_fft = np.zeros(initial_fft.shape, dtype=complex)
     

    for i in range(initial_fft.shape[0]):
        for j in range(initial_fft.shape[1]):
            k_squared = np.power(initial_k1[i], 2) + np.power(initial_k2[j], 2)
            k_radial = np.sqrt(k_squared) 
            temp_field_fft[i, j] = initial_fft[i,j]*ttg_k(k_radial)

    # Create an array for the temperature filed
    return scipy.fft.ifft2(temp_field_fft)

def run_superposition(i_cpu, timesteps_per_cpu, t, k1, k2, ttg_data, initial, initial_fft, dump_folder, X, Y, x_lim, y_lim, x, probe, probe_beam):
   
    # Create an array to store temperature field
    temperature=np.zeros(initial.shape)

    for n, tt in enumerate(t):   
        # Create a k-grid
        k = np.array(list(ttg_data.keys()))
        k = np.sort(k)

        # Create an array for Fourier spectrum in k space for a given t
        ttg_points = np.array([ttg_data[kk](tt) for kk in k]) 
        # Add a point for k=0
        k = np.insert(k, 0, 0)
        ttg_points = np.insert(ttg_points, 0, 1)
        
        # Create an interpolated function for the Fourier spectrum
        ttg_k = CubicSpline(k, ttg_points)

        # Create an array to store convoluted spectrum
        temp_field_fft = np.zeros(initial_fft.shape, dtype=complex)
     
        for i in range(initial_fft.shape[0]):
            for j in range(initial_fft.shape[1]):
                k_squared = np.power(k1[i], 2) + np.power(k2[j], 2)
                k_radial = np.sqrt(k_squared) 
                temp_field_fft[i, j] = initial_fft[i,j]*ttg_k(k_radial)
        temperature = scipy.fft.ifft2(temp_field_fft)

        np.savetxt(os.path.join(dump_folder, f'{n+timesteps_per_cpu*i_cpu}.csv'), np.real(temperature))

        if probe_beam:
            probe_signal = np.sum(np.sum(np.multiply(temperature, probe)))
            np.savetxt(os.path.join(dump_folder,  f'{n+timesteps_per_cpu*i_cpu}_probe.csv'), np.array([tt, np.real(probe_signal)]))
        nanoseconds = tt*1e9
        logging.info( '%.1f nanoseconds calculated', nanoseconds)

if __name__ == "__main__":

    # Load ttg data
    #ttg_directory = './graphite/80K/vhe_dynamical_purified/grating'
    #ttg_directory = './graphene/300K/nonthermal_q/k/grating/fft/'
    #ttg_directory = './MoS2/50K/grating/'
    #ttg_directory = './MoS2/300K/grating_gauss_hpc/'
    #ttg_directory = './MoS2/300K/grating_good_backup/'

    #ttg_directory = './MoS2/gke/300K/grating/'
    #ttg_directory = './MoSe2/GKE/300K/bulk/'

    #ttg_directory = './MoS2/ballistic/50K/ZA_only_gratings/'
    #ttg_directory = './MoSe2/ballistic/300K/gratings/'

    #ttg_directory = 

    #ttg_directory = './MoSe2/300K/grating/'

    #ttg_directory = './benchmark/fourier_diffusivity/grating/'
    #ttg_directory = './benchmark/vhe/grating/'
    ttg_directory = './NaF/25K/grating/'
    #ttg_directory = './graphite/150K/grating/'


    # Initialize a  3d array for ttg data
    ttg_data = {} 

    # Iterate over ttg data for all grating lengths
    for filename in os.listdir(ttg_directory):
        # Check if it is a data file
        if not filename.endswith('.txt'):
            continue

        # Read datafile
        file_path = os.path.join(ttg_directory, filename)

        # Get grating length for of this file
        grating_length = float(filename.strip('um.txt'))

        # Append interpolated for this grating to ttg_data as a function of wavenumber k = 1 / L
        ttg_data[1/grating_length] = read_data(np.loadtxt(file_path))


    # Create a new time grid to produce temperature fields for given moments of time
    t_max = 4e-6
    t = np.linspace(0, t_max, 30) # [ns]
    #t = np.array([0.0, 5.0e-9, 10.0e-9, 15.0e-9, 20.0e-9, 25.0e-9, 30.0e-9])
    t_size = len(t)
    #t = np.array([15.0e-9])

    # Set up pool for parallelisation
    n_cpu = 10
    pool=mp.Pool(n_cpu)
    timesteps_per_cpu=int(t_size/n_cpu)
    t_split = np.split(t, n_cpu)

    # Define parameters
    fwhm = 100 # [um]
    fwhm_probe = 3.0 #[um]
    sigma = fwhm/2.355*1e-6
    sigma_probe = fwhm_probe/2.335*1e-6
    ring_radius = 15.0e-6 # [um] Radius of the ring according to Jeong
    #normalize_gaussian = 1/np.sqrt(2*np.pi)/sigma*1e-6
    normalize_gaussian = 1
    
    # Generate x and y values
    x_lim = 1500.0e-6
    y_lim = 1500.0e-6
    dx = 20.e-6
    dy = 20.e-6
    x = np.arange(-x_lim, x_lim, dx)
    y = np.arange(-y_lim, y_lim, dy)

    # Create a meshgrid
    X, Y = np.meshgrid(x, y)

    # Probe signal for thermoreflectance experiment
    probe_beam = False
    probe = np.exp(-0.5 * ((X)**2 + (Y)**2) / (sigma_probe**2))
    probe_signal=np.zeros(len(t))

    # Create the initial signal
    profiles = {}
    profiles.update({'gaussian':normalize_gaussian*np.exp(-0.5 * ((X)**2 + (Y)**2) / (sigma**2))})
    profiles.update({'circle':np.where(np.multiply(abs(X) <= sigma, abs(Y) <= sigma), 1, 0)})
    profiles.update({'ring':np.exp(-0.5 * np.power(np.sqrt(np.power(X, 2) + np.power(Y, 2)) - ring_radius, 2) / (sigma**2))})

    profile = 'gaussian'
    initial = profiles[profile]

    

    # Apply fft 
    initial_fft = scipy.fft.fft2(initial)

    # Create frequency grid
    k1 = scipy.fft.fftfreq(x.size, d=dx)
    k2 = scipy.fft.fftfreq(y.size, d=dy)

    # Create the meshgrid for the reciprocal space
    K1, K2 = np.meshgrid(k1, k2)

    # For each point from x_grid and interp_time_grid sum up all the contributions from all TTG wavelengths with for a given initial condition.
    animation_folder = 'tmp/animation'
    if os.path.exists(animation_folder):
        shutil.rmtree(animation_folder)
    os.makedirs(animation_folder)

    material_str = ttg_directory.split('/')[1]
    temperature_str = ttg_directory.split('/')[2]
    comment_str = ttg_directory.split('/')[3]

    dump_folder = './tmp/dump/'
    if os.path.exists(dump_folder):
        shutil.rmtree(dump_folder)
    os.makedirs(dump_folder)

    dump_folder_profile = './tmp/dump/profile'
    if os.path.exists(dump_folder_profile):
        shutil.rmtree(dump_folder_profile)
    os.makedirs(dump_folder_profile)


    logging.info('Started')
    start = time.time()
    for i in np.arange(0, n_cpu):
        pool.apply_async(run_superposition, args=(i, timesteps_per_cpu, t_split[i], k1, k2, ttg_data, initial, initial_fft, dump_folder, X, Y, x_lim, y_lim, x, probe, probe_beam))

    pool.close()
    pool.join()
    end = time.time()
    calculation_time = end - start
    logging.info('Calculation Finished')

    logging.info('Simulation took %.1f seconds', calculation_time)

    logging.info('Plotting starts')

    
    if probe_beam:
        figure_aspect = 0.3
        n_plots = 3
    else:
        figure_aspect = 0.6
        n_plots = 2

    for n, tt in enumerate(t):
        fig=plt.figure(figsize=plt.figaspect(figure_aspect))
        temperature = np.loadtxt(os.path.join(dump_folder, str(n)+'.csv'))
        if probe_beam:
            probe_signal[n] = np.loadtxt(os.path.join(dump_folder, str(n)+'_probe.csv'))[1]
        if filename.endswith('_profile.csv'):
            continue
        
        ax = fig.add_subplot(1, n_plots, 1, projection='3d')
        ax.plot_surface(X * 1e6, Y *1e6, temperature, cmap='viridis')
        ax.set_title('t = {:.2f} [ns]'.format(tt*1e9))
        ax.set_xlim([-x_lim * 1e6, x_lim * 1e6])
        ax.set_ylim([-y_lim * 1e6, y_lim * 1e6])
        #ax.set_xlim([-x_lim*0.1 * 1e6, x_lim*0.1  * 1e6])
        #ax.set_ylim([-y_lim*0.1 * 1e6, y_lim*0.1  * 1e6])
        ax.set_zlim([0, 1])
        ax.set_xlabel('x [um]')
        ax.set_ylabel('y [um]')
        ax.set_zlabel('T [a.u.]')

        ax = fig.add_subplot(1, n_plots, 2)
        ax.set_title(material_str + ' at ' + temperature_str)
        ax.plot(x*1e6, temperature[ np.argmin(abs(y)), :])
        #ax.set_ylim([0.0, 0.05])
        ax.set_ylim([-0.15, 1])
        #ax.set_xlim([-9*sigma*10e6, 9*sigma*10e6])
        ax.set_xlim([-x_lim * 1e6, x_lim * 1e6])
        ax.set_xlabel('x [um]')
        ax.set_ylabel('T [a.u.]')
        plt.grid(True)
        plt.xticks(minor=True)

        # Dump temperature profile along x axis
        dump_temp_profile = True
        if dump_temp_profile:
            temp_profile_file = '{:.2f}'.format(tt*1e9)+'_profile.csv'
            np.savetxt(os.path.join(dump_folder_profile, temp_profile_file), np.transpose([x*1e6, temperature[ np.argmin(abs(y)), :]]), delimiter=',')

        if probe_beam:
            ax = fig.add_subplot(1, 3, 3)
            ax.set_title('Probe beam signal')
            probe_signal[n] = np.sum(np.multiply(temperature, probe))
            normalizing_const = np.sum(probe)
            ax.plot(t[:(n+1)] * 1e9, probe_signal[:(n+1)]/normalizing_const)
            ax.set_xlabel('t [ns]')
            ax.set_ylabel('Probe signal [a.u.]')
            ax.set_xlim([0, t_max*1e9])
            ax.set_ylim([-0.05, 0.5])

        plt.savefig(os.path.join(animation_folder, f'{n}.png'))
        plt.close()

    

    logging.info('Plotting finished')

    # Dump raw data for the probe plot
    if probe_beam:
        probe_signal_file = 'mp.probe_signal.' + material_str + '.'+temperature_str+'.'+comment_str+'.csv'
        np.savetxt(os.path.join('figures', probe_signal_file), np.transpose([t, probe_signal]), delimiter=',')

    from PIL import Image

    images = [Image.open(os.path.join(animation_folder, f"{n}.png")) for n, tt in enumerate(t)]

    animation_file = 'mp.superposition.'+profile+'.'+material_str+'.'+temperature_str+'.'+comment_str+'.gif'
    images[0].save(os.path.join('figures', animation_file), save_all=True, append_images=images[1:], duration=100, loop=0)

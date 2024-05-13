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


def read_data(read_ttg_data):
    # Create data arrays for time and amplitude. Convert data for time to ns.
    time_grid = read_ttg_data[:,0]
    ttg_grid = read_ttg_data[:,1]

    #interpolate for t=0
    time_grid 

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

    # Plot original and interpolated spectrum
    spectrum_monitor = True
    
    if spectrum_monitor:
        spectrum_path = 'tmp/spectrum_monitor'
        if os.path.exists(spectrum_path):
            shutil.rmtree(spectrum_path)
        os.makedirs(spectrum_path)

        fig, ax = plt.subplots()
        ttg_points_norm = ttg_points
        ttg_points_norm /= np.max(ttg_points_norm)
        initial_fft_norm = np.abs(initial_fft[0, :initial_fft.shape[0]//2]) 
        initial_fft_norm /= np.max(initial_fft_norm)
        interpolated_points_norm = ttg_k(initial_k1[:initial_k1.size//2])
        interpolated_points_norm /= np.max(interpolated_points_norm)

        ax.plot(k*1e-2, ttg_points_norm, marker='v', label='Given Data')
        ax.plot(initial_k1[:initial_k1.size//2]*1e-2, initial_fft_norm, marker='o', label='Initial FFT')
        ax.plot(initial_k1[:initial_k1.size//2]*1e-2, interpolated_points_norm, 'r', label='Interpolated spectrum')
        ax.set_xlabel('Frequency [cm-1]')
        ax.set_ylabel('Amplitude')
        ax.set_xlim(0, initial_k1[:initial_k1.size//2].max()*1e-2)
        #ax.set_xlim(0, 5e5)
        ax.legend()
        plt.show()
        #plt.savefig(os.path.join(spectrum_path, f'{t}.png'))
        plt.close()
     

    for i in range(initial_fft.shape[0]):
        for j in range(initial_fft.shape[1]):
            k_squared = np.power(initial_k1[i], 2) + np.power(initial_k2[j], 2)
            k_radial = np.sqrt(k_squared) 
            temp_field_fft[i, j] = initial_fft[i,j]*ttg_k(k_radial)

    # Create an array for the temperature filed
    return scipy.fft.ifft2(temp_field_fft)

# Create log
logging.basicConfig(format='%(asctime)s %(message)s',filename='superposition.log', encoding='utf-8', level=logging.INFO)

# Load ttg data
ttg_directory = './MoS2/ballistic/10K/gratings/'

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
t_max = 15e-9
t = np.linspace(0, t_max, 50) # [ns]
#t = np.array([15.0e-9])

# Define parameters
fwhm = 10 # [um]
fwhm_probe = 3 #[um]
sigma = fwhm/2.355*1e-6  
sigma_probe = fwhm_probe/2.335*1e-6
ring_radius = 7.5e-6 # [um] Radius of the ring according to Jeong
 
# Generate x and y values
x_lim = 100e-6
y_lim = 100e-6
dx = 0.1e-6
dy = 0.1e-6
x = np.arange(-x_lim, x_lim, dx)
y = np.arange(-y_lim, y_lim, dy)

# Create a meshgrid
X, Y = np.meshgrid(x, y)

# Create the initial signal

profiles = {}
profiles.update({'gaussian':1/np.sqrt(2*np.pi)/sigma*1e6*np.exp(-0.5 * ((X)**2 + (Y)**2) / (sigma**2))})
profiles.update({'circle':np.where(np.multiply(abs(X) <= sigma, abs(Y) <= sigma), 1, 0)})
profiles.update({'ring':np.exp(-0.5 * np.power(np.sqrt(np.power(X, 2) + np.power(Y, 2)) - ring_radius, 2) / (sigma**2))})

probe = np.exp(-0.5 * ((X)**2 + (Y)**2) / (sigma_probe**2))

profile = 'gaussian'
initial = profiles[profile]

probe_signal=np.zeros(len(t))

# Apply fft 
initial_fft = scipy.fft.fft2(initial)

# Create frequency grid
k1 = scipy.fft.fftfreq(x.size, d=dx)
k2 = scipy.fft.fftfreq(y.size, d=dy)

# Create the meshgrid for the reciprocal space
K1, K2 = np.meshgrid(k1, k2)

# Create an array to store temperature field
temperature=np.zeros(initial.shape) 

# For each point from x_grid and interp_time_grid sum up all the contributions from all TTG wavelengths with for a given initial condition.
animation_folder = 'tmp/animation'
if os.path.exists(animation_folder):
    shutil.rmtree(animation_folder)
os.makedirs(animation_folder)

material_str = ttg_directory.split('/')[1]
temperature_str = ttg_directory.split('/')[2]

probe_beam = True

logging.info('Started')

normalizing_const = 1
probe_signal = np.zeros(t.size)

for n, tt in enumerate(t):

    temperature = ttg_integrate(tt, ttg_data, initial_fft, k1, k2) 

    n_plots = 2 

    if probe_beam:
        n_plots = 3
        fig = plt.figure(figsize=plt.figaspect(0.3))

        ax = fig.add_subplot(1, 3, 3)
        ax.set_title('Probe beam signal')
        probe_signal[n] = np.sum(np.multiply(temperature, probe))
        normalizing_const = np.sum(probe)
        ax.plot(t[:(n+1)] * 1e9, probe_signal[:(n+1)]/normalizing_const)
        ax.set_xlabel('t [ns]')
        ax.set_ylabel('Probe signal [a.u.]')
        ax.set_xlim([0, t_max*1e9])
        ax.set_ylim([-0.2, 1]) 
    else:
        fig=plt.figure(figsize=plt.figaspect(0.6))

    ax = fig.add_subplot(1, n_plots, 1, projection='3d')
    ax.plot_surface(X * 1e6, Y *1e6, temperature, cmap='viridis')
    ax.set_title('t = {:.2f} [ns]'.format(tt*1e9))
    ax.set_xlim([-x_lim * 1e6, x_lim * 1e6])
    ax.set_ylim([-y_lim * 1e6, y_lim * 1e6])
    ax.set_zlim([0, 1])
    ax.set_xlabel('x [um]')
    ax.set_ylabel('y [um]')
    ax.set_zlabel('T [a.u.]')

    ax = fig.add_subplot(1, n_plots, 2)
    ax.set_title(material_str + ' at ' + temperature_str)
    ax.plot(x*1e6, temperature[ np.argmin(abs(y)), :])
    ax.set_ylim([-0.15, 1])
    ax.set_xlim([-9*sigma*10e6, 9*sigma*10e6])
    ax.set_xlabel('x [um]')
    ax.set_ylabel('T [a.u.]')


    plt.savefig(os.path.join(animation_folder, f'{n}.png'))
    plt.close()
    
    nanoseconds = tt*1e9
    logging.info( '%.1f nanoseconds calculated', nanoseconds)

logging.info('Calculation Finished')

# Dump raw data for the probe plot
if probe_beam:
    probe_signal_file = 'probe_signal.' + material_str + '.'+temperature_str+'.csv' 
    np.savetxt(os.path.join('figures', probe_signal_file), np.transpose([t, probe_signal]), delimiter=',')

from PIL import Image

images = [Image.open(os.path.join(animation_folder, f"{n}.png")) for n, tt in enumerate(t)]

animation_file = 'superposition.'+profile+'.'+material_str+'.'+temperature_str+'.gif' 
images[0].save(os.path.join('figures', animation_file), save_all=True, append_images=images[1:], duration=100, loop=0)

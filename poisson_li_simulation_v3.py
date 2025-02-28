import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import poisson

# Saving Parameter
SAVE_FIG = False

# Square Wave Parameters
frequency = 1000  # 1kHz
sampling_rate = 1000  # Sampling rate
# True Signal Parameters
r = 1  # Signal Amplitude
phi = np.pi / 6  # Signal Phase
# Lock-in Parameters
freq = 100
w = 2 * np.pi * freq
tf = 1 # Exhibiting time window (sec)
dt = 1e-3  # Time resolution (sec); pulse repetition rate ~ 1000 Hz
num_points_exhibit = int(tf / dt) # Number of points; Exhibiting time window
T = 1  # Integration time (sec)
num_points_int = int(T / dt)  # Number of points; Integration
t = np.linspace(0, tf + T, num_points_exhibit + num_points_int, endpoint=False)  # Sampling time window

# Variables initialization
# Create Image time stack
image = np.zeros((240, 240, num_points_exhibit + num_points_int))
# Randomly select n pixels for the signal
num_pixels = 20
X = np.zeros((num_pixels, num_points_exhibit))
Y = np.zeros((num_pixels, num_points_exhibit))

sig_out = np.zeros((num_pixels, num_points_exhibit))
true_signal = (signal.square(w * t) + 1) / 2
hit_num = 10  # True signal value
noisy_signal = np.zeros((num_pixels, num_points_exhibit + num_points_int))

for i in range(num_pixels):
    # Generate Poisson noise
    hit_noise = poisson.rvs(mu=10, size=int(num_points_exhibit + num_points_int))  # Noise in hit value

    # Combine true signal and noise
    noisy_signal[i] = (hit_noise + hit_num) * true_signal

    # Background noise
    noisy_signal[i] = noisy_signal[i] + np.random.normal(0, 0, num_points_exhibit + num_points_int)  # Adding Gaussian noise
    noisy_signal[i] = noisy_signal[i] + poisson.rvs(mu=0, size=int(num_points_exhibit + num_points_int))  # Adding Poisson noise

    x = np.random.randint(0, 240)
    y = np.random.randint(0, 240)
    # Place the signal in a random pixel
    image[x, y, :] = noisy_signal[i]
    x = np.random.randint(0, 240)
    y = np.random.randint(0, 240)


# Lock-in Amplifier Simulation
mixer1 = np.cos(w * t)
mixer2 = -np.sin(w * t)


signal_mixed_cos = np.zeros((num_pixels, num_points_exhibit + num_points_int))
signal_mixed_sin = np.zeros((num_pixels, num_points_exhibit + num_points_int))
for i in range(num_pixels):
    signal_mixed_cos[i] = noisy_signal[i] * mixer1
    signal_mixed_sin[i] = noisy_signal[i] * mixer2

# Integration
integrated_cos = np.zeros(num_pixels) # Accumulator for in-phase component
integrated_sin = np.zeros(num_pixels) # Accumulator for quadrature component

for i in range(num_pixels):
    for j in range(num_points_int):
        integrated_cos[i] += signal_mixed_cos[i,j] * dt / T
        integrated_sin[i] += signal_mixed_sin[i,j] * dt / T

for i in range(num_pixels):
    for j in range(num_points_exhibit):
        X[i, j] = integrated_cos[i]
        Y[i, j] = integrated_sin[i]
        # Rolling integration window
        integrated_cos[i] = integrated_cos[i] - (signal_mixed_cos[i, j] * dt / T) + (signal_mixed_cos[i, j + num_points_int] * dt / T)
        integrated_sin[i] = integrated_sin[i] - (signal_mixed_sin[i, j] * dt / T) + (signal_mixed_sin[i, j + num_points_int] * dt / T)

# Recover the signal for each pixel
r_out = np.sqrt(X**2 + Y**2)
theta_out = np.arctan2(Y, X)
r_out *= np.pi  # Add correction for frequency splitting

for i in range(num_pixels):
    for j in range(num_points_exhibit):
        sig_out[i, j] = r_out[i, j] * np.cos(w * t[j] + theta_out[i, j])  # Recovered signal



# Plotting for each pixel
figures = []
num_subplots = 9  # 3x3 grid
num_figures = int(np.ceil(num_pixels / num_subplots))

for fig_idx in range(num_figures):
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    axs = axs.flatten()
    figures.append(fig)
    
    for i in range(num_subplots):
        pixel_idx = fig_idx * num_subplots + i
        if pixel_idx >= num_pixels:
            fig.delaxes(axs[i])
        else:
            axs[i].plot(t[:num_points_exhibit], noisy_signal[pixel_idx, :num_points_exhibit], label='Original Signal')
            axs[i].plot(t[:num_points_exhibit], sig_out[pixel_idx, :num_points_exhibit], label='Recovered Signal', linestyle='--')
            axs[i].set_xlabel('Time [s]')
            axs[i].set_ylabel('Amplitude')
            axs[i].set_title(f'Pixel {pixel_idx+1}')
            axs[i].legend()
            axs[i].grid(True)

            # Modify the data limits. Change as needed
            axs[i].set_xlim([0, 0.1])
            axs[i].set_ylim([-1, np.max(noisy_signal[pixel_idx, :num_points_exhibit]) + 1])

    plt.tight_layout(pad=3.0)  # Adjust spacing between subfigures
    plt.show()
    if SAVE_FIG:
        fig.savefig(f'figures/pixel_{fig_idx+1}.png', dpi=300)
        plt.close(fig)

# Plotting for the image
fig, ax = plt.subplots()
im = ax.imshow(image[:, :, 0], cmap='hot', vmin=0, vmax=np.max(image))

def update(frame):
    im.set_array(image[:, :, frame])
    return [im]

ani = animation.FuncAnimation(fig, update, frames=num_points_exhibit + num_points_int, blit=True)
plt.show()

if SAVE_FIG:
    ani.save('figures/image_animation.gif', writer='imagemagick', fps=30)
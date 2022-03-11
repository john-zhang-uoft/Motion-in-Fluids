import pandas as pd
import numpy as np
from uncertainties import unumpy
from uncertainties import ufloat
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
from scipy import stats
from scipy.optimize import curve_fit


data = pd.read_csv('Water.txt', header=0)

velocity = np.array(data['velocity(m/s)'].tolist())
velocity_error = np.array(data['standard_error'].tolist())
diameter = np.array(data['diameter'].tolist())

# add errors
velocity = unumpy.uarray(velocity, velocity_error)
diameter = unumpy.uarray(diameter, 0.1)

C = 1 - 2.104 * (diameter / ufloat(93.8, 0.1)) \
    + 2.089 * ((diameter / ufloat(93.8, 0.1)) ** 2)

corrected_velocity = velocity / C

# diameters are currently in mm
diameter = diameter * 0.001

density = 1.26 * 1000  # in kg/m^3
viscosity = 0.934  # in kg/m*s

reynolds = density * (diameter / 6) * velocity / viscosity
print(reynolds)

radius = diameter / 2

plt.errorbar(x=unumpy.nominal_values(radius), y=unumpy.nominal_values(velocity), yerr=unumpy.std_devs(velocity),
             xerr=unumpy.std_devs(radius), fmt='bo', ecolor='black', linestyle='None')

plt.title('Terminal Velocity vs Bead Radius for Water')
plt.xlabel('Bead Radius (m)')
plt.ylabel('Terminal Velocity (m/s)')

# Fit curve
def calculate_velocity(r, b):
    return b * np.sqrt(r)


popt, pcov = curve_fit(calculate_velocity, unumpy.nominal_values(radius), unumpy.nominal_values(velocity))
print(popt)
print(np.sqrt(np.diag(pcov)))
t = np.arange(0, 0.0035, 0.0001)
plt.plot(t, calculate_velocity(t, popt))

plt.show()
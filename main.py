import pandas as pd
import numpy as np
from uncertainties import unumpy
from uncertainties import ufloat
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
from scipy import stats

# files = [f for f in listdir('C:\\Users\\johnz\\PycharmProjects\\Motion in Fluids\\Data')]


# Import data

# for file in files:
#     data = pd.read_csv(f'Data/{file}', sep='\t', header=1)
#     sns.lineplot(x=data['Time(sec)'], y=data['Position(mm)'], color='blue', alpha=0.4)
#     plt.title(file)
#     plt.show()
#
#     start = float(input("Start time"))
#     stop = float(input("Stop time"))
#
#     terminal_velocity = data.loc[(data['Time(sec)'] >= start) & (data['Time(sec)'] <= stop)]
#
#     terminal_velocity.to_csv(f'Terminal_Velocities/{file}')

    # data['Position(mm)'] = data['Position(mm)'] * 0.001     # convert to meters
    # position = np.array(data['Position(mm)'].tolist())
    # time = np.array(data['Time(sec)'].tolist())


files = [f for f in listdir('C:\\Users\\johnz\\PycharmProjects\\Motion in Fluids\\Terminal_Velocities')]

for file in files:
    data = pd.read_csv(f'Terminal_Velocities/{file}', sep=',', header=0)
    sns.lineplot(x=data['Time(sec)'], y=data['Position(mm)'], color='blue', alpha=0.4)
    plt.title(file)
    plt.show()

    data['Position(mm)'] = data['Position(mm)'] * 0.001     # convert to meters
    position = np.array(data['Position(mm)'].tolist())
    time = np.array(data['Time(sec)'].tolist())

    slope, intercept, r, p, se = stats.linregress(time, position)
    print(str(slope) + ',' + str(r ** 2) + ',' + str(se))


#
# terminal_velocity_df = pd.read_csv('Cleaned_Terminal_Velocities.txt', header=0)
# # terminal_velocity_df['velocity_error(m/s)'] = 0
#
# for ind in terminal_velocity_df.index:
#
#     velocity = ufloat(float(terminal_velocity_df['velocity(m/s)'][ind]), float(terminal_velocity_df['standard_error'][ind]))
#
#     C = 1 - 2.104 * (ufloat(terminal_velocity_df['diameter'][ind], 0.1)/ufloat(93.8, 0.1)) \
#         + 2.089 * ((ufloat(terminal_velocity_df['diameter'][ind], 0.1)/ufloat(93.8, 0.1)) ** 2)
#
#     corrected_velocity = velocity / C
#
#     print(corrected_velocity)


    #
    # terminal_velocity_df.loc['velocity(m/s)', ind] = corrected_velocity.n
    # terminal_velocity_df.loc['velocity_error(m/s)', ind] = corrected_velocity.s

# terminal_velocity_df.to_csv('Corrected_Velocities.csv')



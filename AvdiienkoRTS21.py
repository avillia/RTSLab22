import random
import math
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime


def dft(Y):
    turning_coefficients_R = [[0 for elem in dots] for elem in dots]
    turning_coefficients_I = [[0 for elem in dots] for elem in dots]

    for x in dots:
        for y in dots:
            turning_coefficients_R[x][y] = round(math.cos((2 * math.pi / amount_of_dots) * ((x*y) % amount_of_dots)))
            turning_coefficients_I[x][y] = round(- math.sin((2 * math.pi / amount_of_dots) * ((x*y) % amount_of_dots)))

    turning_coefficients_R = np.array(turning_coefficients_R)
    turning_coefficients_I = np.array(turning_coefficients_I)

    Y = np.transpose(np.array(Y))

    dft_magnitude_R = turning_coefficients_R.dot(Y)
    dft_magnitude_I = turning_coefficients_I.dot(Y)

    results_for_plotting = [math.sqrt(dft_magnitude_I[index] ** 2 + dft_magnitude_R[index] ** 2) for index in dots]
    return results_for_plotting


if __name__ == '__main__':
    # Signal generation

    amount_of_harmonics = 14
    harmonics = range(amount_of_harmonics)

    amount_of_dots = 100
    dots = range(amount_of_dots)

    Ω_max = 1100
    Ω_interval = Ω_max / amount_of_harmonics

    amplitudes = [random.randint(1, 10) for i in harmonics]

    phases = [random.uniform(0, math.pi) for i in harmonics]

    mainX = [sum([amplitudes[j] * math.sin(Ω_interval * (j + 1) * i + phases[j]) for j in harmonics]) for i in dots]

    # Execution and time test for transformation

    dft_start = datetime.now()
    dft_results = dft(Y=mainX)
    dft_finish = datetime.now()
    print(f"FFT time of execution (non-parallel):\n{dft_finish - dft_start} ms.")

    # Results plotting

    fig, (dft_plot) = plt.subplots(1)
    dft_plot.plot(dft_results)
    fig.show()

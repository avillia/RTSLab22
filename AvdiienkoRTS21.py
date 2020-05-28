import random
import math
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime


def dft(Y):
    turning_coefficients_R = [[0 for _ in dots] for _ in dots]
    turning_coefficients_I = [[0 for _ in dots] for _ in dots]

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


def table_dft(Y):
    turning_coefficients_R = [[0 for _ in dots] for _ in dots]
    turning_coefficients_I = [[0 for _ in dots] for _ in dots]
    Fp2_real = [0] * 1024
    Fp2_image = [0] * 1024

    for x in dots:
        for y in dots:
            turning_coefficients_R[x][y] = math.cos(2 * math.pi / Ω_max * x * y)
            turning_coefficients_I[x][y] = math.sin(2 * math.pi / Ω_max * x * y)

    for i in range(len(turning_coefficients_R)):
        for j in range(len(turning_coefficients_R[i])):
            Fp2_real[i] += Y[j] * turning_coefficients_R[i][j]
            Fp2_image[i] += Y[j] * turning_coefficients_I[i][j]

    results_for_plotting = [(r ** 2 + i ** 2) ** 0.5 for (r, i) in zip(Fp2_real, Fp2_image)]
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
    print(f"DFT time of execution:\n{dft_finish - dft_start} ms.")

    table_dft_start = datetime.now()
    table_dft_results = table_dft(Y=mainX)
    table_dft_finish = datetime.now()
    print(f"Table DFT time of execution:\n{table_dft_finish - table_dft_start} ms.")

    # Results plotting

    fig, (dft_plot, table_dft_plot) = plt.subplots(2, 2)
    dft_plot.plot(dft_results)
    table_dft_plot.plot(table_dft_results)
    fig.show()

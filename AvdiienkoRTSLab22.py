import random
import math
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures

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


def fft(Y):
    fft_dot_number = amount_of_dots // 2

    x_real_first = [0]*fft_dot_number
    x_real_second = [0]*fft_dot_number
    x_imag_first = [0]*fft_dot_number
    x_imag_second = [0]*fft_dot_number

    x_real = [0]*amount_of_dots
    x_imag = [0]*amount_of_dots
    results_for_plotting = [0]*amount_of_dots

    for x in range(fft_dot_number):
        for y in range(fft_dot_number):
            wpm_cos = math.cos(4 * math.pi * x * y / fft_dot_number)
            wpm_sin = math.sin(4 * math.pi * x * y / fft_dot_number)
            x_real_first[x] += Y[2 * y + 1] * wpm_cos
            x_imag_first[x] += Y[2 * y + 1] * wpm_sin
            x_real_second[x] += Y[2 * y] * wpm_cos
            x_imag_second[x] += Y[2 * y] * wpm_sin

        wpn_cos = math.cos(2 * math.pi * x / fft_dot_number)
        wpn_sin = math.sin(2 * math.pi * x / fft_dot_number)

        x_real[x] = x_real_second[x] + x_real_first[x] * wpn_cos - x_imag_first[x] * wpn_sin
        x_imag[x] = x_imag_second[x] + x_imag_first[x] * wpn_cos + x_real_first[x] * wpn_sin

        x_real[x + fft_dot_number] = x_real_second[x] - (x_real_first[x] * wpn_cos - x_imag_first[x] * wpn_sin)
        x_imag[x + fft_dot_number] = x_imag_second[x] - (x_imag_first[x] * wpn_cos + x_real_first[x] * wpn_sin)

        results_for_plotting[x] = math.sqrt((x_real[x]) ** 2 + (x_imag[x]) ** 2)
        results_for_plotting[x + fft_dot_number] = math.sqrt((x_real[x + fft_dot_number]) ** 2 + (x_imag[x + fft_dot_number]) ** 2)

    return results_for_plotting


def parallel_fft(Y):
    fft_dot_number = amount_of_dots // 2

    x_real = [0] * amount_of_dots
    x_imag = [0] * amount_of_dots
    results_for_plotting = [0] * amount_of_dots

    with concurrent.futures.ThreadPoolExecutor() as executor:
        fft_first_half  = executor.submit(fft_first_thread,  Y, fft_dot_number)
        fft_second_half = executor.submit(fft_second_thread, Y, fft_dot_number)
        x_real_first, x_imag_first = fft_first_half.result()
        x_real_second, x_imag_second = fft_second_half.result()

    for x in range(fft_dot_number):
        wpn_cos = math.cos(2 * math.pi * x / fft_dot_number)
        wpn_sin = math.sin(2 * math.pi * x / fft_dot_number)

        x_real[x] = x_real_second[x] + x_real_first[x] * wpn_cos - x_imag_first[x] * wpn_sin
        x_imag[x] = x_imag_second[x] + x_imag_first[x] * wpn_cos + x_real_first[x] * wpn_sin

        x_real[x + fft_dot_number] = x_real_second[x] - (x_real_first[x] * wpn_cos - x_imag_first[x] * wpn_sin)
        x_imag[x + fft_dot_number] = x_imag_second[x] - (x_imag_first[x] * wpn_cos + x_real_first[x] * wpn_sin)

        results_for_plotting[x] = math.sqrt((x_real[x]) ** 2 + (x_imag[x]) ** 2)
        results_for_plotting[x + fft_dot_number] = math.sqrt(
            (x_real[x + fft_dot_number]) ** 2 + (x_imag[x + fft_dot_number]) ** 2)

    return results_for_plotting


def fft_first_thread(Y, fft_dot_number):
    x_real_first = [0] * fft_dot_number
    x_imag_first = [0] * fft_dot_number

    for x in range(fft_dot_number):
        for y in range(fft_dot_number):
            wpm_cos = math.cos(4 * math.pi * x * y / fft_dot_number)
            wpm_sin = math.sin(4 * math.pi * x * y / fft_dot_number)
            x_real_first[x] += Y[2 * y + 1] * wpm_cos
            x_imag_first[x] += Y[2 * y + 1] * wpm_sin

    return x_real_first, x_imag_first


def fft_second_thread(Y, fft_dot_number):
    x_real_second = [0] * fft_dot_number
    x_imag_second = [0] * fft_dot_number

    for x in range(fft_dot_number):
        for y in range(fft_dot_number):
            wpm_cos = math.cos(4 * math.pi * x * y / fft_dot_number)
            wpm_sin = math.sin(4 * math.pi * x * y / fft_dot_number)
            x_real_second[x] += Y[2 * y] * wpm_cos
            x_imag_second[x] += Y[2 * y] * wpm_sin

    return x_real_second, x_imag_second


if __name__ == '__main__':
    # Signal generation

    amount_of_harmonics = 14
    harmonics = range(amount_of_harmonics)

    amount_of_dots = 1000
    dots = range(amount_of_dots)

    Ω_max = 1100
    Ω_interval = Ω_max / amount_of_harmonics

    amplitudes = [random.randint(1, 10) for i in harmonics]

    phases = [random.uniform(0, math.pi) for i in harmonics]

    mainX = []
    for i in dots:
        currentX = 0
        for j in harmonics:
            currentX += amplitudes[j] * math.sin(Ω_interval * (j + 1) * i + phases[j])
        mainX.append(currentX)

    # Execution and time test for each of transformations

    fft_start = datetime.now()
    fft_results = fft(Y=mainX)
    fft_finish = datetime.now()
    print(f"FFT time of execution (non-parallel):\n{fft_finish - fft_start} ms.")

    parallel_ftt_start = datetime.now()
    parallel_ftt_results = parallel_fft(Y=mainX)
    parallel_ftt_finish = datetime.now()
    print(f"Parallel FFT time of execution:\n{parallel_ftt_finish - parallel_ftt_start} ms.")

    # Results plotting

    fig, (fft_plot, parallel_ftt_plot) = plt.subplots(2)
    fft_plot.plot(fft_results)
    parallel_ftt_plot.plot(parallel_ftt_results)
    fig.show()

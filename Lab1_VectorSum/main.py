import numpy as np
import time
from numba import cuda
from prettytable import PrettyTable

def sum_vector_cpu(vector):
    total = 0
    for value in vector:
        total += value
    return total

# Ядро для суммирования на GPU
@cuda.jit
def sum_vector_gpu_kernel(vector, part_sums):
    ind = cuda.grid(1)  # Получаем индекс потока
    if ind < vector.size:
        part_sums[ind] = vector[ind]  # Копируем значения 

# Ядро для редукции
@cuda.jit
def reduce_kernel(part_sums, result):
    ind = cuda.grid(1)
    step = cuda.gridDim.x * cuda.blockDim.x  # Шаг для редукции

    # Суммируем элементы с шагом 
    for i in range(ind, part_sums.size, step):
        result[0] += part_sums[i]


def sum_vector_gpu(vector):
    n = vector.size
    threads_per_block = 256
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

    # Выделяем память для частичных сумм
    part_sums = np.zeros(n, dtype=np.float64)
    part_sums_device = cuda.to_device(part_sums)  # Перемещение на GPU

    sum_vector_gpu_kernel[blocks_per_grid, threads_per_block](vector, part_sums_device)

    # Выделяем память для хранения результата
    result = np.zeros(1, dtype=np.float64)
    result_device = cuda.to_device(result)

    reduce_kernel[blocks_per_grid, threads_per_block](part_sums_device, result_device)

    # Копируем результат обратно на CPU
    result_device.copy_to_host(result)

    return result[0]  

sizes = [1000, 10000, 100000, 500000, 1000000]
table = PrettyTable()
table.field_names = ["Vector Size", "CPU Time (s)", "GPU Time (s)", "Acceleration"]

def main():
    for size in sizes:
        vector = np.random.rand(size)  # Генерация вектора 

        # Сложение на CPU
        start_time_cpu = time.time()
        cpu_sum = sum_vector_cpu(vector)
        end_time_cpu = time.time()
        cpu_time = end_time_cpu - start_time_cpu

        # Сложение на GPU
        start_time_gpu = time.time()
        gpu_sum = sum_vector_gpu(vector)
        end_time_gpu = time.time()
        gpu_time = end_time_gpu - start_time_gpu

        if gpu_time > 0: # Для того, чтобы избежать ошибки в случае, когда gpu_time = 0
            Acceleration = cpu_time / gpu_time
        else:
            Acceleration = 0
            

        # Добавляем результаты в таблицу
        table.add_row([size, f"{cpu_time:.6f}", f"{gpu_time:.6f}", f"{Acceleration:.6f}"])

    with open("vector_sum_results.txt", "w") as f: 
        f.write(table.get_string()) # Запись таблички 
    
    print(table)
main()

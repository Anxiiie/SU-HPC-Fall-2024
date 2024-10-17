import numpy as np
from numba import jit, cuda
import time 
from prettytable import PrettyTable

# @jit(nopython=True) # используя декоратор jit можно значительно ускорить
# работу компилятора используя для оптимизации компилятор numba 
# Результаты были получены без него
def matrix_multiplication_cpu(A, B):
    size = A.shape[0] # "Узнаем" размер матриц для данного метода
    C = np.zeros((size,size)) # определяем размерность результирующей матрицы
    
    # Перемножение матриц
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
    return C

@cuda.jit
def matrix_multiplication_gpu(A, B, C):
    row, col = cuda.grid(2) # Функция grid нужна для того, чтобы задать размерности сетки
    # она возвращается переменные row - индекс текущей строки и col - индекс текущего столбца
    if row < C.shape[0] and col < C.shape[1]: # Условие нахождение индексов в пределах матрицы С
        temp = 0.0
        for k in range(A.shape[1]):
            temp += A[row, k] * B[k, col] # Переменная temp накапливает сумму 
            #произведений для конкретной ячейки матрицы C
        C[row, col] = temp
        
def check_correctness(C_cpu, C_gpu):
    return np.allclose(C_cpu, C_gpu) # Функция allclose возвращает True, если массивы поэлементно равны

def main():
    size = range(100, 2001, 100) # Диапазон размеров матриц с шагом в 100
    
    table = PrettyTable() # Создание красивой таблички для вывода данных
    table.field_names = ["Matrix Size", "CPU Time (s)", "GPU Time (s)", "Correctness", "Acceleration"] 
    
    for el in size:
        # Генерация случайных матриц
        A = np.random.rand(el, el).astype(np.float32) #astype для изменения типа данных массива
        B = np.random.rand(el, el).astype(np.float32) # на тип float32 - 32бит. числа с плавающей точкой
        
        # Перемножение на CPU
        start_cpu = time.time() # Засекаем время
        C_cpu = matrix_multiplication_cpu(A, B)
        end_cpu = time.time() # Останавливаем "хронометр"
        cpu_time = end_cpu - start_cpu # Высчитываем время 
        # print(f"CPU time for {el}x{el}: {cpu_time:.5f} sec") # нужно было для отладки 

        # Перемножение на GPU
        C_gpu = np.zeros((el, el), dtype=np.float32)
        A_device = cuda.to_device(A) # Загрузка массива в память GPU 
        B_device = cuda.to_device(B)
        C_device = cuda.to_device(C_gpu)

        threads_per_block = (32, 32) # Задаем кол-во потоков на блок
        blocks_per_grid_x = int(np.ceil(el / threads_per_block[0])) # Вычисляем кол-во блоков по оси Х
        blocks_per_grid_y = int(np.ceil(el / threads_per_block[1])) # Вычисляем кол-во блоков по оси Y # np.ceil для учёта неполных блоков
        matrix_multiplication_gpu[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](A_device, B_device, C_device) # [(кол-во потоков в сетке), кол-во потоков в блоке]  # (массивы из памяти GPU) 
        C_gpu = C_device.copy_to_host() # Возвращает значение из памяти GPU 

        end_gpu = time.time()
        gpu_time = end_gpu - end_cpu
        # print(f"GPU time for {el}x{el}: {gpu_time:.5f} sec") # для отладки

        # Проверка корректности
        if check_correctness(C_cpu, C_gpu):
            correctness = "Yes"
        else:
            correctness = "No"
            
        if gpu_time > 0: # Для того, чтобы избежать ошибки в случае, когда gpu_time = 0
            Acceleration = cpu_time / gpu_time
        else:
            Acceleration = float('None')
        
        # Добавление строк в красивую табличку    
        table.add_row([f"{el}x{el}", f"{cpu_time:.5f}", f"{gpu_time:.5f}", correctness, f"{Acceleration:.3f}"]) 
        
    with open("matrix_multiplication_results.txt", "w") as f: 
        f.write(table.get_string()) # Запись таблички на всякий случай
        
    print(table) 
main()

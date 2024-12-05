import numpy as np
from numba import cuda
import time

def initialize_population(size, polynomial_degree):
    return np.random.rand(size, polynomial_degree + 1)

@cuda.jit
def fitness(individuals, points, fitness_values):
    idx = cuda.grid(1)
    if idx < individuals.shape[0]:
        error_sum = 0.0
        for i in range(points.shape[0]):
            f_approx = 0.0
            for j in range(individuals.shape[1]):
                f_approx += individuals[idx, j] * (points[i] ** j)
            target_value = np.tanh(points[i])  # f(x)
            error_sum += (f_approx - target_value) ** 2
        fitness_values[idx] = error_sum

def crossover(individuals):
    new_population = []
    while len(new_population) < len(individuals):
        parent1 = individuals[np.random.randint(len(individuals))]
        parent2 = individuals[np.random.randint(len(individuals))]
        cross_point = np.random.randint(1, individuals.shape[1])
        child1 = np.concatenate((parent1[:cross_point], parent2[cross_point:]))
        child2 = np.concatenate((parent2[:cross_point], parent1[cross_point:]))
        new_population.append(child1)
        new_population.append(child2)
    return np.array(new_population)

def mutation(individuals):
    Em = 0  # Среднее для мутации
    Dm = 0.1  # Дисперсия для мутации
    for i in range(individuals.shape[0]):
        for j in range(individuals.shape[1]):
            if np.random.rand() < 0.01:  # Вероятность мутации
                individuals[i, j] += np.random.normal(Em, np.sqrt(Dm))  # Нормальное распределение
    return individuals

def selection(individuals, fitnesses):
    sorted_indices = np.argsort(fitnesses)
    return individuals[sorted_indices]

def genetic_algorithm(points):
    POLYNOMIAL_DEGREE = 4
    population_size = 1500  # Размер популяции
    max_iter = 1000  # Максимальное количество поколений
    max_const_iter = 50  # Максимальное количество поколений с постоянным фитнесом
    
    population = initialize_population(population_size, POLYNOMIAL_DEGREE)
    fitness_values = np.zeros(population_size)

    start_time = time.time()  # Начало отсчета времени
    best_fitness = float('inf')
    last_generation = 0
    const_iter_count = 0

    for generation in range(max_iter):
        # Вычисление фитнеса на GPU
        d_individuals = cuda.to_device(population)
        d_fitness_values = cuda.device_array(population_size)
        d_points = cuda.to_device(points)

        threads_per_block = 256
        blocks_per_grid = (population_size + (threads_per_block - 1)) // threads_per_block
        fitness[blocks_per_grid, threads_per_block](d_individuals, d_points, d_fitness_values)

        fitness_values = d_fitness_values.copy_to_host()

        # Селекция
        population = selection(population, fitness_values)

        # Генерация нового поколения
        new_population = crossover(population)
        population = mutation(new_population)

        # Обновление лучшего фитнеса
        current_best_fitness = np.min(fitness_values)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            const_iter_count = 0  # Сброс счетчика при улучшении
        else:
            const_iter_count += 1

        last_generation = generation

        # Проверка на максимальное количество итераций с постоянным фитнесом
        if const_iter_count >= max_const_iter:
            break

    end_time = time.time()  # Конец отсчета времени

    # Возвращаем результаты
    best_index = np.argmin(fitness_values)
    return population[best_index], best_fitness, last_generation, end_time - start_time

points = np.linspace(-10, 10, 500)  # Набор точек на поверхности

# Запуск генетического алгоритма
best_coefficients, best_fitness, last_generation, processing_time = genetic_algorithm(points)

# Вывод результатов
print("Лучшие коэффициенты полинома:", best_coefficients)
print("Лучшее значение фитнеса:", best_fitness)
print("Номер последнего поколения:", last_generation)
print("Время обработки на GPU:", processing_time, "секунд")

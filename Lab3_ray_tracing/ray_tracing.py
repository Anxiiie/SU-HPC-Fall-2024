import numpy as np
from numba import cuda


def normalize(v):
    # Нормализация вектора
    return v/(np.linalg.norm(v, axis=1, keepdims=True)+1e-10)


@cuda.jit
def ray_sphere_intersect(rays_o, rays_d, min_t, 
                         centers, radii, albedos, materials, roughnesses,
                         obj_idxs, hit_record,
                         rand_vec3):
    # Определение количества лучей и сфер
    N, M = len(rays_o), len(centers)
    start_i = cuda.grid(1)  # Индекс текущего потока
    grid_i = cuda.gridDim.x * cuda.blockDim.x  # Общее количество потоков в сетке
    
    for i in range(start_i, N, grid_i):  # Итерация по всем лучам
        o, d = rays_o[i], rays_d[i]  # Начало и направление луча
        for j in range(M):  # Итерация по всем сферам
            center = centers[j]  # Центр текущей сферы
            radius = radii[j]  # Радиус текущей сферы
            albedo = albedos[j]  # Цвет текущей сферы
            material = materials[j]  # Материал текущей сферы
            roughness = roughnesses[j]  # Шероховатость текущей сферы
            obj_idx = obj_idxs[j]  # Индекс объекта текущей сферы
            
            # Вектор от начала луча до центра сферы
            co0 = o[0] - center[0]
            co1 = o[1] - center[1]
            co2 = o[2] - center[2]
            
            # Коэффициенты квадратного уравнения
            a = d[0]**2 + d[1]**2 + d[2]**2
            half_b = co0 * d[0] + co1 * d[1] + co2 * d[2]
            c = co0**2 + co1**2 + co2**2 - radius**2
            
            disc = half_b**2 - a * c
            
            if disc > 0:  # Проверка наличия пересечения
                t = (-half_b - disc**0.5) / a  # Вычисление параметра t
                if t > min_t and t < hit_record[i, 0]:  # Проверка на минимальное значение t
                    hit_record[i, 0] = t  # Сохранение t в hit_record
                    hit_record[i, 1] = obj_idx  # Сохранение индекса объекта
                    # Вычисление координат точки пересечения
                    hit_record[i, 2] = o[0] + t * d[0]
                    hit_record[i, 3] = o[1] + t * d[1]
                    hit_record[i, 4] = o[2] + t * d[2]
                    hit_record[i, 12] = albedo[0]  # Сохранение цвета
                    hit_record[i, 13] = albedo[1]
                    hit_record[i, 14] = albedo[2]
                    
                    r_inv = radius**-1  # Обратный радиус
                    n0 = (co0 + t * d[0]) * r_inv  # Нормаль в точке пересечения
                    n1 = (co1 + t * d[1]) * r_inv
                    n2 = (co2 + t * d[2]) * r_inv
                    
                    ddotn = d[0] * n0 + d[1] * n1 + d[2] * n2  # Скаляры
                    if ddotn < 0:  # Если нормаль направлена в сторону луча
                        hit_record[i, 5] = n0
                        hit_record[i, 6] = n1
                        hit_record[i, 7] = n2
                        hit_record[i, 8] = 1  # Передняя грань
                    else:  # Если нормаль направлена от луча
                        hit_record[i, 5] = -n0
                        hit_record[i, 6] = -n1
                        hit_record[i, 7] = -n2
                        hit_record[i, 8] = 0  # Задняя грань

                    # Обработка материалов
                    if material == 0:  # Матовый
                        hit_record[i, 9] = hit_record[i, 5] + rand_vec3[i, 0]
                        hit_record[i, 10] = hit_record[i, 6] + rand_vec3[i, 1]
                        hit_record[i, 11] = hit_record[i, 7] + rand_vec3[i, 2]
                        hit_record[i, 15] = 1
                    elif material == 1:  # Металлический
                        hit_record[i, 9] = d[0] - 2 * ddotn * n0 + roughness * rand_vec3[i, 0]
                        hit_record[i, 10] = d[1] - 2 * ddotn * n1 + roughness * rand_vec3[i, 1]
                        hit_record[i, 11] = d[2] - 2 * ddotn * n2 + roughness * rand_vec3[i, 2]
                        hit_record[i, 15] = \
                            hit_record[i, 9] * n0 + \
                            hit_record[i, 10] * n1 + \
                            hit_record[i, 11] * n2 > 0  # Проверка отражения

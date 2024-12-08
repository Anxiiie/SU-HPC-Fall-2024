from ray_tracing import *
from objects import *
from random_utils import *
from camera import Camera
import matplotlib.pyplot as plt
import time

# define constants
SAMPLES_PER_RAY = 100  # Количество выборок на один луч
MAX_RAY_BOUNCE = 50  # Максимальное количество отражений луча
MAX_RAY_LENGTH = 1e20  # Максимальная длина луча


def create_grid_scene(N, space=2):
    # Создание сцены с поверхностью и сферами
    scene = [Sphere(np.float32([0, -1000, 0]), 1000, 
                    albedo=np.float32([0.76, 0.64, 0.52]),
                    material=Material.MATTE)] # Создание поверхности

    # Создание сетки и размещение на ней сфер
    for a in range(-N, N):  # Ось Х
        for b in range(-N, N):  # Ось Z 
            # Определение центра сферы
            center = np.float32([a * space, 0.6, b * space*2])

            # Создаем сферы вокруг центра сцены
            if np.linalg.norm(center - np.float32([0, 0.2, 0])) > 0.9:
                # Создаём металлические сферы
                albedo = np.float32(np.random.random(3) * 0.5 + 0.5) # Случайный цвет
                roughness = np.random.random() * 0.5 # Случайная шероховатость
                scene += [Sphere(center, 0.6, 
                                 albedo=albedo, 
                                 material=Material.METAL,
                                 roughness=roughness)] # Добавление сферы в сцену

    return scene



if __name__ == '__main__':
    # Определение параметров камеры
    img_wh, fov = (1280, 720), np.pi/6 # Ширина и высота изображения, угол обзора
    lookfrom = np.float32([4, 4, 13]) # Положение камеры
    lookat = np.float32([0, 0, -1]) # Точка, на которую смотрит камера
    vup = np.float32([0, 1, 0]) # Вектор вверх для камеры
    focus_dist = 13 # Расстояние фокусировки
    aperture = 0.1 # Диафрагма камеры
    camera = Camera(lookfrom, lookat, vup, img_wh, fov, focus_dist, aperture) # Создание камеры

    # Параметры CUDA 
    threads_per_block = 512 # Количество потоков на блок
    blocks = 4096 # Общее количество блоков
    chunk = int(2**20) # Размер чанка для обработки


    tscene = time.time() # Начало отсчёта времени создания сцены
    
    # Конструкция сцены
    scene = create_grid_scene(11)

    # Создание буферов для сцены
    centers = [] # Список центров объектов
    radii = [] # Список радиусов объектов
    albedos = [] # Список цветов объектов
    materials = [] # Список материалов объектов
    roughnesses = [] # Список шероховатостей объектов
    obj_idxs = [] # Список индексов объектов

    # Добавление элементов
    for obj_idx, obj in enumerate(scene):
        centers += [obj.center] 
        radii += [obj.radius]
        albedos += [obj.albedo]
        materials += [obj.material.value]
        roughnesses += [obj.roughness]
        obj_idxs += [obj_idx]

    # Преобразование списков в массивы NumPy
    centers = np.ascontiguousarray(centers, dtype=np.float32)
    radii = np.ascontiguousarray(radii, dtype=np.float32)
    albedos = np.ascontiguousarray(albedos, dtype=np.float32)
    materials = np.ascontiguousarray(materials, dtype=np.int32)
    roughnesses = np.ascontiguousarray(roughnesses, dtype=np.float32)
    obj_idxs = np.ascontiguousarray(obj_idxs, dtype=np.int32)

    print(f'scene has {len(scene)} objs, takes {time.time()-tscene:.4f} s to construct')
    
    
    rand_vec3 = random_unit_vector(int(1e6)) # Генерация случайных векторов

    start_time = time.time() # Запись времени начала трассировки лучей

    # Создание начальных лучей
    rays_o, rays_d = camera.get_rays(SAMPLES_PER_RAY) # Получение начальных лучей из камеры
    rays_o = np.ascontiguousarray(rays_o, dtype=np.float32) # Преобразование в массив
    rays_d = np.ascontiguousarray(rays_d, dtype=np.float32)
    rays_idx = np.arange(len(rays_o)) # Индексы лучей
    rays_color = np.ones_like(rays_o) # Инициализация цвета лучей

    bounce = 0 # Счетчик отражений
    while len(rays_o) > 1: # Пока есть лучи
        N = len(rays_o) # Количество лучей
        hit_record = np.zeros((N, 16), dtype=np.float32) # Запись результатов попаданий
        hit_record[:, 0] += MAX_RAY_LENGTH # Индекс 0 хранит длину попадания
        hit_record[:, 1] += -1 # Индекс 1 хранит объект попадания
        # Индексы 2~4 хранят позицию попадания
        # Индексы 5~7 хранят нормали (нормализованные векторы)
        hit_record[:, 8] += 1 # Индекс 8 хранит информацию о передней поверхности
        # Индексы 9~11 хранят направление рассеяния
        # Индексы 12~14 хранят альбедо
        # Индекс 15 хранит информацию о рассеянии или поглощении

        rand_vec3_ = rand_vec3[np.random.randint(rand_vec3.shape[0], size=len(rays_o)), :] # Случайные векторы для рассеяния
        
        for i in range(0, N, chunk): # Обработка лучей чанками для избежания переполнения памяти CUDA
            ray_sphere_intersect[blocks, threads_per_block] \
                (rays_o[i:i+chunk], rays_d[i:i+chunk], 1e-4, 
                 centers, radii, albedos, materials, 
                 roughnesses, obj_idxs, hit_record[i:i+chunk],
                 rand_vec3_[i:i+chunk]) # Вызов функции для проверки пересечения лучей со сферами

        # Поиск попаданий и непопаданий
        rays_valid = hit_record[:, 0]<MAX_RAY_LENGTH # Проверка на валидность попаданий
        rays_idx_hit = rays_idx[rays_valid] # (N) Индексы лучей, которые попали
        rays_idx_nohit = rays_idx[~rays_valid] # (N) Индексы лучей, которые не попали

        # Если луч не попал ни во что - цвет фона
        t = normalize(rays_d[~rays_valid])[:, 1:2]*0.5+0.5 # Нормализация направления
        # Установка цвета фона
        rays_color[rays_idx_nohit] *= (1.0-t)*np.array([1.0, 1.0, 1.0]) + t*np.array([0.5, 0.7, 1.0])

        bounce += 1 # Увеличение счетчика отражений
        if bounce >= MAX_RAY_BOUNCE: # Если достигнуто максимальное количество отражений
            rays_color[rays_idx_hit] = 0.0 # Черный цвет для лучей, которые не остановились
            break

        # Если луч попал во что-то, луч отражается в другом направлении для получения его цвета
        # Подготовка лучей для следующего отражения
        hit_record_ = hit_record[rays_valid] # Запись результатов попаданий
        # Поиск лучей, которые были рассеяны
        scatter = hit_record_[:, 15]>0 # Проверка на рассеяние

        # Если не произошло рассеяния (луч поглощен) - черный цвет
        rays_idx_nohit = rays_idx_hit[~scatter] # Индексы лучей, которые не рассеяны
        rays_color[rays_idx_nohit] = 0 # Установка черного цвета

        # Иначе продолжаем трассировку луча
        hit_record_ = hit_record_[scatter] # Запись результатов рассеянных лучей
        rays_o = np.ascontiguousarray(hit_record_[:, 2:5]) # Обновление начальных точек лучей
        rays_d = np.ascontiguousarray(hit_record_[:, 9:12]) # Обновление направлений лучей
        attenuation = hit_record_[:, 12:15] # Запись коэффициентов ослабления

        rays_idx_hit = rays_idx_hit[scatter] # Индексы лучей, которые рассеялись
        rays_color[rays_idx_hit] *= attenuation # Установка нового цвета с учетом ослабления

        rays_idx = rays_idx_hit # Обновление индексов лучей


    # Усреднение цвета
    rays_color = rays_color.reshape(-1, SAMPLES_PER_RAY, 3).mean(1)
    # Коррекция гаммы
    GAMMA = 2
    rays_color = rays_color**(1/GAMMA) # Применение гамма-коррекции

    end_time = time.time() # Запись времени окончания трассировки

    print(f'Rendering time: {end_time-start_time:.4f} s')
    print('Saving to test.bmp ...')
    plt.imsave('C:/Images/test.bmp', rays_color.reshape(img_wh[1], img_wh[0], 3))
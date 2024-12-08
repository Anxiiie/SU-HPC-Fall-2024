import numpy as np  
from random_utils import *  

def normalize(v):
    # Нормализация вектора 
    return v / (np.linalg.norm(v) + 1e-10)

class Camera:
    def __init__(self, 
                 lookfrom=np.float32([0, 0, 0]),  # Позиция камеры
                 lookat=np.float32([0, 0, -1]),   # Точка, на которую смотрит камера
                 vup=np.float32([0, 1, 0]),       # Вектор "вверх"
                 img_wh=(1280, 720),              # Ширина и высота изображения
                 fov=np.pi/2,                     # Поле зрения
                 focus_dist=0.,                   # Расстояние фокусировки
                 aperture=0.,                     # Диафрагма
                 ):

        self.origin = lookfrom  # Сохранение позиции камеры

        # Вычисление направляющих векторов
        self.w = normalize(lookfrom - lookat)  # Направление взгляда (вектор z)
        self.u = normalize(np.cross(vup, self.w))  # Вектор x 
        self.v = np.cross(self.w, self.u)  # Вектор y 
        
        # Определение расстояния фокусировки
        if focus_dist <= 0:
            self.focus_dist = np.linalg.norm(lookfrom - lookat)
        else:
            self.focus_dist = focus_dist
            
        self.lens_radius = aperture / 2  # Радиус объектива

        # Размеры изображения
        self.W = img_wh[0]
        self.H = img_wh[1]
        
        # Вычисление фокуса
        self.focal = 0.5 * self.W / np.tan(0.5 * fov)

        # Создание сетки пикселей
        self.i, self.j = np.mgrid[:self.H, :self.W][::-1]  # Обратный порядок для правильной ориентации

    def get_rays(self, samples_per_ray=1):
        # Генерация лучей из камеры
        i = np.stack([self.i] * samples_per_ray, -1)  # Массив индексов по высоте
        j = np.stack([self.j] * samples_per_ray, -1)  # Массив индексов по ширине

        # Если требуется больше одного образца на луч
        if samples_per_ray > 1:
            i = i + np.random.random(i.shape)  # Добавление случайного смещения по высоте
            j = j + np.random.random(j.shape)  # Добавление случайного смещения по ширине

        # Вычисление направлений лучей
        rays_d = (i[..., np.newaxis] - self.W / 2) / self.focal * self.focus_dist * self.u + \
                 -(j[..., np.newaxis] - self.H / 2) / self.focal * self.focus_dist * self.v + \
                 -self.focus_dist * self.w
                 
        rays_d = rays_d.reshape(-1, 3).astype(np.float32)  
        rays_o = np.tile(self.origin, (len(rays_d), 1))  # Позиция камеры для каждого луча

        # Смещение лучей для глубины резкости
        if self.lens_radius > 0:
            offset = self.lens_radius * random_in_unit_disk(len(rays_d))  
            offset = offset[:, 0:1] * self.u + offset[:, 1:2] * self.v  # Смещение по u и v
            rays_o += offset  # Применение смещения к началу лучей
            rays_d -= offset  # Применение смещения к направлениям лучей

        return rays_o, rays_d  # Возвращение начальных точек и направлений лучей

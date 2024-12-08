import numpy as np  

def random_in_unit_disk(N):
    phi = 2 * np.pi * np.random.random(N)  # Угол в полярной системе координат
    r = np.random.random(N) ** 0.5  # Радиус, взятый из равномерного распределения (с квадратным корнем для равномерного заполнения диска)
    
    # Вычисление координат x и y
    x = r * np.sin(phi)  
    y = r * np.cos(phi)  
    z = np.zeros_like(x)  # z-координата равна 0 (все векторы лежат в плоскости xy)
    
    # Возвращение стака координат (x, y, z) 
    return np.stack((x, y, z), axis=1).astype(np.float32)

def random_unit_vector(N):
    # Генерация N случайных векторов на поверхности единичной сферы
    a = 2 * np.pi * np.random.random(N)  # Угол в полярной системе координат
    z = np.random.random(N) * 2 - 1  # z-координата в диапазоне [-1, 1]
    r = (1 - z**2) ** 0.5  # Радиус в плоскости xy

    # Вычисление координат x, y и z
    return np.stack((r * np.cos(a), r * np.sin(a), z), axis=1).astype(np.float32)

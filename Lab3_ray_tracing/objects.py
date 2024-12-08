import numpy as np
from enum import Enum


class Material(Enum):
    MATTE = 0  # Матовый 
    METAL = 1  # Металлический 


class Sphere:
    def __init__(self, center, radius, 
                 albedo=np.zeros(3, dtype=np.float32),  # цвет по умолчанию - черный
                 material=Material.MATTE,  # Материал по умолчанию - матовый
                 roughness=0.0):  # Шероховатость по умолчанию - 0.0  
        self.center = center  
        self.radius = radius  
        self.albedo = albedo  
        self.material = material  
        self.roughness = roughness  


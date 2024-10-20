<h1 align="center">Лабораторная работа №1</h1>

<h2> Цель лабораторной работы: </h2>
<h5> Реализовать алгоритм перемножения матриц.
 Реализация должна содержать 2 функции перемножения матриц: на CPU и на GPU с применением CUDA.</h5>
<h6>Язык: Python</h6>
<h6>  
	Входные данные: 2 матрицы размером от 100х100 до 2000х2000 каждая.  
	
Выходные данные: проверка корректности перемножения + время вычисления</h6>

<h2> Описание проделанной работы: </h2>
<h5> 
В результате выполнения данной работы было реализовано 2 алгоритма перемножения матриц. 
Перемножение матриц на CPU стандартными средствами Python, а также перемножение матриц на GPU с применением CUDA.
</h5>

<h2> Что было распараллелено: </h2>
<h5> Каждый поток рассчитывает значение для отдельного элемента результирующего массива.</h5>

<h2> Как производились вычисления: </h2>
<h5> 

 
1. Поочередно генерировались массивы A и B с размерностями от 100х100 до 2000х2000 с шагом 100.  
	
2. Вычисляется результирующий массив на CPU.
   
3. С помощью функции time() определяется время затраченное на перемножение матриц на CPU.
   
4. Вычисляется результирующий массив на GPU.
   
5. С помощью функции time() определяется время затраченное на перемножение матриц на GPU с CUDA.
   
6. Производится проверка на корректность путём поэлементного сравнения массивов.

</h5>


<h2> Оборудование: </h2>
<h5>  
	
CPU: AMD Ryzen 7 PRO 2700 Eight-Core Processor  
	
GPU: NVIDIA GeForce GTX 1060 6GB  
</h5>  
 
<h2> Результаты представлены в таблице: </h2>

|Matrix Size|CPU Time (s)|GPU CUDA (s)|Correctness|Acceleration|  
|---|---|---|---|---|
|100x100|1.29805|0.57047|Yes|2.28096|  
|200x200|10.37842|0.01563|Yes|663.23436|  
|300x300|35.35782|0.02158|Yes|1647.51918|  
|400x400|83.62423|0.03246|Yes|2576.56427|  
|500x500|163.70377|0.04435|Yes|3697.68999|  
|600x600|281.63577|0.07511|Yes|3745.75133|  
|700x700|447.27207|0.11610|Yes|3842.13956|  
|800x800|675.03381|0.14181|Yes|4755.91758|  
|900x900|959.03934|0.19296|Yes|4977.79888|  
|1000x1000|1310.25777|0.16758|Yes|7820.79884|  
|1100x1100|1752.12545|0.22080|Yes|7934.16473|  
|1200x1200|2268.87949|0.25219|Yes|8992.48739|  
|1300x1300|2785.63353|0.28358|Yes|9826.80306|  
|1400x1400|3302.38757|0.31497|Yes|10401.92731|  
|1500x1500|3820.14161|0.34636|Yes|11056.33030|  
|1600x1600|4336.89565|0.37775|Yes|11484.68866|  
|1700x1700|4853.64969|0.40914|Yes|11800.46151|  
|1800x1800|5370.40373|0.44053|Yes|12192.89098|  
|1900x1900|5887.15777|0.47192|Yes|12475.86003|  
|2000x2000|6403.91181|0.50331|Yes|12721.22281|  

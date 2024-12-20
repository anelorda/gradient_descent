import numpy as np
import matplotlib.pyplot as plt

def custom_loss(x):
   return x**3 - 6*x**2 + 12*x - 8

def compute_gradient(x):
   return 3*x**2 - 12*x + 12

def optimize_params(init_value, alpha, max_steps):
   current_x = init_value
   path = [current_x]
   
   for step in range(max_steps):
       current_grad = compute_gradient(current_x)
       current_x = current_x - alpha * current_grad
       path.append(current_x)
       
   return current_x, path

init_value = 5.0
alpha = 0.01    
max_steps = 50    

best_x, path = optimize_params(init_value, alpha, max_steps)

print(f"Найденный минимум x: {best_x:.4f}")
print(f"Значение функции в минимуме: {custom_loss(best_x):.4f}")

x_plot = np.linspace(0, 6, 100)
y_plot = [custom_loss(x) for x in x_plot]

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, 'b-', label='Функция потерь')
plt.plot(path, [custom_loss(x) for x in path], 'ro-', label='Путь оптимизации')
plt.grid(True)
plt.legend()
plt.title('Процесс оптимизации кубической функции')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
import numpy as np
import matplotlib.pyplot as plt

def cost_function(y, y_hat):
    return np.sum((y - y_hat) ** 2) / len(y)

# Gradient descent function
def gradient_descent(x, y, epochs=1000, learning_rate=0.00001, stopping_threshold=1e-6):
    current_weight = np.random.rand()
    current_b = np.random.rand()
    previous_cost = float('inf')

    costs = []

    for i in range(epochs):
        y_hat = (current_weight * x) + current_b
        current_cost = cost_function(y, y_hat)

        if abs(previous_cost - current_cost) <= stopping_threshold:
            break
        
        previous_cost = current_cost
        costs.append(current_cost)

        weight_d = -(2 / len(x)) * np.sum(x * (y - y_hat))
        bias_d = -(2 / len(x)) * np.sum(y - y_hat)


        current_weight -= learning_rate * weight_d
        current_b -= learning_rate * bias_d

    plt.figure(figsize=(8,6))
    plt.plot(costs, label="Cost Function")
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.title("Gradient Descent Optimization")
    plt.legend()
    plt.show()

    return current_b, current_weight


x = np.random.randint(1, 1000, 100)  
y = 3 * x + 0.6653 + np.random.randn(100) * 10  


b, w = gradient_descent(x, y)
print(f"Final parameters: w = {w:.4f}, b = {b:.4f}")

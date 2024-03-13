from pyswarm import pso
from model import train_and_evaluate_model

def objective_function(hyperparameters):
    

    # Call the model training and evaluation function
    score = train_and_evaluate_model(hyperparameters)

    # Return the score to minimize (or negative if it's something to maximize)
    return -score 

def optimize_hyperparameters():
    lb = [1e-4, 0.1, 32, 3, 16]
    ub = [1e-2, 0.5, 128, 7, 128]
    
    xopt, fopt = pso(objective_function, lb, ub, swarmsize=10, maxiter=10, debug=True)
    
    return xopt, -fopt  # xopt contains the optimal hyperparameter values

if __name__ == '__main__':
    optimal_hyperparameters, best_score = optimize_hyperparameters()
    print(f"Optimal hyperparameters found: {optimal_hyperparameters}")
    print(f"Best score: {best_score}")

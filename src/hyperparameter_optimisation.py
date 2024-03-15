import json

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

    iterations = 100
    best_score = float('-inf')
    no_improvement = 0
    max_no_improvement = 20
    
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}")
        xopt, fopt = pso(objective_function, lb, ub, swarmsize=10, maxiter=1, debug=True)
        if -fopt < best_score:
            best_score = -fopt
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement > max_no_improvement:
                print(f"Optimization stopped after {i} iterations.")
                break

    optimal_hyperparameters = {
        "learning_rate": xopt[0],
        "dropout_rate": xopt[1],
        "num_filters": int(xopt[2]),
        "kernel_size": int(xopt[3]),
        "embedding_dim": int(xopt[4]),
        "best_score": -fopt
    }

    with open('./dataset/optimal_hyperparameters.json', 'w') as outfile:
        json.dump(optimal_hyperparameters, outfile, indent=4)
    
    return xopt, -fopt  # xopt contains the optimal hyperparameter values

if __name__ == '__main__':
    optimal_hyperparameters, best_score = optimize_hyperparameters()
    print(f"Optimal hyperparameters found: {optimal_hyperparameters}")
    print(f"Best score: {best_score}")

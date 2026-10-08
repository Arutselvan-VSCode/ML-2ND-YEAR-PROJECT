import numpy as np
try:
    import shap
except ImportError:
    shap = None

def optimize_rescue_route_qlearning(centroids, epochs=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    if centroids is None or len(centroids) < 2:
        return [0]
    
    n_zones = len(centroids)
    q_table = np.zeros((n_zones, n_zones))
    
    dist_matrix = np.zeros((n_zones, n_zones))
    for i in range(n_zones):
        for j in range(n_zones):
            if i != j:
                dist_matrix[i, j] = -np.linalg.norm(centroids[i] - centroids[j])

    for _ in range(epochs):
        state = np.random.randint(0, n_zones)
        visited = [state]
        
        while len(visited) < n_zones:
            unvisited = [z for z in range(n_zones) if z not in visited]
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.choice(unvisited)
            else:
                q_values = [q_table[state, a] for a in unvisited]
                action = unvisited[np.argmax(q_values)]
                
            reward = dist_matrix[state, action]
            best_next_q = np.max([q_table[action, next_a] for next_a in range(n_zones)]) if len(visited) < n_zones - 1 else 0
            q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * best_next_q - q_table[state, action])
            
            state = action
            visited.append(state)
            
    route = [0]
    curr = 0
    while len(route) < n_zones:
        unvisited = [z for z in range(n_zones) if z not in route]
        q_values = [q_table[curr, a] for a in unvisited]
        curr = unvisited[np.argmax(q_values)]
        route.append(curr)
        
    return route

def explain_model_predictions(model, feature_df):
    if shap is None:
        return {"Error": "SHAP library not installed."}
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_df)
    
    if isinstance(shap_values, list):
        mean_shap = np.abs(shap_values[0]).mean(axis=0)
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)
        
    return dict(zip(feature_df.columns, mean_shap))
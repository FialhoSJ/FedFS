import numpy as np

# ALGORITMO 2 - FEDERATED SERVER

class FederatedServer:

    def aggregate(self, probabilities, weights):
        probabilities = np.array(probabilities)
        # Clip para evitar divisão por zero no logit
        probabilities = np.clip(probabilities, 1e-4, 1 - 1e-4)

        # Transforma em odds ratio logarítmica (Log-odds / Logit)
        logits = np.log(probabilities / (1 - probabilities))

        # Média ponderada no espaço dos logits
        weights = np.array(weights) / np.sum(weights)
        avg_logits = np.zeros(probabilities.shape[1])
        for w, logit in zip(weights, logits):
            avg_logits += w * logit
            
        # Retorna ao espaço de probabilidade (função sigmoide)
        p_global = 1 / (1 + np.exp(-avg_logits))
        return p_global
    
class FederatedClient:
    def __init__(self, X, y, client_id):
        self.X = X
        self.y = y
        self.client_id = client_id
    def local_update(self, p_global):
        p_local = CE_feature_selection(X=self.X, y=self.y, p_init=p_global, T=3, S=60, alpha=0.2, beta=0.8)
        return p_local
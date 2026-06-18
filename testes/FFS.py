import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class FFS:
    def __init__(self, clients):
        self.clients = clients
        self.server = FederatedServer()

    def fit(self, rounds=10):
        m = self.clients[0].X.shape[1]
        p_global = np.ones(m) * 0.5  # Inicia com 50% de chance para cada feature

        for r in range(rounds):
            print(f"\nROUND {r+1}")
            local_probabilities = []
            weights = []

            for client in self.clients:
                p_local = client.local_update(p_global)
                local_probabilities.append(p_local)
                weights.append(len(client.y))

            p_global = self.server.aggregate(local_probabilities, weights)
            print("Probabilidades globais (amostra dos primeiros 10):")
            print(np.round(p_global[:10], 3))

        return p_global

if __name__ == "__main__":
    data = load_breast_cancer()
    X, y = data.data, data.target

    X1, X_temp, y1, y_temp = train_test_split(X, y, test_size=0.66, random_state=42)
    X2, X3, y2, y3 = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    client1 = FederatedClient(X1, y1, "Client-1")
    client2 = FederatedClient(X2, y2, "Client-2")
    client3 = FederatedClient(X3, y3, "Client-3")

    ffs = FFS([client1, client2, client3])
    p_final = ffs.fit(rounds=20)

    selected_features = np.where(p_final > 0.85)[0]
    rejected_features = np.where(p_final < 0.15)[0]
    
    print("===================")
    print("Resultados FFS:")
    print("===================")
    print(f"Features Selecionadas (p > 0.85): {selected_features}")
    print(f"Features Descartadas (p < 0.15): {rejected_features}")
    print(f"Total de features originais: {X.shape[1]} -> Selecionadas: {len(selected_features)}")
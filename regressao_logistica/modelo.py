import numpy as np


class ModeloRegressaoLogistica:
    def __init__(self, taxa_aprendizagem=0.01, n_iteracoes=1000):
        """
        Inicializa o ModeloRegressaoLogistica.

        Parâmetros:
        - taxa_aprendizagem (float): Taxa de aprendizagem para o algoritmo de
        otimização (default: 0.01).
        - n_iteracoes (int): Número de iterações para o algoritmo de
        otimização (default: 1000).
        """
        self.taxa_aprendizagem = taxa_aprendizagem
        self.n_iteracoes = n_iteracoes
        self.pesos = None
        self.vies = None

    def sigmoid(self, z):
        """
        Calcula a função sigmoid.

        Parâmetros:
        - z (float ou array): Valor(es) de entrada.

        Retorna:
        - float ou array: Valor(es) após a aplicação da função sigmoid.
        """
        return 1 / (1 + np.exp(-z))

    def ajuste(self, X, y):
        """
        Realiza o ajuste do modelo de regressão logística aos dados de
        treinamento.

        Parâmetros:
        - X (array): Array de shape (n_amostras, n_observacoes) contendo as
        amostras de treinamento.
        - y (array): Array de shape (n_amostras,) contendo as classes
        correspondentes às amostras de treinamento.
        """
        # Inicializando pesos e viés
        n_amostras, n_observacoes = X.shape
        self.pesos = np.zeros(n_observacoes)
        self.vies = 0

        # Gradiente descendente para otimização
        for _ in range(self.n_iteracoes):
            modelo_linear = np.dot(X, self.pesos) + self.vies
            y_previsto = self.sigmoid(modelo_linear)

            # Cálculo dos gradientes
            dw = (1 / n_amostras) * np.dot(X.T, (y_previsto - y))
            db = (1 / n_amostras) * np.sum(y_previsto - y)

            # Atualização de pesos e viés
            self.pesos -= self.taxa_aprendizagem * dw
            self.vies -= self.taxa_aprendizagem * db

    def previsao(self, X):
        """
        Realiza a previsão das classes para as amostras de entrada.

        Parâmetros:
        - X (array): Array de shape (n_amostras, n_observacoes) contendo as
        amostras de entrada.

        Retorna:
        - array: Array de shape (n_amostras,) contendo as classes previstas
        para as amostras de entrada.
        """
        modelo_linear = np.dot(X, self.pesos) + self.vies
        y_previsto = self.sigmoid(modelo_linear)
        y_previsto_classes = [1 if i > 0.5 else 0 for i in y_previsto]
        return y_previsto_classes

    def previsao_prob(self, X):
        """
        Realiza a previsão das probabilidades para as amostras de entrada.

        Parâmetros:
        - X (array): Array de shape (n_amostras, n_observacoes) contendo as
        amostras de entrada.

        Retorna:
        - array: Array de shape (n_amostras,) contendo as probabilidades
        previstas para as amostras de entrada.
        """
        modelo_linear = np.dot(X, self.pesos) + self.vies
        y_previsto = self.sigmoid(modelo_linear)
        return y_previsto

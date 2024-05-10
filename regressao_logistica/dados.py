import pandas as pd
from sklearn.datasets import load_breast_cancer


def carregando_dados():
    """
    Carrega um banco de dados de câncer de mama da biblioteca scikit-learn.

    Retorna:
    X (array): Array com as observações do banco de dados.
    y (array): Array com os resultados do banco de dados.
    """
    dados = load_breast_cancer()
    df = pd.DataFrame(dados.data, columns=dados.feature_names)
    df['target'] = dados.target
    X = df.drop(columns=['target']).values
    y = df['target'].values
    return X, y

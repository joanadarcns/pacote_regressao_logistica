import numpy as np


def avaliacao_residuos(y_verdadeiro, y_previsto):
    """
    Calcula a avaliação dos resíduos de uma regressão logística.

    Parâmetros:
    - y_verdadeiro: array-like
        Array contendo os valores verdadeiros do target.
    - y_previsto: array-like
        Array contendo os valores previstos do target.

    Retorna:
    - eqm: float
        Erro quadrático médio dos resíduos.
    - residuos: array-like
        Array contendo os resíduos calculados.

    Exemplo de uso:
    >>> y_verdadeiro = [1, 0, 1, 0]
    >>> y_previsto = [0.9, 0.1, 0.8, 0.2]
    >>> eqm, residuos = avaliacao_residuos(y_verdadeiro, y_previsto)
    >>> print(eqm)
    0.025
    >>> print(residuos)
    [0.1, -0.1, 0.2, -0.2]
    """
    residuos = y_verdadeiro - y_previsto
    eqm = np.mean(residuos ** 2)
    return eqm, residuos

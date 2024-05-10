import matplotlib.pyplot as plt
import seaborn as sns


def grafico_ajuste(X, y_verd, y_prev):
    """
    Plota um gráfico de dispersão dos valores verdadeiros (y_verd) e
    valores previstos (y_prev) em relação às observações (X).

    Parâmetros:
    - X (numpy.ndarray): Array de observações.
    - y_verd (numpy.ndarray): Array de valores verdadeiros.
    - y_prev (numpy.ndarray): Array de valores previstos.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X[:, 0], y=y_verd, label='Verdadeiro')
    sns.scatterplot(x=X[:, 0], y=y_prev, label='Previsto')
    plt.xlabel('Observação')
    plt.ylabel('Resultado')
    plt.title('Gráfico do Ajuste')
    plt.legend()
    plt.show()


def grafico_residuos(X, residuos):
    """
    Plota um gráfico de dispersão dos resíduos em relação às observações (X).

    Parâmetros:
    - X (numpy.ndarray): Array de observações.
    - residuos (numpy.ndarray): Array de resíduos.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X[:, 0], y=residuos)
    plt.xlabel('Observação')
    plt.ylabel('Resíduo')
    plt.title('Gráfico dos Resíduos')
    plt.show()

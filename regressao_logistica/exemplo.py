from logistic_regression import (
    ModeloRegressaoLogistica,
    carregando_dados,
    avaliacao_residuos,
    grafico_ajuste,
    grafico_residuos,
)

# Carrega os dados de teste
X, y = carregando_dados()

# Cria o modelo de regressão logística
modelo = ModeloRegressaoLogistica()

# Ajusta o modelo aos dados
modelo.ajuste(X, y)

# Prediz os resultados
y_previsto = modelo.previsao(X)

# Avalia o modelo
eqm, residuos = avaliacao_residuos(y, y_previsto)
print(f'Erro Quadrático Médio: {eqm}')

# Plota gráficos
grafico_ajuste(X, y, y_previsto)
grafico_residuos(X, residuos)

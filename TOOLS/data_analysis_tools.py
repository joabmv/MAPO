import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from langchain_core.tools import tool

@tool
def obter_dados_mercado(tickers: list[str], period: str = "12mo"):
    """Busca dados históricos de mercado para calcular retornos e a matriz de covariância.

    Esta função utiliza a biblioteca yfinance para baixar os preços de fechamento
    ajustados ('Adj Close') de uma lista de tickers de ativos. Com base nesses dados,
    calcula os retornos diários, e então anualiza a média dos retornos (retorno esperado)
    e a matriz de covariância (considerando 252 dias de negociação em um ano).

    Args:
        tickers (list[str]): Uma lista de códigos de ativos (tickers) a serem
            pesquisados, conforme o padrão do Yahoo Finance.
            Exemplo: ["PETR4.SA", "VALE3.SA", "ITUB4.SA"]

        period (str, optional): O período para o qual os dados históricos serão
            baixados. O padrão é "12mo" (12 meses). Formatos válidos incluem
            "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max".

    Returns:
        dict[str, dict]: Um dicionário contendo os retornos esperados e a matriz de
                         covariância, com as seguintes chaves:
            - "retornos_esperados" (dict[str, float]): Um dicionário onde as chaves
              são os tickers e os valores são os retornos anuais esperados.
              Ex: {'PETR4.SA': 0.22, 'VALE3.SA': 0.15}

            - "matriz_covariancia" (dict[str, list[float]]): A matriz de covariância
              anualizada, representada como um dicionário onde cada chave é um ticker
              e o valor é uma lista (linha da matriz) com as covariâncias
              correspondentes a todos os tickers.
              Ex: {'PETR4.SA': [0.09, 0.04], 'VALE3.SA': [0.04, 0.06]}
    """
    print(f"--- Ferramenta: Obtendo dados para {tickers} ---")
    try:
        data = yf.download(tickers, period=period, auto_adjust=False, progress=False)['Adj Close']
        if data.empty:
            return {"erro": f"Nenhum dado encontrado para os tickers {tickers} no período de {period}."}
        
        retornos_diarios = data.pct_change().dropna()
        
        # Garante que temos dados suficientes para o cálculo
        if retornos_diarios.empty:
            return {"erro": f"Não foi possível calcular os retornos diários para {tickers}. Verifique os dados históricos."}
        
        retornos_esperados = retornos_diarios.mean() * 252
        matriz_covariancia = retornos_diarios.cov() * 252
        
        # Converte a matriz de covariância para o formato de lista de listas, que é mais padrão para APIs
        matriz_cov_list = matriz_covariancia.values.tolist()

        return {
            "retornos_esperados": retornos_esperados.to_dict(),
            "matriz_covariancia": matriz_cov_list
        }
    except Exception as e:
        return {"erro": f"Ocorreu uma falha ao buscar os dados do mercado: {e}"}

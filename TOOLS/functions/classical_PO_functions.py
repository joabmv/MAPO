import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

def optimize_portfolio_mvo(
    tickers: list,
    start_date: str,
    end_date: str,
    num_portfolios: int = 10000,
    risk_free_rate: float = 0.0 # Taxa livre de risco, ex: 0.05 para 5%
) -> tuple:
    """
    Realiza a otimização de portfólio usando o método Média-Variância (MVO) de Markowitz.

    Args:
        tickers (list): Uma lista de símbolos de ticker dos ativos (ex: ['ITSA4.SA', 'PETR4.SA']).
        start_date (str): Data de início para os dados históricos (formato 'AAAA-MM-DD').
        end_date (str): Data de término para os dados históricos (formato 'AAAA-MM-DD').
        num_portfolios (int, optional): Número de portfólios aleatórios a simular. Padrão é 10000.
        risk_free_rate (float, optional): Taxa livre de risco anualizada para o cálculo do Sharpe Ratio.
                                           Padrão é 0.0 (0%).

    Returns:
        tuple: Uma tupla contendo:
            - portfolio_results_df (pd.DataFrame): DataFrame com os retornos, riscos e Sharpe Ratios de todos os portfólios simulados, junto com seus pesos.
            - max_sharpe_portfolio (pd.Series): Série pandas com os detalhes do portfólio de maior Sharpe Ratio.
            - min_risk_portfolio (pd.Series): Série pandas com os detalhes do portfólio de mínima variância.
    """

    print(f"--- Iniciando Otimização MVO para {tickers} de {start_date} a {end_date} ---")

    # 1. Baixar os dados históricos de preços
    print("Baixando dados históricos...")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
        if data.empty:
            print("Erro: Não foi possível baixar os dados. Verifique os tickers ou o período.")
            return None, None, None
    except Exception as e:
        print(f"Erro ao baixar dados: {e}")
        return None, None, None

    print("Dados baixados com sucesso!")
    # print(data.head()) # Removido para não poluir a saída da função

    # 2. Calcular os retornos diários e anuais
    log_returns = np.log(data / data.shift(1)).dropna()

    if log_returns.empty:
        print("Erro: Não há retornos válidos para o período selecionado.")
        return None, None, None

    annual_returns = log_returns.mean() * 252 # Assumindo 252 dias úteis no ano
    annual_cov_matrix = log_returns.cov() * 252

    # print("\nRetornos médios anuais:\n", annual_returns) # Removido para não poluir a saída da função
    # print("\nMatriz de covariância anual:\n", annual_cov_matrix) # Removido para não poluir a saída da função

    # 3. Simular múltiplos portfólios aleatórios
    results = np.zeros((3, num_portfolios))
    weights_record = []

    print(f"\nSimulando {num_portfolios} portfólios...")
    for i in range(num_portfolios):
        # Gerar pesos aleatórios
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)

        weights_record.append(weights)

        # Calcular retorno e risco do portfólio
        portfolio_return = np.sum(weights * annual_returns)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(annual_cov_matrix, weights)))

        # Calcular Sharpe Ratio
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev

        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        results[2, i] = sharpe_ratio

    # Criar DataFrame com os resultados
    portfolio_results_df = pd.DataFrame(results.T, columns=['Retorno', 'Risco', 'Sharpe Ratio'])
    portfolio_results_df['Pesos'] = weights_record

    # 4. Identificar o portfólio com maior Sharpe Ratio (ótimo)
    max_sharpe_portfolio = portfolio_results_df.loc[portfolio_results_df['Sharpe Ratio'].idxmax()]

    # 5. Identificar o portfólio com menor risco (mínima variância)
    min_risk_portfolio = portfolio_results_df.loc[portfolio_results_df['Risco'].idxmin()]


    print(f"--- Otimização MVO Concluída ---")
    return portfolio_results_df, max_sharpe_portfolio, min_risk_portfolio
import numpy as np
from scipy.optimize import minimize

from langchain_core.tools import tool

@tool
def otimizar_portfolio_Markowitz(retornos_esperados: dict, matriz_covariancia: list[list[float]]):
    """Optimizes an asset portfolio to maximize the Sharpe Ratio using the Markowitz model.

    This function calculates the optimal weights for a portfolio of assets in order to
    maximize the risk-adjusted return (Sharpe Ratio). The optimization
    assumes a risk-free rate of 2% (0.02) per year. The weights are
    constrained to sum to 1, and each individual weight must be between 0 and 1 (no short selling).

    Args:
        retornos_esperados (dict[str, float]): A dictionary where the keys are the
            asset tickers and the values are their respective annual expected
            returns in decimal format.
            Example: {"ASSET_A": 0.15, "ASSET_B": 0.20, "ASSET_C": 0.12}

        matriz_covariancia (list[list[float]]): A list of lists (matrix) representing
            the annual covariance between the assets. The order of rows and columns
            must be the same as the assets provided in `retornos_esperados`.
            Example: [[0.04, 0.018, 0.01], [0.018, 0.09, 0.03], [0.01, 0.03, 0.02]]

    Returns:
        dict[str, any]: A dictionary containing the optimization result with the following keys:
            - "tipo_otimizacao" (str): The optimization method used ("Classical (Sharpe Ratio)").
            - "pesos_otimos" (dict[str, float]): A dictionary mapping each asset ticker
            to its optimal weight (percentage) in the portfolio. The sum of the weights is 1.0.
            Example of return:
            {
                "tipo_otimizacao": "Classical (Sharpe Ratio)",
                "pesos_otimos": {
                    "ASSET_A": 0.5,
                    "ASSET_B": 0.3,
                    "ASSET_C": 0.2
                }
            }
    """
    print("--- Ferramenta: Executando Otimização Clássica (Markowitz) ---")
    
    # Extrai os tickers na ordem correta para garantir a correspondência
    tickers = list(retornos_esperados.keys())
    retornos = np.array([retornos_esperados[ticker] for ticker in tickers])
    
    # Converte a matriz de covariância para um array numpy
    cov_matrix = np.array(matriz_covariancia)
    num_assets = len(retornos)

    # Função objetivo: Maximizar o Sharpe Ratio (minimizando seu negativo)
    def neg_sharpe_ratio(weights, retornos, cov_matrix, risk_free_rate=0.02):
        p_ret = np.sum(retornos * weights)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        if p_vol == 0:
            return 0 # Evita divisão por zero se a volatilidade for nula
        return -(p_ret - risk_free_rate) / p_vol

    # Restrições: A soma dos pesos deve ser 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Limites: Cada peso deve estar entre 0 e 1 (sem posições vendidas)
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Pesos iniciais: Distribuição igual entre os ativos
    initial_weights = np.array([1./num_assets] * num_assets)

    # Executa a otimização
    result = minimize(neg_sharpe_ratio, initial_weights, args=(retornos, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    # Formata o resultado
    pesos_otimos = dict(zip(tickers, result.x))
    
    return {"tipo_otimizacao": "Clássica (Sharpe Ratio)", "pesos_otimos": pesos_otimos}

@tool
def otimizar_portfolio_risk_parity(tickers: list[str], matriz_covariancia: list[list[float]]):
    """
    Optimizes a portfolio using the Risk Parity approach.

    The objective is to allocate the weights in such a way that each asset contributes equally
    to the total portfolio risk, creating a more balanced portfolio in terms of risk.

    Args:
        tickers (list[str]): A list of the asset tickers.
        matriz_covariancia (list[list[float]]): The annualized covariance matrix
            between the assets.

    Returns:
        dict: A dictionary containing the optimal weights for the Risk Parity portfolio.
            Example: {"tipo_otimizacao": "Risk Parity", "pesos_otimos": {"ASSET_A": 0.6, "ASSET_B": 0.4}}
    """
    print("--- Ferramenta: Executando Otimização por Paridade de Risco (Risk Parity) ---")
    try:
        cov_matrix = np.array(matriz_covariancia)
        num_assets = len(tickers)

        def _calculate_risk_contribution(weights, cov_matrix):
            """Calcula a contribuição de risco de cada ativo."""
            weights = np.array(weights)
            portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
            # Contribuição Marginal de Risco
            mrc = (cov_matrix @ weights) / portfolio_volatility
            # Contribuição Total de Risco
            trc = weights * mrc
            return trc

        def _risk_parity_objective(weights, cov_matrix):
            """Função objetivo para o otimizador: minimizar a variância das contribuições de risco."""
            trc = _calculate_risk_contribution(weights, cov_matrix)
            # Queremos que todas as contribuições de risco sejam iguais.
            # Minimizar a soma das diferenças quadráticas entre elas atinge esse objetivo.
            return np.sum((trc - trc.mean())**2)

        # Restrições: pesos somam 1 e não são negativos.
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        
        # Pesos iniciais: distribuição igual.
        initial_weights = np.array([1.0 / num_assets] * num_assets)

        # Otimização
        result = minimize(
            fun=_risk_parity_objective,
            x0=initial_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            raise Exception(f"Otimização de Paridade de Risco falhou: {result.message}")

        pesos_otimos = {ticker: round(weight, 4) for ticker, weight in zip(tickers, result.x)}
        
        return {"tipo_otimizacao": "Paridade de Risco", "pesos_otimos": pesos_otimos}

    except Exception as e:
        return {"erro": f"Falha na otimização de Paridade de Risco: {e}"}



@tool
def otimizar_portfolio_minima_variancia(
    tickers: list[str],
    matriz_covariancia: list[list[float]]
):
    """
    Optimizes a portfolio to find the combination of weights that results in the Minimum Variance (lowest possible risk).
    This method does not use expected returns, focusing exclusively on the risk structure
    and correlation of the assets to create the most defensive portfolio possible.

    Args:
        tickers (list[str]): A list of the asset tickers.
        matriz_covariancia (list[list[float]]): The annualized covariance matrix
            between the assets.

    Returns:
        dict: A dictionary containing the optimal weights for the Minimum Variance portfolio.
            Example: {"tipo_otimizacao": "Minimum Variance", "pesos_otimos": {"ASSET_A": 0.7, "ASSET_B": 0.3}}
    """
    print("--- Ferramenta: Executando Otimização de Mínima Variância ---")
    try:
        cov_matrix = np.array(matriz_covariancia)
        num_assets = len(tickers)

        def portfolio_variance(weights, cov_matrix):
            """Função objetivo: calcula a variância do portfólio."""
            return weights.T @ cov_matrix @ weights

        # Restrições: pesos somam 1 e não são negativos (sem venda a descoberto).
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        
        # Pesos iniciais: distribuição igual.
        initial_weights = np.array([1.0 / num_assets] * num_assets)

        # Otimização para minimizar a função de variância
        result = minimize(
            fun=portfolio_variance,
            x0=initial_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            raise Exception(f"Otimização de Mínima Variância falhou: {result.message}")

        pesos_otimos = {ticker: round(weight, 4) for ticker, weight in zip(tickers, result.x)}
        
        return {"tipo_otimizacao": "Mínima Variância", "pesos_otimos": pesos_otimos}

    except Exception as e:
        return {"erro": f"Falha na otimização de Mínima Variância: {e}"}
# Arquivo: ferramentas_quanticas_pennylane.py
from langchain_core.tools import tool
import pennylane as qml
from pennylane import numpy as np
from typing import Dict, List, Any, Tuple

@tool
def otimizar_portfolio_qaoa(
    retornos_esperados: Dict[str, float],
    matriz_covariancia: List[List[float]],
    budget: int = 3,
    camadas_qaoa: int = 5,
    passos_otimizador: int = 150
) -> Dict[str, Any]:
    """
    Optimizes a portfolio using the QAOA algorithm to select a fixed number of assets.

    The function seeks the combination of assets that maximizes the risk-adjusted return,
    given a fixed budget (number of assets to be chosen).

    Args:
        retornos_esperados (Dict[str, float]): A dictionary where the keys are the
            asset tickers and the values are their respective annual expected
            returns in decimal format.
            Example: {"ASSET_A": 0.15, "ASSET_B": 0.20, "ASSET_C": 0.12}

        matriz_covariancia (List[List[float]]): A list of lists (matrix) representing
            the annual covariance between the assets. The order of rows and columns
            must be the same as the assets provided in `retornos_esperados`.
            Example: [[0.04, 0.018, 0.01], [0.018, 0.09, 0.03], [0.01, 0.03, 0.02]]
            
        budget (int): The exact number of assets to be selected for the portfolio. Defaults to 3.

        camadas_qaoa (int): The number of layers (p) for the QAOA circuit. Defaults to 5.
        
        passos_otimizador (int): The number of steps for the classical optimizer. Defaults to 150.


    Returns:
        Dict[str, Any]: A dictionary containing the optimization result with the
            following keys:
            - "tipo_otimizacao" (str): The optimization method used ("QUBO - QAOA").
            - "pesos_otimos" (Dict[str, float]): A dictionary mapping each ticker 
            to its weight (1.0 for selected, 0.0 for not selected).
    """
    
    # --- 1. Preparação dos Dados e Parâmetros ---
    tickers = list(retornos_esperados.keys())
    num_assets = len(tickers)
    
    mu = np.array(list(retornos_esperados.values()), requires_grad=False)
    sigma = np.array(matriz_covariancia, requires_grad=False)

    # Parâmetros do Hamiltoniano
    gamma = 1.0  # Coeficiente de aversão ao risco
    penalty = 2.0 # Penalidade por violar a restrição de budget

    # --- Funções Auxiliares Aninhadas ---
    def build_hamiltonian():
        zz_ops = [qml.PauliZ(i) @ qml.PauliZ(j) for i in range(num_assets) for j in range(i + 1, num_assets)]
        zz_coeffs = [0.5 * (gamma * sigma[i, j] + penalty) for i in range(num_assets) for j in range(i + 1, num_assets)]
        
        z_ops = [qml.PauliZ(i) for i in range(num_assets)]
        z_coeffs = [-0.5 * mu[i] + 0.5 * gamma * (np.trace(sigma) - np.sum(sigma[i, :])) + penalty * (budget - 0.5 * (num_assets - 1)) for i in range(num_assets)]

        return qml.Hamiltonian(zz_coeffs + z_coeffs, zz_ops + z_ops)

    def create_circuits(hamiltonian):
        dev = qml.device("default.qubit", wires=num_assets)

        def qaoa_layer(gamma_p, beta_p):
            qml.ApproxTimeEvolution(hamiltonian, gamma_p, 1)
            for w in range(num_assets):
                qml.RX(2 * beta_p, wires=w)

        @qml.qnode(dev)
        def cost_circuit(params):
            for w in range(num_assets): qml.Hadamard(wires=w)
            for i in range(camadas_qaoa): qaoa_layer(params[i], params[i + camadas_qaoa])
            return qml.expval(hamiltonian)

        @qml.qnode(dev)
        def probability_circuit(params):
            for w in range(num_assets): qml.Hadamard(wires=w)
            for i in range(camadas_qaoa): qaoa_layer(params[i], params[i + camadas_qaoa])
            return qml.probs(wires=range(num_assets))
            
        return cost_circuit, probability_circuit

    # --- 2. Construção do Hamiltoniano e Circuitos ---
    portfolio_hamiltonian = build_hamiltonian()
    cost_fn, prob_fn = create_circuits(portfolio_hamiltonian)

    # --- 3. Otimização Clássica ---
    print(f"Iniciando otimização QAOA para {num_assets} ativos com um budget de {budget}...")
    optimizer = qml.AdamOptimizer(stepsize=0.05)
    params = np.random.uniform(low=0, high=2 * np.pi, size=2 * camadas_qaoa, requires_grad=True)

    for i in range(passos_otimizador):
        params, cost = optimizer.step_and_cost(cost_fn, params)
        if (i + 1) % 50 == 0:
            print(f"  Passo {i+1:3d}, Custo (H): {cost:.4f}")

    print("Otimização QAOA concluída.")

    # --- 4. Análise e Formatação do Resultado ---
    final_probabilities = prob_fn(params)
    
    # Ordena os estados pela maior probabilidade
    portfolio_states = sorted(
        [(prob, format(i, f'0{num_assets}b')) for i, prob in enumerate(final_probabilities)],
        key=lambda x: x[0],
        reverse=True
    )
    
    # Encontra o portfólio mais provável que satisfaz a restrição de budget
    melhor_portfolio_str = None
    for _, portfolio_str in portfolio_states:
        if sum(int(bit) for bit in portfolio_str) == budget:
            melhor_portfolio_str = portfolio_str
            break
            
    # Caso nenhum portfólio válido seja o mais provável, usa o mais provável de todos
    if melhor_portfolio_str is None:
        print("Aviso: Nenhum portfólio com o budget exato foi a solução ótima. Retornando o mais provável.")
        melhor_portfolio_str = portfolio_states[0][1]

    # Formata os pesos ótimos (1.0 para selecionado, 0.0 para não selecionado)
    pesos_otimos = {ticker: float(bit) for ticker, bit in zip(tickers, melhor_portfolio_str)}
    
    resultado = {
        "tipo_otimizacao": "QUBO - QAOA",
        "pesos_otimos": pesos_otimos
    }

    return resultado

@tool
def otimizar_portfolio_vqe(
    retornos_esperados: Dict[str, float],
    matriz_covariancia: List[List[float]],
    budget: int = 3,
    camadas_vqe: int = 4,
    passos_otimizador: int = 150
) -> Dict[str, Any]:
    """
    Otimiza um portfólio usando o algoritmo VQE para selecionar um número fixo de ativos.

    A função busca a combinação de ativos que minimiza o Hamiltoniano de custo, 
    representando um balanço entre retorno e risco, sujeito a uma restrição de orçamento.

    Args:
        retornos_esperados (Dict[str, float]): Dicionário com tickers e seus retornos esperados.
            Exemplo: {"ATIVO_A": 0.15, "ATIVO_B": 0.20}
        matriz_covariancia (List[List[float]]): Matriz de covariância entre os ativos.
            A ordem deve corresponder aos tickers em `retornos_esperados`.
        budget (int): O número exato de ativos a serem selecionados. O padrão é 3.
        camadas_vqe (int): O número de camadas para o ansatz do VQE. O padrão é 4.
        passos_otimizador (int): O número de passos para o otimizador clássico. O padrão é 150.

    Returns:
        Dict[str, Any]: Um dicionário com o resultado da otimização, incluindo:
            - "tipo_otimizacao" (str): O método utilizado ("QUBO - VQE").
            - "pesos_otimos" (Dict[str, float]): Mapeamento de cada ticker para seu peso 
              (1.0 para selecionado, 0.0 para não selecionado).
    """
    
    # --- 1. PREPARAÇÃO DOS DADOS E PARÂMETROS ---
    tickers = list(retornos_esperados.keys())
    num_assets = len(tickers)
    
    mu = np.array(list(retornos_esperados.values()), requires_grad=False)
    sigma = np.array(matriz_covariancia, requires_grad=False)

    gamma = 1.0  # Coeficiente de aversão ao risco
    penalty = 2.0 # Penalidade por violar a restrição de budget

    # --- FUNÇÕES AUXILIARES ANINHADAS ---
    def build_hamiltonian():
        """Constrói o Hamiltoniano de Ising a partir dos dados do portfólio."""
        zz_ops = [qml.PauliZ(i) @ qml.PauliZ(j) for i in range(num_assets) for j in range(i + 1, num_assets)]
        zz_coeffs = [0.5 * (gamma * sigma[i, j] + penalty) for i in range(num_assets) for j in range(i + 1, num_assets)]
        
        z_ops = [qml.PauliZ(i) for i in range(num_assets)]
        z_coeffs = [-0.5 * mu[i] + 0.5 * gamma * (np.trace(sigma) - np.sum(sigma[i, :])) + penalty * (budget - 0.5 * (num_assets - 1)) for i in range(num_assets)]

        return qml.Hamiltonian(zz_coeffs + z_coeffs, zz_ops + z_ops)

    def create_circuits_vqe(hamiltonian):
        """Cria os circuitos VQE (ansatz, custo e probabilidade)."""
        dev = qml.device("default.qubit", wires=num_assets)
        num_params = camadas_vqe * num_assets

        def ansatz(params):
            """Define o circuito variacional (ansatz) para o VQE."""
            # Parâmetros são remodelados para facilitar o acesso por camada e qubit
            weights = params.reshape(camadas_vqe, num_assets)
            for layer in range(camadas_vqe):
                # Camada de rotações
                for i in range(num_assets):
                    qml.RY(weights[layer, i], wires=i)
                # Camada de emaranhamento
                for i in range(num_assets - 1):
                    qml.CNOT(wires=[i, i + 1])
        
        @qml.qnode(dev)
        def cost_circuit(params):
            """Executa o ansatz e calcula o valor esperado do Hamiltoniano."""
            ansatz(params)
            return qml.expval(hamiltonian)

        @qml.qnode(dev)
        def probability_circuit(params):
            """Executa o ansatz e retorna as probabilidades do estado final."""
            ansatz(params)
            return qml.probs(wires=range(num_assets))
            
        return cost_circuit, probability_circuit, num_params

    # --- 2. CONSTRUÇÃO DO HAMILTONIANO E CIRCUITOS ---
    portfolio_hamiltonian = build_hamiltonian()
    cost_fn, prob_fn, num_params = create_circuits_vqe(portfolio_hamiltonian)

    # --- 3. OTIMIZAÇÃO CLÁSSICA (VQE) ---
    print(f"Iniciando otimização VQE para {num_assets} ativos com um budget de {budget}...")
    optimizer = qml.AdamOptimizer(stepsize=0.05)
    params = np.random.uniform(low=0, high=2 * np.pi, size=num_params, requires_grad=True)

    for i in range(passos_otimizador):
        params, cost = optimizer.step_and_cost(cost_fn, params)
        if (i + 1) % 50 == 0:
            print(f"  Passo {i+1:3d}, Custo (H): {cost:.4f}")

    print("Otimização VQE concluída.")

    # --- 4. ANÁLISE E FORMATAÇÃO DO RESULTADO ---
    final_probabilities = prob_fn(params)
    
    portfolio_states = sorted(
        [(prob, format(i, f'0{num_assets}b')) for i, prob in enumerate(final_probabilities)],
        key=lambda x: x[0],
        reverse=True
    )
    
    melhor_portfolio_str = None
    for _, portfolio_str in portfolio_states:
        if sum(int(bit) for bit in portfolio_str) == budget:
            melhor_portfolio_str = portfolio_str
            break
            
    if melhor_portfolio_str is None:
        print("Aviso: Nenhum portfólio com o budget exato foi a solução ótima. Retornando o mais provável.")
        melhor_portfolio_str = portfolio_states[0][1]

    pesos_otimos = {ticker: float(bit) for ticker, bit in zip(tickers, melhor_portfolio_str)}
    
    resultado = {
        "tipo_otimizacao": "QUBO - VQE",
        "pesos_otimos": pesos_otimos
    }

    return resultado
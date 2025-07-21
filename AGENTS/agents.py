from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

# Importar as ferramentas
from TOOLS.data_analysis_tools import obter_dados_mercado
from TOOLS.classical_PO_tools import (
    otimizar_portfolio_Markowitz, 
    otimizar_portfolio_risk_parity, 
    otimizar_portfolio_minima_variancia
)
from TOOLS.quantum_PO_tools import otimizar_portfolio_qaoa, otimizar_portfolio_vqe
from configs import llm


# Prompts otimizados e mais concisos
DATA_SCIENTIST_PROMPT = """
**Data Specialist Agent**

Execute data tasks only. Return raw tool output without explanatory text.

**Rules:**
- Default to 12-month period if time not specified
- Use tool defaults for missing non-time parameters
- Return error if critical data missing
- Scope: data collection/analysis/visualization only

**Output:** Direct tool result or error message only.
"""

OPTIMIZER_PROMPT_BASE = """
**{agent_type} Portfolio Optimizer**

Execute {optimization_type} optimization and return raw results.

**Rules:**
- Requires: expected_returns + covariance_matrix
- Return error if data missing: "Required data not provided"
- Scope: {scope_restriction} only
- Output: Raw JSON optimization results only

**No explanatory text - results only.**
"""

SUPERVISOR_PROMPT = """
**Portfolio Analysis Supervisor**

Orchestrate specialist agents for portfolio optimization tasks.

**Agents:**
- data_scientist_agent: Market data (expected_returns, covariance_matrix)  
- portfolio_optimizer_agent: Classical optimization (Markowitz, Risk Parity, Min Variance)
- quantum_portfolio_optimizer_agent: Quantum optimization (QAOA, VQE)

**Process:**
1. Analyze user request + conversation history
2. Plan sequential execution (data → optimization)
3. Execute one task at a time, await completion
4. Synthesize final comprehensive response

**Key Rules:**
- Only use on agent at time
- Only use the optmization models requested
- Always get data first if only tickers provided
- Only provide the tickers and time period to the data scientist agent; do not mention anything about portfolio optimization or models for this agent.
- Default to 1-year period for data requests
- Pass expected_returns + covariance_matrix to optimizers
- If you have only the ticks follow these, at the data_scientist_agent use the ticks and time period of 12mo, and at portfolio_optimizer_agent use Markowitz
- Use quantum agent only for explicit quantum requests
- Always display the optimized results, if many optimization models ware requested, display all of then and compare.
- Silent operation - only final results or terminal errors
- Answer general questions directly without agents

**Output Format:**
Clear summary with numerical results + methodology explanation.
"""


# Criação dos agentes otimizada
def create_portfolio_agents():
    """Factory function para criar agentes de forma eficiente."""
    
    agents = {}
    
    # Data Scientist Agent
    agents['data_scientist'] = create_react_agent(
        model=llm,
        tools=[obter_dados_mercado],
        prompt=DATA_SCIENTIST_PROMPT,
        name="data_scientist_agent",
    )
    
    # Portfolio Optimizer Agent
    agents['portfolio_optimizer'] = create_react_agent(
        model=llm,
        tools=[
            otimizar_portfolio_Markowitz, 
            otimizar_portfolio_risk_parity, 
            otimizar_portfolio_minima_variancia
        ],
        prompt=OPTIMIZER_PROMPT_BASE.format(
            agent_type="Classical",
            optimization_type="classical",
            scope_restriction="portfolio optimization"
        ),
        name="portfolio_optimizer_agent",
    )
    
    # Quantum Portfolio Optimizer Agent
    agents['quantum_optimizer'] = create_react_agent(
        model=llm,
        tools=[otimizar_portfolio_qaoa, otimizar_portfolio_vqe],
        prompt=OPTIMIZER_PROMPT_BASE.format(
            agent_type="Quantum",
            optimization_type="quantum",
            scope_restriction="quantum portfolio optimization"
        ),
        name="quantum_portfolio_optimizer_agent",
    )
    
    return agents


# Criação otimizada do sistema
def create_portfolio_system():
    """Cria o sistema completo de otimização de portfolio."""
    
    # Criar agentes
    agents = create_portfolio_agents()
    
    # Criar supervisor
    supervisor = create_supervisor(
        model=llm,
        agents=list(agents.values()),
        prompt=SUPERVISOR_PROMPT,
        add_handoff_back_messages=True,
        output_mode="full_history",
    )
    
    return supervisor, agents


# Instanciação única e eficiente
portfolio_system, portfolio_agents = create_portfolio_system()

# Aliases para compatibilidade com código existente
data_scientist_agent = portfolio_agents['data_scientist']
portfolio_optimizer_agent = portfolio_agents['portfolio_optimizer'] 
quantum_portfolio_optimizer_agent = portfolio_agents['quantum_optimizer']
supervisor = portfolio_system


# Utilitário para verificação rápida do sistema
def system_health_check():
    """Verifica se todos os componentes estão funcionais."""
    components = {
        'Data Scientist': data_scientist_agent,
        'Portfolio Optimizer': portfolio_optimizer_agent,
        'Quantum Optimizer': quantum_portfolio_optimizer_agent,
        'Supervisor': supervisor
    }
    
    status = {}
    for name, component in components.items():
        status[name] = "OK" if component is not None else "ERROR"
    
    return status


if __name__ == "__main__":
    # Teste de integridade do sistema
    print("Portfolio Analysis System - Health Check:")
    for component, status in system_health_check().items():
        print(f"  {component}: {status}")
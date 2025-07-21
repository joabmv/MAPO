<p align="center">
  <img width="250" height="250" alt="CacheLib" src=https://i.imgur.com/PbI5qbp.png>
</p>

<h1 align="center"> Multi-agent Applied to Portfolio Optimization (MAPO) 🧠✨ </h1>
<p align="center"> An innovative multi-agent LLM system for intelligent financial portfolio optimization, blending classical algorithms with cutting-edge quantum computing. </p>
<p align="center">
<img loading="lazy" src="http://img.shields.io/static/v1?label=STATUS&message=EM%20DESENVOLVIMENTO&color=GREEN&style=for-the-badge"/>
</p>

This application employs a multi-agent LLM system for advanced financial portfolio optimization, seamlessly integrating classical algorithms with cutting-edge quantum computing tools to analyze markets, formulate strategies, and dynamically manage portfolios for enhanced, intelligent financial decision-making.

# 🛠️ Getting Started

Follow these steps to get your local copy up and running.

1. Clone the repository:
  ```
  git clone https://github.com/joabmv/MAPO.git
  cd MAPO
  ```
2. Install dependencies:    
It's recommended to use a virtual environment.
  ```
  python -m venv venv
  source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
  pip install -r requirements.txt
  ```
3. Set up environment variables:  
Create a .env file in the root directory and add your API keys.

  ```
  # .env
  GROQ_API_KEY="your-groq-api-key"
  ANOTHER_API_KEY="your-other-key"
  ```
4. Configure the LLM:  
Open configs.py and set up your chosen LLM provider. The project is pre-configured to use Groq, but you can easily adapt it for other providers like OpenAI or Anthropic.

Example using Groq in configs.py:

  ```
  import os
  from dotenv import load_dotenv
  from langchain_groq import ChatGroq
  load_dotenv()


  # Load API key from the .env file
  GROQ_API_KEI = os.getenv("GROQ_API_KEY")
  
  # Configure the LLM instance
  llm = ChatGroq(
      model='meta-llama/llama-4-maverick-17b-128e-instruct', # Or another model of your choice
      api_key=GROQ_API_KEY,
      temperature=0.2
  )
  ```

# ⚙️ Usage

Once the configuration is complete, you can run the main application script.

  ```
  python main.py
  ```

# 🔗 Model

<p align="center">
  <img width="700" height="800" alt="CacheLib" src=https://i.imgur.com/jlcN8tP.jpeg>
</p>

# 📖 Input examples

## User Input:
  ```
  Optimize this portfolio: AAPL, MSFT, GOOGL, NVDA using the Risk Parity optimizer, Markowitz optimizer, and Minimum Variance optimizer. Show me the results comparing all three, using a 1-year period.
  ```
## Expected output:

<p align="center">
  <img width="700" height="800" alt="CacheLib" src=https://i.imgur.com/DXOomvc.jpeg>
</p>



## User Input:
  ```
  Optimize this portfolio: AAPL, MSFT, GOOGL, NVDA. Compare the results of the Risk Parity and the VQE optmizations.
  ```

## Expected output:

<p align="center">
  <img width="700" height="800" alt="CacheLib" src=https://i.imgur.com/P2uOUGz.jpeg>
</p>

# 🖊️ Author
| [<img loading="lazy" src="https://avatars.githubusercontent.com/u/87044734?v=4" width=115><br><sub>Joab Varela</sub>](https://github.com/joabmv) |
| :---: |



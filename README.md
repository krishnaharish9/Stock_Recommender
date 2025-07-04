# 📈 Stock Recommender using Mistral + FinBERT + Yahoo Finance + TradingView

This project is a Gen AI-powered intelligent stock recommendation tool that combines:

- ✅ **Fundamental analysis** via Yahoo Finance
- ✅ **Technical indicators** from TradingView
- ✅ **Sentiment analysis** using FinBERT
- ✅ **Conversational insights** powered by the open-source **Mistral-7B GPTQ** model

---

## 🚀 Features

🔹 **Conversational Q&A**: Ask natural language questions about any stock — fundamentals, technicals, or news  
🔹 **Multi-source Analysis**: Combines real-time data, financial ratios, indicators, and sentiment  
🔹 **LLM-Powered Insights**: Uses Mistral-7B GPTQ model with GPU for fast, explainable responses  
🔹 **FinBERT Sentiment**: Analyzes latest financial news to detect positive, negative, and neutral tones  
🔹 **API Integration**:
- [x] Yahoo Finance (`yfinance`)
- [x] TradingView (`tradingview-ta`)
- [x] NewsAPI
- [x] Alpha Vantage

---

## 🧠 LLM Stack

| Component | Model Used |
|----------|------------|
| Language Model | `Mistral-7B-Instruct-v0.2` (GPU) |
| Sentiment Model | `yiyanghkust/finbert-tone` |
| Vector DB | *Not yet integrated* |
| Backend Framework | Standalone Python CLI |

---

## 🗃️ Directory Structure

Stock_Recommender/
├── .env # API keys (ignored by Git)
├── .gitignore
├── README.md
├── requirements.txt
└── Stock_Recommendation.py



---

## 🔐 .env Configuration

Create a `.env` file in the root directory with the following:
```env
ALPHA_VANTAGE_KEY=your_alpha_vantage_key
NEWS_API_KEY=your_news_api_key
HF_API_TOKEN=your_huggingface_token


# Clone the repo
git clone https://github.com/krishnaharish9/Stock_Recommender.git
cd Stock_Recommender

# (Optional) Create virtual environment
conda create -n stock_recommender python=3.10
conda activate stock_recommender

# Install dependencies
pip install -r requirements.txt

# Run the tool
python Stock_Recommendation.py


[INFO] Ticker: AAPL
You: What does a PE ratio of 30 mean?
Mistral: A PE ratio of 30 means the stock is priced at 30 times its earnings, indicating growth expectations...


🧪 Next Improvements (Planned)
 Web-based Streamlit interface

 Add VectorDB + RAG for knowledge-based Q&A

 Dockerize for easy deployment

 Add logging and monitoring

 Extend to Crypto or Commodities

🤝 Credits
TheBloke/Mistral-7B-Instruct-v0.2-GPTQ

yiyanghkust/finbert-tone

Yahoo Finance

TradingView TA

NewsAPI





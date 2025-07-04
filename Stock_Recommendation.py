import os
import torch
import requests
import yfinance as yf
from tradingview_ta import TA_Handler, Interval
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv

load_dotenv()
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
HF_TOKEN = os.getenv("HF_API_TOKEN")

# Load Mistral LLM on GPU
llm_tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    trust_remote_code=True,
    token=HF_TOKEN
)
llm_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    token=HF_TOKEN
)

# Load FinBERT Sentiment Analyzer
sent_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone",token=HF_TOKEN)
sent_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone",token=HF_TOKEN)
sent_pipeline = pipeline("sentiment-analysis", model=sent_model, tokenizer=sent_tokenizer)

def get_llm_response(prompt: str) -> str:
    """
    Generate response using Mistral-7B on GPU.
    """
    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(llm_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output = llm_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            eos_token_id=llm_tokenizer.eos_token_id,
            pad_token_id=llm_tokenizer.eos_token_id
        )
    return llm_tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()


def fetch_stock_data(symbol: str, start_date: str, end_date: str):
    """
    Downloads historical stock data using Yahoo Finance.

    Args:
        symbol (str): Ticker symbol.
        start_date (str): Start date in YYYY-MM-DD.
        end_date (str): End date in YYYY-MM-DD.

    Returns:
        pd.DataFrame: Historical stock price data.
    """
    return yf.download(symbol, start=start_date, end=end_date)


def perform_fundamental_analysis(symbol: str, start_date: str, end_date: str) -> dict:
    """
    Performs fundamental analysis on the given stock using Yahoo Finance.

    Args:
        symbol (str): Stock ticker symbol.
        start_date (str): Start date for price history.
        end_date (str): End date for price history.

    Returns:
        dict: Dictionary of key financial ratios and metrics.
    """
    stock_data = fetch_stock_data(symbol, start_date, end_date)
    ticker = yf.Ticker(symbol)
    financials = ticker.financials
    balance_sheet = ticker.balance_sheet
    info = ticker.info

    # Growth calculations
    revenue_growth = (stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[0]) / stock_data['Close'].iloc[0]
    earnings_growth = (financials.loc['Net Income'].iloc[-1] - financials.loc['Net Income'].iloc[0]) / financials.loc['Net Income'].iloc[0]

    revenue = financials.loc['Total Revenue'].iloc[0]
    net_income = financials.loc['Net Income'].iloc[0]
    total_assets = balance_sheet.loc['Total Assets'].iloc[0]
    liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0]
    equity = total_assets - liabilities
    current_assets = total_assets - balance_sheet.loc['Total Non Current Assets'].iloc[0]
    current_liabilities = balance_sheet.loc['Current Liabilities'].iloc[0]

    # Market Cap Calculation with safeguards
    closing_price = stock_data['Close'].iloc[-1]
    shares_outstanding = info.get('sharesOutstanding', None)

    if shares_outstanding and isinstance(closing_price, (int, float)):
        market_cap = closing_price * shares_outstanding
    else:
        market_cap = 0

    # Determine market cap category
    if market_cap > 10_000_000_000:
        cap_category = "Large Cap"
    elif market_cap > 2_000_000_000:
        cap_category = "Mid Cap"
    elif market_cap > 300_000_000:
        cap_category = "Small Cap"
    else:
        cap_category = "Micro Cap"

    return {
        "PE Ratio": info.get("trailingPE"),
        "PB Ratio": info.get("priceToBook"),
        "EPS": info.get("trailingEps"),
        "Industry PE": info.get("enterpriseToEbitda"),
        "Revenue Growth Rate": revenue_growth,
        "Earnings Growth Rate": earnings_growth,
        "Gross Profit Margin": (net_income / revenue) * 100,
        "Net Profit Margin": (net_income / revenue) * 100,
        "ROE": (net_income / equity) * 100,
        "ROA": (net_income / total_assets) * 100,
        "Debt-to-Equity Ratio": liabilities / equity,
        "Current Ratio": current_assets / current_liabilities,
        "Market Cap": market_cap,
        "Market Cap Category": cap_category,
        "Industry": info.get("industry")
    }

def perform_technical_analysis(symbol: str) -> dict:
    """
    Performs technical analysis using TradingView's TA API wrapper.

    Args:
        symbol (str): Ticker symbol.

    Returns:
        dict: Dictionary of technical indicators.
    """
    analysis = TA_Handler(
        symbol=symbol,
        screener='america',
        exchange='NASDAQ',
        interval=Interval.INTERVAL_1_MINUTE
    ).get_analysis().indicators
    return analysis

def fetch_news(company: str, industry: str) -> list:
    """
    Fetches recent financial news about the company and its industry.

    Args:
        company (str): Company name.
        industry (str): Industry name.

    Returns:
        list: List of news articles.
    """
    query = f'"{company}" OR "{industry}" AND (finance OR earnings OR acquisition OR merger OR launch)'
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&apiKey={NEWS_API_KEY}"
    response = requests.get(url).json()
    return response.get("articles", [])


def sentiment_calculation(articles: str) -> list:
    """
    Performs sentiment analysis on financial news using FinBERT.

    Args:
        text (str): News article text.

    Returns:
        list: Sentiment prediction output.
    """
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for article in articles:
        content = article.get("content", "")
        if content:
            try:
                result = sent_pipeline(content)[0]
                sentiment_counts[result["label"]] += 1
            except:
                continue
    return sentiment_counts


def find_ticker_symbol(company_name: str) -> str:
    """
    Finds the ticker symbol of a company using Alpha Vantage.

    Args:
        company_name (str): Name of the company.

    Returns:
        str: Ticker symbol.
    """
    url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={company_name}&apikey={ALPHA_VANTAGE_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        return data["bestMatches"][0]["1. symbol"] if data.get("bestMatches") else None
    except:
        return None


def initiate_conversation(company_name: str, start_date: str, end_date: str):
    """
    Main conversation loop to analyze a stock and chat using an LLM.

    Args:
        company_name (str): Name of the company.
        start_date (str): Historical data start date.
        end_date (str): Historical data end date.
    """
    symbol = find_ticker_symbol(company_name)
    if not symbol:
        print("Ticker symbol not found.")
        return

    print(f"\n[INFO] Ticker: {symbol}")

    fa = perform_fundamental_analysis(symbol, start_date, end_date)
    ta = perform_technical_analysis(symbol)
    articles = fetch_news(company_name, fa['Industry'])
    sa = sentiment_calculation(articles)

    # Build context for LLM
    context = (
        "### Fundamental Analysis:\n"
        + "\n".join(f"- {k}: {v}" for k, v in fa.items())
        + "\n\n### Technical Indicators:\n"
        + "\n".join(f"- {k}: {v}" for k, v in ta.items())
        + "\n\n### Sentiment Summary:\n"
        + "\n".join(f"- {k}: {v}" for k, v in sa.items())
    )

    print("\n[INFO] Analysis complete. Ask your questions below.\n(Type 'exit' to quit)\n")

    context = context[:2000]
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Session ended.")
            break
        prompt = f"""
        You are a professional stock market advisor with deep knowledge in technical, fundamental, and sentiment analysis.
        
        Your job is to provide clear, practical, and explainable advice to users based on provided financial data.

        ### Examples:
        User: What does a PE ratio of 35 mean?
        Assistant: A PE ratio of 35 means the stock is trading at 35 times its earnings. This typically indicates high growth expectations, but it could also suggest the stock is overvalued.

        User: Is a current ratio below 1 bad?
        Assistant: Yes, a current ratio below 1 may indicate the company may struggle to cover short-term liabilities with its short-term assets.

        ### Context:
        {context}

        ### User Question:
        {user_input}

        ### Assistant Response:
        """


        response = get_llm_response(prompt)
        print("Mistral:", response)


if __name__ == "__main__":
    """
    Entry point to run the stock recommender tool.
    """
    initiate_conversation("Apple Inc", "2025-06-01", "2025-07-04")
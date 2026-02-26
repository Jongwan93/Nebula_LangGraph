# Nebula LangGraph — Stock Analysis Pipeline

A LangGraph-based pipeline that predicts **1-week stock price changes** using historical prices, news, and macro data, then ranks the top 5 tickers and optionally writes results to Google Sheets.

## Features

- **Data gathering:** yfinance (30+ days price data) + Tavily (news and economic macro)
- **LLM analysis:** Deepseek (deepseek-reasoner) predicts % price change and a short reason per ticker
- **Ranking:** Keeps only positive predictions, sorts by predicted gain, takes top 5
- **Output:** Console summary + optional Google Sheets export (date, ticker, % change, reason)
- **Flow:** One ticker at a time with explicit state and conditional routing (good for debugging and scaling)

## Tech Stack

| Layer          | Technology                                      |
|----------------|--------------------------------------------------|
| Orchestration  | **LangGraph** (state graph, nodes, conditional edges) |
| LLM & prompts  | **LangChain** + **langchain-openai** (Deepseek)  |
| Price data    | **yfinance**                                    |
| News / macro   | **Tavily** (langchain-tavily)                    |
| Output        | **gspread** (Google Sheets, service account)     |

## Project Structure

```
Nebula_Langraph/
├── main.py          # Entry point; sets tickers, runs pipeline, prints results
├── graph.py         # LangGraph definition: nodes, edges, conditional routing
├── state.py         # AgentState (tickers, current_ticker, gathered_data, analysis_results, ranked_results, sheet_id)
├── nodes.py         # Node logic: select_ticker, gather_data, analyst, ranking, sheets_writer
├── tools.py         # Helpers: yfinance fetch, Tavily search, gspread append
├── requirements.txt
├── .env.example     # Template for API keys and optional GOOGLE_SHEET_ID
└── service_account.json  # Google Cloud service account key (add locally for Sheets; not in repo)
```

## Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Environment variables**

   Copy `.env.example` to `.env` and set:

   - `DEEPSEEK_API_KEY` — Deepseek API key (get from https://platform.deepseek.com/)
   - `TAVILY_API_KEY` — Tavily API key for search
   - `GOOGLE_SHEET_ID` — (optional) Google Sheet ID for writing results

3. **Google Sheets (optional)**

   To write the top 5 results to a sheet:

   - Create a service account in Google Cloud with Sheets API enabled.
   - Download the JSON key and save it as `service_account.json` in the project root.
   - Share the target Google Sheet with the service account email (Editor).
   - Set `GOOGLE_SHEET_ID` in `.env` to the sheet ID from the URL.

## Usage

Run the pipeline (tickers are configured in `main.py`):

```bash
python main.py
```

Example output:

```
Running pipeline for: 5 tickers ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL']
Output sheet: 1ZPMdTK7ZS...
  Gathering data for NVDA...
  Analyzing NVDA...
    -> NVDA: 2.5% | Strong demand for AI chips...
  ...
  Ranked 3 positive predictions (top 5 written to sheet).
--- Top performers (written to Sheet) ---
2025-02-23 | NVDA | % change: 2.5 | ...
Done.
```

## How It Works

1. **select_ticker** — Pops the next ticker from the list; when empty, routes to **ranking** instead of **gather_data**.
2. **gather_data** — Fetches price (yfinance) and news/macro (Tavily) for the current ticker only.
3. **analyst** — Builds a prompt with price summary and news, calls Deepseek, parses JSON (`predicted_change_pct`, `reason`), appends one result to state.
4. **ranking** — Filters to positive `predicted_change_pct`, sorts descending, takes top 5.
5. **sheets_writer** — Appends the ranked list to the Google Sheet (if `GOOGLE_SHEET_ID` is set).

The graph runs **select_ticker → gather_data → analyst → select_ticker** in a loop until all tickers are done, then **ranking → sheets_writer → END**.


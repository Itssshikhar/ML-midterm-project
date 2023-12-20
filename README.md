# ML-midterm-project

### Machine Learning Model for S&P 500 Companies with Financial Information

This repository contains a machine learning model developed using the S&P 500 Companies with Financial Information dataset from Kaggle. The model is designed to predict stock prices, analyze financial trends(still being worked on). Explore the code, dataset, and results in this repository.

Github Repo: https://github.com/Itssshikhar/ML-midterm-project/__

Dataset Link: https://www.kaggle.com/datasets/franoisgeorgesjulien/s-and-p-500-companies-with-financial-information__

Linkedin: https://www.linkedin.com/in/shikhar-mishra-b2079a218/__

### Dataset

This dataset provides comprehensive financial information for companies listed in the S&P 500 index. The dataset encompasses a range of fundamental financial metrics and attributes, making it a valuable resource for financial analysis, investment research, and market insights.

#### Features:

- Symbol: The unique stock symbol or ticker identifier for each S&P 500 company.

- Name: The official name or full corporate title of each company.

- Sector: The sector to which the company belongs, categorizing it into specific industry groups within the S&P 500.

- Price: The current trading price of the company's stock.

- Price/Earnings: The price-to-earnings (P/E) ratio, a key valuation metric, indicating the relationship between the stock's price and its earnings per share.

- Dividend Yield: The dividend yield, representing the ratio of the annual dividend payment to the stock's current price.

- Earnings/Share: The earnings per share (EPS), a measure of a company's profitability, calculated as earnings divided by the number of outstanding shares.

- 52 Week Low: The lowest price at which the stock has traded over the past 52 weeks.

- 52 Week High: The highest price at which the stock has traded over the past 52 weeks.

- Market Cap: The total market capitalization of the company, representing the product of the stock's current price and the total number of outstanding shares.

- EBITDA: Earnings before interest, taxes, depreciation, and amortization, a measure of a company's operating performance.

- Price/Sales: The price-to-sales ratio, which compares the stock's price to the company's revenue per share.

- Price/Book: The price-to-book (P/B) ratio, comparing the stock's price to its book value per share, an indicator of the stock's relative value.

- SEC Filings: Information regarding the company's filings with the U.S. Securities and Exchange Commission (SEC), providing transparency and compliance data.

## Midterm Project Requirements (Evaluation Criteria)

- Problem description
- EDA
- Model training
- Exporting notebook to script
- Model deployment
- Reproducibility
- Dependency and environment management
- Containerization
- Cloud deployment

### Dependency and Environment Management Guide

You can easily install dependencies from requirements.txt and use virtual environment.

- `pip install pipenv`

- `pip shell`

- `pip install -r requirements.txt`

If can't or don't know how to, here are the needed packages, just run

- `pip install pipenv Flask==3.0.0
graphviz==0.20.1
matplotlib==3.8.0
numpy==1.26.1
pandas==2.1.2
Requests==2.31.0
scikit_learn==1.3.1
seaborn==0.13.0`

### Depolyment Guide

#### To run it locally:

- Run `python model_predict.py` on a terminal
- Open a terminal and run python `test.py`

#### To run it docker:

- Download and run Docker Desktop: https://www.docker.com/

- Open a terminal

- `docker build -t midterm_project .`

- `docker run -it --rm -p 6969:6969 midterm_project`

- Open a new terminal and run python `test.py`

#### To run it in cloud:

- This is still being worked on.


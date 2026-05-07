# Analyzing Fund Flow Patterns and Market Efficiency in Pakistan with Predictive Models

Source PDF: `FYP Report (Analyzing Mutual Funds).pdf`
Pages: 15

This Markdown file is a text extraction of the source PDF with repeated headers and footers lightly cleaned.

## Page 1

Department of Computational Finance
Analyzing Fund Flow Patterns and Market Efficiency in Pakistan
with Predictive Models

Submitted By: Submitted to:
Roohan Khan CF-22029 Dr Fahim Raees
Aina Batool CF-22005
Maida Murataza CF-22016
Laiba Sarfaraz CF-22019

## Page 2

TABLE OF CONTENTS
1. Author’s Declaration -------------------------------------------------------------------------------
2. Statement of Contribution -------------------------------------------------------------------------------
3. United Nations Sustainable Development goals ------------------------------------------------------
4. Executive Summary -------------------------------------------------------------------------------
5. Acknowledgement ----------------------------------------------------------------------------
6. Similarity Index Report -------------------------------------------------------------------------------
7. Abbreviations -------------------------------------------------------------------------------
8. Chapter: 01 (Introduction)
1.1 Overview of Fund Flow ---------------------------------------------------------------------
1.2 Problem Identification ------------------------------------------------------------------------
1.3 Scope ----------------------------------------------------------------
1.4 Objective of The Study -----------------------------------------------------------------
9. Chapter: 02 (Literature Review) ----------------------------------------------------------------
10. Chapter: 03 (Methodology) ----------------------------------------------------------------------
11. Chapter: 04 (Conclusion) ------------------------------------------------------------------------
12. Chapter: 05 (Recommendations) ----------------------------------------------------------------
13. References ------------------------------------------------------------------------------------------

## Page 3

UNITED NATIONS SUSTAINABLE DEVELOPMENT GOALS

## Page 4

EXECUTIVE SUMMARY
This project, “Analyzing Fund Flow Patterns and Market Efficiency in Pakistan with Predictive
Models,” aims to explore the relationship between mutual fund flows and market efficiency in the
Pakistani financial market. Despite the growing importance of mutual funds and index investing, there
is limited research on how investor sentiment and fund flows impact market dynamics in Pakistan.
This project addresses this gap by leveraging data science and machine learning techniques to predict
fund flows and develop data-driven investment strategies.
The study begins with the collection of historical index and mutual fund data from the Pakistan Stock
Exchange (PSX). The data will be cleaned, preprocessed, and visualized to understand trends, patterns,
and correlations. Statistical analysis, including correlation and regression, will be conducted to identify
relationships between fund flows and market efficiency indicators. Machine learning models will then
be implemented to forecast fund flows, enabling the team to determine optimal weight allocation for
funds in an index.
The project is expected to provide insights into investor behavior and contribute to more efficient
portfolio management practices within P akistan’s financial market. In alignment with the United
Nations Sustainable Development Goals, the project promotes Decent Work and Economic Growth
and Industry, Innovation, and Infrastructure by applying analytical and predictive techniques to
support informed financial decision-making.
The project spans from August 2025 to June 2026 and is undertaken by a team of four students under
the supervision of Ms. Ubaida Fatima, Assistant Professor in the Department of Mathematics. The
outcomes will not only fill a research gap but also offer practical applications for investors and
policymakers in the Pakistani financial market.

## Page 5

CHAPTER: 01
INTRODUCTION
1.1 OVERVIEW OF MUTUAL FUNDS
Mutual funds are collective investment vehicles that pool money from multiple investors to invest
in a diversified portfolio of securities such as stocks, bonds, and other financial instruments. These
funds are managed by professional fund managers who make investment decisions on behalf of
the investors, aiming to achieve specific financial goals. Mutual funds provide investors with
access to a diversified portfolio, which helps in spreading risk and reducing the impact of
individual security performance on overall returns.

A mutual fund works on the principle of ownership through units or shares, where each investor
owns units proportional to their investment in the fund. The Net Asset Value (NAV) of a mutual
fund represents the per-unit value of its assets, which fluctuates based on the performance of the
underlying securities. Investors can earn returns through capital appreciation, dividends, or interest
income generated by the fund’s portfolio.

Mutual funds are regulated investment products, and in Pakistan, the Securities and Exchange
Commission of Pakistan (SECP) oversees their operations. Fund management companies, such as
conventional or Islamic asset management firms, offer a variety of mutual funds tailored to
different risk appetites and investment objectives. Common types of mutual funds include equity
funds, de bt funds, balanced funds, and money market funds. Equity funds primarily invest in
stocks, debt funds invest in bonds and fixed -income securities, balanced funds combine equities
and fixed-income assets, and money market funds focus on short-term, low-risk investments.

Investors in mutual funds are broadly categorized into retail investors —individuals investing for
personal financial goals—and institutional investors, such as banks, insurance companies, pension
funds, and other large financial organization s. Mutual funds can also be open -ended, allowing
investors to enter or exit the fund at any time, or closed-ended, where units are fixed for a specific
period and traded on stock exchanges.

The performance of a mutual fund depends on factors such as the skill of the fund manager, the
market conditions, economic indicators, and investor sentiment. Diversification, risk management,
and proper asset allocation are key strategies employed by fund managers to achieve consistent
returns.

Overall, mutual funds play a critical role in the financial market by providing an accessible
investment option for individuals and institutions, promoting capital formation, and supporting
efficient allocation of financial resources across the economy.

## Page 6

1.2 PROBLEM IDENTIFICATION
The Pakistani financial market has witnessed significant growth over the past decades, with increasing
participation in equity markets and mutual funds. However, despite this growth, research on index and
mutual funds remains limited, particularly in understanding the dynamics of investor sentiment, fund
flows, and their impact on market efficiency. Most existing studies focus on developed economies,
where market structures, investor behavior, and regulatory frameworks differ substantially from those
in Pakistan.
This research gap presents challenges for investors and policymakers. Without a clear understanding
of how fund flows interact with market efficiency in the local context, it becomes difficult to make
informed investment decisions or develop policies that support market stability and growth. Investor
sentiment, which can drive short-term market movements, and fund flows, which reflect the collective
behavior of investors, are critical indicators that remain underexplored in Pakistan.
Addressing this gap is crucial for multiple reasons. First, it can provide investors with insights into
optimal fund allocation strategies and potential market trends. Second, it can assist policymakers in
identifying patterns that may influence market e fficiency, liquidity, and stability. Finally, the study
contributes to the broader field of finance and data science by applying predictive models and machine
learning techniques to a market that has been largely overlooked in empirical research.
By analyzing fund flow patterns and their relationship with market efficiency, this project aims to
bridge the knowledge gap, offering practical and data-driven insights for both investors and regulatory
authorities in Pakistan.
1.3 SCOPE

This project focuses on the Pakistani financial market, which is considered an under -developed
economy with relatively primitive financial infrastructure compared to more developed countries.
Despite this, a variety of complex financial products exist, offering diverse techniques to address
investment and risk management challenges. In Pakistan, the Securities and Exchange
Commission of Pakistan (SECP) and the State Bank of Pakistan (SBP) act as regulatory authorities
overseeing financial institutions and market practices.

Currently, financial institutions use established methods, such as historical analysis, to monitor
and evaluate fund flows, portfolio performance, and risk measures. However, relying on a single
approach may limit the accuracy and reliability of these assessments. T his project aims to apply
multiple analytical and predictive techniques —including statistical analysis, correlation studies,
and machine learning models —to examine fund flow patterns and their impact on market
efficiency. By exploring different modeling ap proaches, we can better understand trends, make
more accurate predictions, and provide actionable insights for investment decision-making.

1.4 OBJECTIVE OF THIS STUDY
Machine learning techniques are widely used in the finance field for analyzing and predi cting fund
flows, which are closely related to investment strategies and market efficiency.

## Page 7

This study helps to utilize machine learning models to predict fund flows in the context of the Pakistani
financial market.
By examining historical fund data, the r esearch aims to forecast future investment strategies and
determine the optimal weight allocation for each fund in an index.
Also, this study will provide insights into the relationship between fund flows and market efficiency,
which can help investors, fu nd managers, and policymakers make more informed decisions for
portfolio management and market performance.
CHAPTER: 02
LITERATURE REVIEW
Yamani, Ehab (2023) investigates the informational role of fund flows in predicting mutual fund
performance for profitable investment strategies. The study uses a sample of 2,217 US equity mutual
funds from 2000 to 2018, analyzing past fund flows and returns. The research builds predictive models
to forecast whether a fund’s next -period return will be positive or negative, using past fund flow
direction as the main input. Lasso Regression (L1-Regularized Regression) and Random Forests (RF)
are employed for prediction. The stu dy finds that fund flows can successfully predict future returns,
and implementing strategies based on these predictions leads to outperformance. The study is limited
to US data and includes a restricted set of predictors, suggesting the need for testing i n other markets
or with additional features.
Larsson, Erik & Wergeland, Jacob (2020) explore the relationship between index fund flows and
market efficiency in the S&P 500, analyzing whether changes in market efficiency cause changes in
index fund flows. T he study uses data from 633 index funds and S&P returns from 2000 to 2019.
Methods include calculating the Hurst exponent via RS and DFA, fund flow calculations, and Granger
causality tests. Results indicate that lower market efficiency reduces index fund flows, suggesting that
market efficiency drives changes in index fund investment. Limitations include the focus on US data
and reliance primarily on the Hurst exponent, highlighting the need for broader studies in other markets
using multiple variables or models.
Barber, B. M., Huang, X., & Odean, T. (2016) examine which performance factors mutual fund
investors consider when deciding to invest, and how this differs between sophisticated and less
sophisticated investors. The study uses U.S. actively managed equity mutual funds from the CRSP
database, including returns, flows, and exposures to market beta, size, value, momentum, and industry
factors. Econometric regressions link fund flows to decomposed returns into alpha and factor returns,
with proxies for investor sophistication. The findings reveal that investors focus mainly on market beta
returns, often confusing factor returns with skill. Sophisticated investors react more accurately, while
unsophisticated investors misinterpret factor returns. The study is limited to U.S. equity mutual funds
and measures investor sophistication indirectly, without assessing long-term effects on fund success.
Jadoon, A. K., Mahmood, T., Sarwar, A., Javaid, M. F., & Iqbal, M. (2024) aim to predict the
movement of the KSE 100 Index using machine learning, considering economic, social, and political
factors. The study uses monthly KSE 100 data and applies an Artificial Neural Network (ANN) with
a Backpropagation algorithm. A Long Short-Term Memory (LSTM) network is employed to forecast

## Page 8

index movements. The model demonstrates high prediction accuracy, confirming that deep learning
algorithms like LSTM are effective for forecasting complex, volatile financial markets. Future research
is suggested to incorporate additional financial features and advanced deep learning models.
Yaqoob, A., & Abdullah, S. M. (2025) develop an LSTM-based deep learning model to predict closing
prices of ten major stocks in the Pakistani market across different sectors using PSX historical OHLCV
data. The model is trained on multi-year data and evaluated using R-squared ($R^2$). Results indicate
strong predictive performance ($R^2 > 0.87$) for stable, high -activity sectors, while performance is
weaker for high-risk, low-liquidity stocks. The study confirms the applicability of LSTM networks in
emerging markets and highlights the need for further improvements to handle volatile stocks and
external shocks.
Hassan, H., Niaz, A., Qureshi, J. A., & Rooh, S. (2025) investigate the effect of South Asian stock
market fluctuations on the KSE-100 Index. Monthly KSE-100 pricing data is analyzed using Multiple
Regression and ARIMA models to examine inter-market linkages. The study finds that past volatility
and lagged conditional variance strongly predict future vola tility. Hybrid models, including SVM -
GARCH and Neural Networks, outperform simpler GARCH models. The findings help investors
diversify risk and assist policymakers in understanding regional market impacts.
Hassan, H., Niaz, A., Qureshi, J. A., & Rooh, S. (2025) also study the impact of macroeconomic factors
on KSE -100 performance, using monthly data from July 2014 to June 2024. Multiple Regression
Analysis and ARIMA models are employed to assess the influence of exchange rates, FDI, and Balance
of Trade (BO T). Results indicate significant effects of exchange rate fluctuations, FDI, and trade
surplus on stock market performance. The study provides actionable insights for policymakers,
investors, and analysts in managing economic risks.
Idrees, M., Sial, M. H., & Hassan, N. U. (2025) compare four deep learning models for predicting
KSE-100 daily closing prices, focusing on LSTM with Attention. Using daily data from 2008 to 2021
(3221 rows), models include ANN, RNN with Attention, LSTM with Attention, and GRU with
Attention. The LSTM-Attention model achieves the highest accuracy (R²: Training 0.9996, Validation
0.9980, Testing 0.9921). The study highlights the effectiveness of the attention mechanism for
sequential financial data and proposes a robust architecture for stock price prediction.
CHAPTER: 03
METHADOLOGY
3.1 Conceptual Framework of Fund Flow Analysis
Fund flows refer to the net movement of money into or out of mutual funds or investment vehicles
over a period of time. Analyzing fund flows is critical for understanding investor behavior and
assessing market efficiency. Fund flows provide information about investor sentiment, revealing which
assets or funds are attracting or losing investment. Positive net fund flows often indicate bullish
sentiment, whereas negative flows reflect caution or pessimism.
The analysis of fund flows over time allows us to observe essential patterns, including trends, cyclic
behavior, and short -term fluctuations. Trends indicate sustained increases or decreases in fund
allocations over time, reflecting overall investor confidence or market perception. Cyclic patterns
emerge due to periodic economic or regulatory factors affecting investment decisions, while short -

## Page 9

term fluctuations capture temporary shifts in investor sen timent. Understanding these dynamics is
essential for forecasting future fund movements and developing predictive investment strategies.
3.2 Data Collection and Preprocessing
The study collects historical data for mutual funds and indices listed on the Pak istan Stock Exchange
(PSX), focusing on the KSE-30 and KSE-100 indices. The dataset spans the period from January 2020
to October 2025. Preprocessing was performed to ensure data accuracy and consistency. Columns with
missing or zero values were filled using the forward-filling method, while rows where index weightage
was zero were removed. Company symbols were standardized to create a unique identifier for each
company, and non -essential columns were reduced to focus on relevant features. The data was
normalized to make it suitable for machine learning algorithms, and visualization techniques were
applied to detect trends, patterns, and relationships within the dataset.
Table 1: 3.2 Data Collection

Data of KSE-30 Index from PSX.

3.3 FEATURE ENGINEERING
3.3.1 Daily Returns
Daily log returns for each company were calculated to capture the percentage change in price over
consecutive trading days. This helps in understanding short-term price movements and forming the
basis for volatility calculations.

3.3.2 Moving Averages
Short-term and long-term trends were identified by calculating moving averages of price and
volume. The 7-day and 30-day rolling averages for price (MA7_Price, MA30_Price) and 7-day
rolling average for volume (MA7_Volume) were computed to smooth out short-term fluctuations.

## Page 10

3.3.3 Volatility
Volatility was estimated as the 30-day rolling standard deviation of daily returns. This feature
captures the magnitude of price fluctuations and is essential for assessing risk in fund flow
prediction.

3.3.4 Momentum Features
Momentum indicators were generated using percentage changes in price over 5-day and 10-day
periods (Momentum_5d, Momentum_10d) to capture short-term price trends and market sentiment.

3.3.5 Relative Strength Index (RSI)
RSI was calculated over a 14-day window to quantify overbought or oversold conditions in stock
prices, helping in identifying potential turning points in price movements.

3.3.6 Bollinger Bands
Upper and lower Bollinger Bands were computed using a 20-day rolling standard deviation around
the 30-day moving average of price. The band width (BB_Width) measures price dispersion and
volatility in the market.

## Page 11

3.3.7 Project-Specific Features
 delta_wt, delta_price, delta_volume: daily changes in index weightage, price, and volume
were calculated to track short-term dynamics.

 prev_ff_based_mcap: lagged fund-flow-based market capitalization was used to capture
historical fund trends.

 flow_proxy: daily change in fund -flow-based market capitalization normalize d by the
previous value, serving as a stable measure of fund movements.

 volume_price_ratio: ratio of daily volume to price, representing interaction between
trading activity and price levels.

## Page 12

_No extractable text on this page._

## Page 13

_No extractable text on this page._

## Page 14

CHAPTER: 05
RECOMMENDATIONS

## Page 15

REFERENCES

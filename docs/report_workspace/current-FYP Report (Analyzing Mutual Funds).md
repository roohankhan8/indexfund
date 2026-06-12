# FYP Report (Analyzing Funds)

Source DOCX: `0-docs\reports\FYP Report (Analyzing Funds) (3).docx`

This Markdown file is generated from the latest DOCX content.

NED UNIVERSITY OF ENGINEERING & TECHNOLOGY

Department of Mathematics (Computational Finance)Analyzing Fund Flow Patterns and Market Efficiency in Pakistan with Predictive Models Group number: 02Batch: 2022-2026

Submitted By:   Submitted to:Roohan Khan CF-22029                                                           Dr Fahim Raees

Aina BatoolCF-22005

Maida Murtaza CF-22016   Project Advisor:

Laiba Sarfaraz CF-22019   Ms. Ubaida Fatima

(Assistant Professor,   NEDUET)

## AUTHOR’S DECLARATION

We declare that we are the sole authors of this project. It is the actual copy of the project that was accepted by our advisor(s) including any necessary revisions. We also grant NED University of Engineering and Technology permission to reproduce and distribute electronic or paper copies of this project.

Signature and Date             Signature and DateSignature and Date  Signature and Date

___________________                 ___________________             ___________________             __________________

12 June 2025                         12 June 2025                       12 June 2025                  12 June 2025   Roohan Khan                          Aina Batool                     Maida Murtaza                Laiba Sarfaraz     CF-22029                               CF-22005                           CF-22016                        CF-22019

## Statement of Contributions

## Executive Summary

This project, “Analyzing Fund Flow Patterns and Market Efficiency in Pakistan with Predictive Models,” aims to explore the relationship between index fund flows and market efficiency in the Pakistani financial market. Despite the growing importance of funds and index investing, there is limited research on how investor sentiment and fund flow impact market dynamics in Pakistan. This project addresses this gap by leveraging data science and machine learning techniques to predict index fund flows and develop data-driven investment strategies.

The study begins with the collection of historical KSE-30 constituent data  from the Pakistan Stock Exchange (PSX). The data will be cleaned, preprocessed, and visualized to understand trends, patterns, and correlations. Statistical analysis, including correlation and regression, will be conducted to recognize relationships between fund flows and market efficiency indicators. Machine learning models will then be implemented to forecast index fund flows, enabling the team to determine optimal weight allocation for stocks within the KSE-30 index.

The project is expected to come up with insights into investor behavior and contribute to more efficient portfolio management practices within Pakistan’s financial market. In alignment with the United Nations Sustainable Development Goals, the project promotes Decent Work and Economic Growth and Industry, Innovation, and Infrastructure by applying analytical and predictive techniques to support informed financial decision-making.

The project spans from August 2025 to June 2026 and is undertaken by a team of four students under the supervision of Ms. Ubaida Fatima, Assistant Professor in the Department of Mathematics. The outcomes will not only fill a research gap but also offer practical applications for investors and policymakers in the Pakistani financial market.

## Acknowledgments

We would like to express our most sincere gratitude to our Project Advisor, Ms. Ubaida Fatima (Lecturer, NED University of Engineering & Technology), for her invaluable guidance, insightful suggestions, and constant encouragement throughout the course of our project, "Analyzing Fund Flow Patterns and Market Efficiency in Pakistan with Predictive Models." Her unwavering support was instrumental in shaping the direction and successful completion of our research.

We would also like to extend our heartfelt thanks to the Department of Mathematics (Computational Finance) at NED University of Engineering & Technology for providing us with the resources, knowledge, and academic environment necessary to undertake this research. The department's commitment to excellence has been a constant source of inspiration throughout our academic journey.

We would also like to thank the FYDP Committee for giving us the opportunity to complete this project and for their guidance at every stage. Their valuable advice has been a tremendous help throughout our research journey.

Finally, we extend our deepest appreciation to our families and peers for their unwavering moral support and encouragement, which kept us motivated during every challenging phase of this project.

## Table of Contents

## List of Figures

Figure 4.1: Reconstructed cumulative KSE-30 return over the final analysis period.

Figure 4.2: AUM trend and aggregate KSE-30 sector total.

Figure 4.3: Distribution of daily NAV returns for the KSE-30 related funds.

Figure 4.4: Aggregate monthly KSE-30 sector fund flows showing alternating inflow and outflow episodes.

Figure 4.5: Monthly correlation matrix for KSE-30 flow, macroeconomic variables, and index measures.

Figure 4.6: Lag-1 Granger causality p-values for macroeconomic variables against aggregate KSE-30 fund flow.

Figure 4.7: Actual versus predicted aggregate KSE-30 sector flow under ARIMAX and VAR.

Figure 4.8: KSE-30 daily returns and model-implied conditional volatility.

Figure 4.9: 5 percent VaR backtesting for the preferred KSE-30 volatility model.

Figure 4.10: Variance-ratio evidence for the KSE-30 across alternative lag horizons.

Figure 4.11: Autocorrelation structure of reconstructed KSE-30 returns.

Figure 5.1: Integrated rebalancing and portfolio tilt framework.

Figure 5.2: Estimated retention probabilities for current KSE-30 constituents.

Figure 5.3: Feature importance ranking for the rebalancing prediction model.

Figure 5.4: Scatter plot of actual versus predicted constituent weights.

Figure 5.5: Forecasted direction and magnitude of KSE-30 weight changes.

Figure 6.1: Fund-flow model performance scorecard.

Figure 6.2: Summary of market efficiency evidence for the KSE-30.

Figure 6.3: KSE-30 realized volatility regimes over the study period.

Figure 6.4: Performance comparison across model families (Econometric vs. ML).

Figure 6.5: KSE-30 rebalancing risk and opportunity map.

## List of Tables

Table 4.1: Aggregate KSE-30 Fund-Flow Forecasting Results

Table 4.2: KSE-30 GARCH and EGARCH Results

Table 4.3: KSE-30 Market Efficiency Test Results

## CHAPTER: 01

### INTRODUCTION

#### 1.1 Overview of Fund Flow

Fund flows refer to the net movement of money into or out of funds over a specific period of time and serve as an important indicator of investor behavior and market sentiment. Positive fund flows occur when new investments exceed redemptions, indicating increased investor confidence and demand for the fund. Conversely, negative fund flows arise when withdrawals surpass new investments, reflecting risk aversion, changing market expectations, or adverse economic conditions. Fund flows play a crucial role in understanding how investors respond to market movements, performance signals, and macroeconomic factors.

In the broader context of investment funds, fund flows are closely associated with changes in Net Asset Value (NAV), overall fund performance, and portfolio rebalancing decisions undertaken by fund managers. Substantial inflows can improve a fund’s liquidity and enable managers to expand or diversify their asset holdings, whereas large outflows may require asset liquidation, potentially affecting asset prices and market stability. Consequently, fund flows not only signal investor preferences but also play a role in shaping short-term price movements and overall market efficiency.

Fund flows are influenced by many factors, including historical fund performance, market volatility, interest rate movements, macroeconomic conditions, and investor sentiment. Individual investors often respond to recent returns and prevailing market trends, while institutional investors typically base their allocation decisions on long-term investment objectives and macroeconomic expectations. In emerging markets such as Pakistan, fund flows may further be shaped by regulatory developments, political conditions, and the limited availability of alternative investment opportunities.

In Pakistan, fund flows are regulated and overseen within the framework established by the Securities and Exchange Commission of Pakistan (SECP). Fund flow data offers important insights for policymakers, investors, and researchers in evaluating market behavior, liquidity dynamics, and the effectiveness of capital allocation. The analysis of fund flows assists in identifying herd behavior, understanding information transmission across financial markets, and assessing how investors respond to market signals.

Overall, fund flows act as a bridge between investor sentiment and market performance, making them a critical variable in empirical studies related to market efficiency, asset pricing, and investment decision-making in both developed and developing financial markets.

#### 1.2 Overview of Market Efficiency

Market efficiency refers to the degree to which financial markets incorporate all available information into asset prices. According to the Efficient Market Hypothesis (EMH), a market is considered efficient if security prices fully and quickly reflect relevant information, making it impossible for investors to consistently earn abnormal returns through analysis or trading strategies based on publicly available data. Market efficiency plays a vital role in ensuring fair pricing, optimal capital allocation, and overall financial stability.

Market efficiency is commonly categorized into three forms: weak, semi-strong, and strong efficiency. Weak-form efficiency suggests that current prices already reflect all past price and volume information, implying that technical analysis cannot consistently generate excess returns. Semi-strong efficiency extends this concept by incorporating all publicly available information, including financial statements, economic indicators, and news announcements, thereby limiting the effectiveness of fundamental analysis. Strong-form efficiency represents the highest level, where prices reflect all information, both public and private, making it impossible for any investor to gain an informational advantage.

Efficient markets facilitate the smooth functioning of financial systems by ensuring that prices act as accurate signals of underlying economic value. This enables investors to make informed decisions and allows firms to raise capital at fair costs. In an efficient market, resources are allocated to their most productive uses, reducing mispricing and speculative distortions.

However, market efficiency may vary across countries and market structures. In emerging markets such as Pakistan, factors such as limited market depth, lower liquidity, information asymmetry, regulatory constraints, and behavioral biases among investors may affect the speed and accuracy with which information is reflected in prices. As a result, deviations from full efficiency may exist, creating opportunities for informed trading and empirical investigation.

Analyzing market efficiency is particularly important in the context of fund flows and investor sentiment, as large inflows or outflows can influence price movements and volatility. Understanding how fund flows interact with market efficiency provides valuable insights into whether markets respond rationally to information or exhibit delayed adjustments and inefficiencies.

Overall, market efficiency serves as a foundational concept in financial economics, offering a framework to evaluate price behavior, investor decision-making, and the effectiveness of financial markets in allocating capital, especially within developing economies like Pakistan.

#### 1.3 Problem Identification

The Pakistani financial market has witnessed significant growth over the past decades, with increasing participation in equity markets and funds. However, despite this growth, research on index and funds remains limited, particularly in understanding the dynamics of investor sentiment, fund flows, and their impact on market efficiency. Most existing studies focus on developed economies, where market structures, investor behavior, and regulatory frameworks differ substantially from those in Pakistan.

This research gap presents challenges for investors and policymakers. Without a clear understanding of how fund flows interact with market efficiency in the local context, it becomes difficult to make informed investment decisions or develop policies that support market stability and growth. Investor sentiment, which can drive short-term market movements, and fund flows, which reflect the collective behavior of investors, are critical indicators that remain underexplored in Pakistan.

Addressing this gap is crucial for multiple reasons. First, it can provide investors with insights into optimal fund allocation strategies and potential market trends. Second, it can assist policymakers in identifying patterns that may influence market efficiency, liquidity, and stability. Finally, the study contributes to the broader field of finance and data science by applying predictive models and machine learning techniques to a market that has been largely overlooked in empirical research.

By analyzing fund flow patterns and their relationship with market efficiency, this project aims to bridge the knowledge gap, offering practical and data-driven insights for both investors and regulatory authorities in Pakistan.

#### 1.4 Scope

This project focuses on the Pakistani financial market, which is considered an under-developed economy with relatively primitive financial infrastructure compared to more developed countries. Despite this, a variety of complex financial products exist, offering diverse techniques to address investment and risk management challenges. In Pakistan, the Securities and Exchange Commission of Pakistan (SECP) and the State Bank of Pakistan (SBP) act as regulatory authorities overseeing financial institutions and market practices.

Currently, financial institutions use established methods, such as historical analysis, to monitor and evaluate fund flows, portfolio performance, and risk measures. However, relying on a single approach may limit the accuracy and reliability of these assessments. This project aims to apply multiple analytical and predictive techniques—including statistical analysis, correlation studies, and machine learning models—to examine fund flow patterns and their impact on market efficiency. By exploring different modeling approaches, we can better understand trends, make more accurate predictions, and provide actionable insights for investment decision-making.

1.5 Objective of This Study

Machine learning techniques are widely used in the finance field for analyzing and predicting fund flows, which are closely related to investment strategies and market efficiency.

This study helps to utilize machine learning models to predict index fund flows patterns through constituent weight shifts in the KSE-30 Index.

By examining historical KSE-30 fund data, the research aims to forecast future investment strategies and determine the optimal weight allocation for stock within the index.

Also, this study will provide insights into the relationship between fund flows and market efficiency, which can help investors, fund managers, and policymakers make more informed decisions for portfolio management and market performance

## CHAPTER: 02

### LITERATURE REVIEW

#### 2.1 Introduction

Financial market forecasting has become one of the most widely researched areas in finance due to the increasing volatility and complexity of global and emerging stock markets. Researchers have extensively explored the relationship between investor behavior, mutual fund flows, market efficiency, volatility clustering, and stock market forecasting using both traditional econometric and modern machine learning approaches. In recent years, deep learning models such as Long Short-Term Memory (LSTM), Artificial Neural Networks (ANN), Random Forest, and hybrid forecasting frameworks have demonstrated strong predictive capabilities in financial time-series analysis. Emerging markets, particularly the Pakistan Stock Exchange (PSX), have attracted significant research interest because of their volatile nature, susceptibility to political and macroeconomic shocks, and evolving market efficiency.

The literature reveals that investor behavior and index fund flows significantly influence market dynamics and investment decisions. Simultaneously, researchers have examined the efficiency of financial markets through random walk tests and volatility modeling techniques such as ARCH, GARCH, EGARCH, and ARIMA. Furthermore, machine learning and hybrid models have increasingly been applied to stock market prediction due to their ability to capture nonlinear relationships and long-term dependencies within financial data.

This chapter reviews prior studies related to index fund flows, investor behavior, market efficiency, volatility modeling, time-series forecasting, machine learning applications, and research on the Pakistan Stock Exchange and KSE-30 index. The chapter also identifies the research gap that motivates the present study.

#### 2.2 Index Fund Flows and Investor Behavior

Index fund flows and investor behavior have been widely examined to understand how investors react to market information and how capital flows within index funds can predict future market performance. Yamani, Ehab (2023) investigated the informational role of fund flows in predicting mutual fund performance using a sample of 2,217 U.S. equity mutual funds from 2000 to 2018. The study employed Lasso Regression and Random Forest models and found that fund flows significantly predict future returns, enabling profitable investment strategies. However, the study was limited to U.S. markets and a restricted set of variables.

Similarly, Jürg Fausch, Moreno Frigg, Stefan Ruenzi, and Florian Weigert (2025) examined the prediction of mutual fund flows using machine learning methods on more than 13,000 U.S. equity mutual fund share classes. Their findings showed that nonlinear machine learning models, especially Random Forests, outperform traditional regression models in predicting future fund flows and identifying high-performing funds.

Brad M. Barber, Xing Huang, and Terrance Odean (2016) explored how sophisticated and unsophisticated investors interpret mutual fund performance. Using actively managed U.S. equity mutual funds, the study concluded that many investors confuse factor-based returns with managerial skill, while sophisticated investors make more informed investment decisions.

Kyoung Lim and Sun-Joong Yoon (2018) analyzed Korean public fund market data to determine whether fund flows could serve as a proxy for investor sentiment. Their results suggested that direct fund flows do not fully capture investor sentiment, although transfer flows between bond and equity funds showed improved predictive power.

Nghia Chu, Binh Dao, Nga Pham, Huy Nguyen, and Hien Tran (2023) investigated whether deep learning models could predict mutual fund performance more accurately than traditional statistical approaches. Using over 600 U.S. open-end mutual funds, the study demonstrated that LSTM and GRU models significantly outperform ARIMA and other traditional forecasting techniques.

The literature collectively highlights that index fund flows contain valuable predictive information and investor behavior strongly influences capital allocation decisions within index funds. However, most studies are concentrated in developed markets, indicating the need for similar research in emerging economies such as Pakistan, particularly for index-based instruments such as the KSE-30.

#### 2.3 Market Efficiency in Emerging Markets

Market efficiency remains a critical area of financial research, particularly in emerging economies where information asymmetry and market volatility are relatively high. Erik Larsson and Jacob Wergeland (2020) examined the relationship between index fund flows and market efficiency in the S&P 500 using Hurst exponent analysis and Granger causality tests. The study concluded that lower market efficiency reduces index fund flows, implying that efficiency levels influence investor participation.

In the Pakistani context, Ushna Akber and Nabeel Muhammad (2014) investigated whether the KSE-100 Index follows weak-form market efficiency and random walk behavior using multiple statistical tests including Runs Test, Phillips-Perron Test, Ljung-Box Test, and ARIMA models. The findings indicated that the KSE-100 Index is weak-form inefficient, although efficiency improved during the later years of the sample period.

More recently, Ishtiaq Khan and Ashfaq Ali Khattak (2026) further examined weak-form market efficiency in the Pakistan Stock Exchange using daily KSE-100 Index data from 2018 to 2024. The study employed Runs Test, Variance Ratio Test, Ljung-Box Q Test, and GARCH-family models. Results rejected the random walk hypothesis and confirmed the existence of significant autocorrelation, volatility clustering, and leverage effects within the PSX.

The evidence from emerging markets suggests that stock prices do not always fully reflect available information, thereby creating opportunities for prediction and abnormal returns. The persistent inefficiency of the PSX supports the relevance of forecasting models and machine learning techniques for investment decision-making.

#### 2.4 Volatility Modeling in Financial Markets

Volatility forecasting is essential for portfolio management, risk assessment, and investment planning. Numerous studies have applied GARCH-family models to capture volatility clustering and leverage effects in stock markets.

Muhammad Asif and Abdul Aziz (2016) examined volatility clustering in KSE-100 returns using GARCH(1,1), EGARCH, and TGARCH models. Their findings confirmed the existence of significant volatility clustering and identified EGARCH as the most suitable model due to its ability to capture leverage effects.

Muhammad Shahzeb Ali and Atiya Javed (2020) compared ARIMA, ARCH, and GARCH models using daily PSX closing prices. The study found GARCH(3,2) to be the best-performing model for capturing market volatility and forecasting stock market movements.

Zahid Iqbal and Lubna Naz (2025) identified ARMA(1,1)-GARCH(1,1) with Skewed Student’s t-distribution as the most effective model for forecasting KSE-100 volatility. Their study confirmed the presence of heteroscedasticity and volatility clustering in the Pakistani stock market.

Komal Batool, Mirza Faizan Ahmed, and Muhammad Ali Ismail (2022) compared standalone GARCH and Neural Network Autoregression (NNAR) models with hybrid combinations. The linear combination of GARCH and NNAR achieved superior forecasting accuracy, confirming the benefits of integrating statistical and machine learning approaches.

Muhammad Usman Rasheed, Taha Fareed, Bismah Ahmed, and Maheen Badar (2024) utilized GARCH models to estimate Value at Risk (VaR) for PSX equity portfolios. Their results demonstrated that GARCH-based dynamic variance models provide more realistic risk estimates compared to static approaches.

Dr. Haseb Hassan, Abubakr Niaz, Dr. Javeria Andleeb Qureshi, and Dr. Samina Rooh (2025) investigated the effect of South Asian stock market fluctuations on KSE-100 volatility using Multiple Regression and ARIMA models. Their findings revealed that past volatility and conditional variance significantly influence future volatility, while hybrid models such as SVM-GARCH outperform standalone GARCH models.

Tayyab Raza Fraz, Samreen Fatima, and Mudassir Uddin (2022) compared GARCH, Markov-Switching GARCH, and ANN models for volatility forecasting in Pakistan and China. Surprisingly, the standard GARCH model achieved the best fit according to AIC and BIC criteria.

The reviewed literature confirms that volatility clustering and leverage effects are dominant characteristics of emerging stock markets, and hybrid forecasting models tend to outperform traditional econometric methods.

#### 2.5 Time-Series Forecasting of Financial Variables

Time-series forecasting models are commonly applied to predict stock prices, volatility, and financial market trends. Traditional statistical techniques such as ARIMA and ARMA remain widely used due to their simplicity and effectiveness for linear financial patterns.

Shoaib Anwer Qambrani, Israr Ahmed, and Abdul Basit (2023) examined ARIMA models for forecasting the KSE-100 Index using daily stock data from 2012 to 2023. Their results indicated that ARIMA(1,0,1) performs better for short-term forecasting, whereas ARIMA(1,1,1) captures long-term trends more effectively.

Asma Zaffar and S. M. Aalim Hussain (2022) combined ANN and ARMA models with news sentiment analysis to forecast KSE-100 closing prices. Their hybrid ANN-ARMA framework demonstrated improved predictive performance compared to standalone ARMA models.

Muhammad Ali, Dost Muhammad Khan, Huda M. Alshanbari, and Al Aziz Hosni Bagoury (2023) proposed an improved hybrid EMD-LSTM model for stock market prediction. Using S&P 500 daily prices, the study showed that decomposing complex financial series before prediction significantly improves forecasting accuracy.

Ubaida Fatima, Rimsha Zafar, Syeda Arsala Shah, and Abdus Samad (2025) developed hybrid forecasting models combining ARIMA, GARCH, Linear Regression, and LSTM techniques for KSE-100 volatility forecasting. Their findings indicated that hybrid Linear Regression-GARCH models achieved the most stable and accurate forecasts.

Taha Munawar, Laiba Mushtaq, and Muhammad Hassan Siddiqui (2025) developed a real-time volatility forecasting platform using the GARCH(1,1) model for companies such as Apple, Microsoft, and Tesla. The platform effectively captured volatility clustering and demonstrated strong forecasting performance for relatively stable stocks.

The literature suggests that time-series forecasting techniques remain highly relevant in financial analysis, particularly when combined with hybrid or machine learning frameworks.

#### 2.6 Machine Learning Applications in Stock and Fund Prediction

Machine learning and deep learning approaches have transformed financial forecasting by effectively modeling nonlinear relationships and sequential dependencies in financial data.

Atif Khan Jadoon, Tariq Mahmood, Ambreen Sarwar, Maria Faiq Javaid, and Munawar Iqbal (2024) applied ANN and LSTM models to predict KSE-100 movements using economic, social, and political variables. The study confirmed the effectiveness of deep learning techniques in forecasting volatile financial markets.

Ahad Yaqoob and Muhammad Abdullah (2025) developed an LSTM-based model to predict stock prices of major Pakistani companies using historical OHLCV data. Their results showed strong predictive performance for stable and liquid stocks.

Muhammad Idrees, Maqbool Hussain Sial, and Najam-ul-Hasan (2025) compared ANN, RNN-Attention, LSTM-Attention, and GRU-Attention models for KSE-100 prediction. The LSTM-Attention model achieved the highest forecasting accuracy, demonstrating the importance of attention mechanisms in sequential data analysis.

Zahid Iqbal and Muhammad Shoaib (2025) utilized LSTM networks to forecast stock market volatility in Pakistan and achieved strong predictive accuracy with low mean squared error values.

Saba Zahid and Hasan Mujtaba Nawaz Saleem (2023) applied Support Vector Machine (SVM) models to predict stock price movements during the COVID-19 pandemic. Their results showed that the RBF kernel outperformed other kernel functions under volatile market conditions.

Rabia Sabri and Sobia Iqbal (2024) compared ARIMA, SARIMA, and LSTM models across Pakistan, Bangladesh, and Sri Lanka. The study found that LSTM consistently outperformed traditional statistical approaches in frontier markets.

Nusrat Rouf, Majid Bashir Malik, Tasleem Arif, Sparsh Sharma, Saurabh Singh, Satyabrata Aich, and Hee-Cheol Kim (2021) conducted a systematic review of machine learning applications in stock market prediction. The review concluded that deep learning models such as LSTM and RNN generally outperform traditional models, particularly when integrated with news sentiment and financial indicators.

Kinza Bukhari, Atif Khan Jadoon, Munawar Iqbal, and Ayesha Arshad (2023) employed ANN with sixteen macroeconomic variables to model and forecast KSE-100 prices, achieving remarkably high prediction accuracy.

Irfan Javid, Rozaida Ghazali, Irteza Syed, Muhammad Zulqarnain, and Noor Aida Husaini (2022) proposed a hybrid feature selection framework integrated with GRU and LSTM models for predicting stock market crises in Pakistan. Their findings indicated that the GRU-based approach achieved superior predictive performance.

Saima Latif, Nadeem Javaid, Faheem Aslam, Abdulaziz Aldegheishem, Nabil Alrajeh, and Safdar Hussain Bouk (2024) developed a PLSTM-TAL hybrid deep learning framework with temporal attention layers for predicting stock market direction across multiple global indices. The model achieved high prediction accuracy and effectively captured long-term temporal dependencies.

S. N. Khan, S. Shafique, Ansar, Z. Imran, P. Altamish, and Hamza (2025) compared KNN, SVM, and Naïve Bayes classifiers for stock prediction. Their results indicated that KNN outperformed the other classifiers across all datasets.

Tahir Munir, Rabia Emhamed Al Mamlook, Abdu R. Rahman, Afaf Alrashidi, and Aqsa Muhammad Yaseen (2024) compared multiple machine learning models during the COVID-19 pandemic and found Random Forest to be the most accurate forecasting model for KSE-100 developments.

Overall, the literature demonstrates that machine learning and hybrid deep learning models significantly improve forecasting performance compared to traditional econometric techniques.

#### 2.7 Literature on Pakistan Stock Exchange and KSE-30 Index

The Pakistan Stock Exchange has been extensively studied due to its volatility, emerging market characteristics, and sensitivity to political and macroeconomic events.

Several studies have focused on KSE-100 forecasting and volatility modeling. Muhammad Shahzeb Ali and Atiya Javed (2020) confirmed that GARCH models outperform ARIMA and ARCH approaches in capturing PSX volatility. Zahid Iqbal and Lubna Naz (2025) further identified ARMA-GARCH specifications as effective tools for forecasting KSE-100 volatility.

Research by Atif Khan Jadoon et al. (2024), Ahad Yaqoob and Muhammad Abdullah (2025), and Muhammad Idrees et al. (2025) demonstrated the strong predictive capabilities of LSTM-based deep learning models in forecasting KSE-100 and individual PSX stock prices.

Kinza Bukhari et al. (2023) highlighted the importance of macroeconomic variables such as exchange rates, inflation, FDI, and GDP growth in predicting KSE-100 performance using ANN models. Similarly, Dr. Haseb Hassan et al. (2025) confirmed that exchange rates, balance of trade, and FDI significantly affect stock market performance.

Studies such as Muhammad Mohsin et al. (2020) emphasized the influence of market risk, interest rates, and exchange rates on Pakistani bank stock volatility. Muhammad Usman Rasheed et al. (2024) explored Value at Risk estimation using GARCH models for PSX portfolios, while Irfan Javid et al. (2022) focused on predicting stock crises using hybrid machine learning frameworks.

The overall literature on PSX indicates that the Pakistani stock market exhibits high volatility, weak-form inefficiency, and strong sensitivity to macroeconomic and political factors. These characteristics make PSX a suitable environment for applying advanced forecasting and volatility prediction models.

#### 2.8 Research Gap Identified from Prior Studies

The reviewed literature identifies several important research gaps. First, most studies on  fund flows and investor behavior are concentrated in developed markets such as the United States and Korea, while limited evidence exists for emerging markets like Pakistan.

Second, although numerous studies have examined stock market forecasting using machine learning and deep learning models, many focus either on stock prices or volatility independently rather than integrating investor behavior, fund flows, macroeconomic variables, and market efficiency into a unified forecasting framework.

Third, existing PSX studies largely emphasize traditional econometric techniques such as ARIMA and GARCH or apply machine learning models without comparing multiple advanced hybrid approaches. Limited research combines deep learning models with volatility forecasting methods and macroeconomic indicators for comprehensive prediction of KSE indices.

Fourth, many prior studies rely solely on historical price data while excluding external factors such as investor sentiment, macroeconomic conditions, political uncertainty, and mutual fund flows, which may significantly improve forecasting performance.

Finally, there remains a lack of comparative analysis between traditional statistical approaches and advanced deep learning techniques within the context of the Pakistan Stock Exchange. Therefore, the present study aims to address these gaps by examining the forecasting performance of advanced machine learning and volatility models for KSE-related financial variables while considering broader economic and behavioral influences.

#### 2.9 Chapter Summary

This chapter reviewed the existing literature on fund flows, investor behavior, market efficiency, volatility forecasting, time-series analysis, machine learning applications, and Pakistan Stock Exchange studies. The review demonstrated that financial markets, especially emerging markets such as Pakistan, exhibit volatility clustering, weak-form inefficiency, and nonlinear behavior, making forecasting highly challenging.

The literature further established that machine learning and deep learning models, particularly LSTM, GRU, Random Forest, and hybrid forecasting frameworks, generally outperform traditional statistical techniques such as ARIMA and GARCH in predicting stock prices, volatility, and mutual fund performance. At the same time, traditional econometric models remain important for capturing volatility dynamics and market risk.

The review also identified several research gaps, including the limited application of integrated forecasting frameworks in the Pakistani context and the need to combine behavioral, macroeconomic, and financial variables within advanced predictive models. These gaps provide the foundation and motivation for the present study.

## CHAPTER: 03

### METHADOLOGY

#### 3.1 Introduction

This chapter explains the research methodology adopted for examining financial market forecasting, volatility behavior, and machine learning applications in the context of the Pakistan Stock Exchange (PSX). The methodology provides a systematic framework for collecting, processing, and analyzing financial market data to achieve the objectives of the study.

The chapter discusses the research philosophy, research design, population and sample selection, data sources, variables, and analytical techniques employed in the study. In addition, econometric and machine learning models including ARIMA, GARCH, Artificial Neural Networks (ANN), Long Short-Term Memory (LSTM), Random Forest, and hybrid forecasting models are discussed in detail. The chapter also explains model evaluation methods and ethical considerations.

#### 3.2 Research Philosophy and Approach

The present study follows a positivist research philosophy because it relies on objective financial data, statistical testing, and empirical analysis. Positivism is appropriate for financial forecasting research as it allows researchers to identify relationships between variables using quantitative methods.

The study adopts a quantitative research approach because numerical stock market and macroeconomic data are analyzed through statistical and machine learning techniques. Quantitative methods are suitable for forecasting studies since they facilitate hypothesis testing, pattern recognition, and prediction of financial variables.

A deductive research approach is employed in this study. Existing theories related to market efficiency, volatility clustering, and financial forecasting are used as the basis for developing hypotheses and testing predictive models.

#### 3.3 Research Design

This study uses an explanatory and predictive research design. The explanatory component investigates the relationship between macroeconomic variables, investor behavior, market volatility, and stock market performance, while the predictive component evaluates the forecasting ability of traditional econometric and machine learning models.

The research is based on time-series analysis because stock market and financial variables are sequential in nature and change over time. Daily and monthly observations are analyzed to capture both short-term and long-term market behavior.

The study compares traditional statistical approaches such as ARIMA and GARCH with modern machine learning and deep learning techniques including ANN, LSTM, Random Forest, and hybrid forecasting frameworks.

#### 3.4 Population and Sample Selection

The population of the study consists of companies and financial indices listed on the Pakistan Stock Exchange (PSX). The primary focus is placed on the KSE-30 Index because it represents the largest and most actively traded companies in Pakistan.

The sample includes daily and monthly observations of the KSE-30 Index and selected macroeconomic variables over a specified period. Depending on data availability, the study may also include sector-specific stocks or mutual fund-related variables.

The study adopts purposive sampling because the KSE-30 Index is considered the most representative indicator of overall stock market performance in Pakistan.

The sample period spans multiple years to ensure that different economic cycles, political events, and market shocks are captured within the analysis.

#### 3.5 Data Sources and Data Collection

The study utilizes secondary data collected from reliable financial and economic databases. Daily and monthly stock market data are obtained from the Pakistan Stock Exchange (PSX), Yahoo Finance, Investing.com, and official financial reports.

Macroeconomic variables such as inflation, exchange rates, foreign direct investment (FDI), interest rates, crude oil prices, and balance of trade are collected from the State Bank of Pakistan (SBP), Pakistan Bureau of Statistics (PBS), and World Bank databases.

The data collection process includes the following steps:

Collection of historical KSE-30 constituent stock prices, volumes, and index weights from PSX.

Collection of macroeconomic indicators.

Cleaning and preprocessing of missing or inconsistent values.

Transformation of data into suitable formats for statistical and machine learning analysis.

Division of the dataset into training, validation, and testing subsets.

The use of secondary data ensures reliability, cost efficiency, and accessibility for financial forecasting analysis.

#### 3.6 Variables of the Study

The study incorporates both dependent and independent variables to analyze stock market forecasting and volatility behavior.

3.6.1 Dependent Variables

The dependent variables of the study include:

KSE-30  constituent stock prices and predicted index weights

Daily log returns of KSE-30 stocks

Stock market volatility of KSE-30 constituents

Index fund flow proxy derived from weight shifts

Value at Risk (VaR)

These variables represent the financial outcomes that the forecasting models attempt to predict.

3.6.2 Independent Variables

The independent variables include both financial and macroeconomic indicators:

Exchange rate

Inflation rate

Foreign Direct Investment (FDI)

Interest rate

Balance of Trade (BOT)

Gross Domestic Product (GDP)

Crude oil prices

Gold prices

Investor sentiment indicators

Historical stock prices

Trading volume

Technical indicators

KSE-30 Index weight allocation and flow proxy

These variables are selected based on prior literature that identifies their influence on stock market behavior and volatility.

#### 3.7 Conceptual Framework

The conceptual framework of the study explains the relationship between macroeconomic variables, investor behavior, KSE-30 constituent stock data, index fund flows, and stock market forecasting.

Independent variables such as exchange rates, inflation, FDI, investor sentiment, and historical stock prices influence dependent variables including KSE-30 returns, volatility, and market performance.

The forecasting models including ARIMA, GARCH, ANN, LSTM, Random Forest, and hybrid models are applied to determine the predictive relationship between these variables.

The framework assumes that both macroeconomic conditions and behavioral factors significantly influence stock market performance and volatility.

#### 3.8 Data Preprocessing and Transformation

Financial data often contain noise, missing values, outliers, and non-stationary patterns. Therefore, preprocessing is necessary before applying forecasting models.

The following preprocessing techniques are applied:

Removal of missing or duplicate observations

Logarithmic transformation of stock prices

Normalization and standardization of variables

Conversion of prices into returns

Handling of outliers

Feature scaling for machine learning models

Stationarity transformation through differencing

The dataset is further divided into training, validation, and testing sets to evaluate model performance effectively.

#### 3.9 Econometric and Statistical Techniques

The study employs both traditional econometric techniques and advanced machine learning models to forecast stock prices and volatility.

3.9.1 Descriptive Statistics

Descriptive statistics are used to summarize the characteristics of the data. Measures including mean, median, standard deviation, skewness, kurtosis, minimum, and maximum values are calculated.

Descriptive analysis helps identify the distribution and volatility of stock market returns.

3.9.2 Stationarity Tests

Stationarity tests are applied to determine whether the time-series data exhibit constant mean and variance over time.

The study uses:

Augmented Dickey-Fuller (ADF) Test

Phillips-Perron (PP) Test

KPSS Test

If the data are non-stationary, differencing techniques are applied.

3.9.3 ARIMA Model

The Autoregressive Integrated Moving Average (ARIMA) model is used for time-series forecasting.

The ARIMA model consists of three parameters:

Autoregressive term (p)

Differencing term (d)

Moving average term (q)

The model captures linear relationships within financial time-series data and is widely used for short-term forecasting.

3.9.4 GARCH Family Models

The Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model is employed to capture volatility clustering in stock returns.

The study applies:

GARCH(1,1)

EGARCH

TGARCH

GJR-GARCH

These models help analyze conditional variance and leverage effects within the stock market.

3.9.5 Artificial Neural Network (ANN)

Artificial Neural Networks (ANN) are applied to model nonlinear relationships within financial data.

The ANN model consists of:

Input layer

Hidden layers

Output layer

The backpropagation algorithm is used to train the network and minimize forecasting error.

3.9.6 Long Short-Term Memory (LSTM) Model

Long Short-Term Memory (LSTM) is a deep learning model specifically designed for sequential and time-series data.

LSTM networks are capable of capturing long-term dependencies and overcoming the vanishing gradient problem associated with traditional recurrent neural networks.

The model is applied to forecast stock prices and volatility using historical market data.

3.9.7 Random Forest Model

Random Forest is an ensemble machine learning technique based on decision trees.

The model improves prediction accuracy by combining multiple trees and reducing overfitting.

Random Forest is particularly useful for handling nonlinear relationships and high-dimensional financial datasets.

3.9.8 Hybrid Forecasting Framework

The study also develops hybrid forecasting models by integrating traditional econometric and machine learning approaches.

Examples include:

ARIMA-LSTM

GARCH-LSTM

ANN-GARCH

Random Forest-GARCH

Hybrid models are expected to provide better forecasting accuracy by combining the strengths of multiple techniques.

#### 3.10 Model Evaluation Techniques

The performance of forecasting models is evaluated using statistical error metrics.

The study uses:

Root Mean Square Error (RMSE)

Mean Absolute Error (MAE)

Mean Absolute Percentage Error (MAPE)

Mean Squared Error (MSE)

R-squared (R²)

Lower forecasting errors indicate better predictive performance.

Cross-validation techniques are also used to improve model robustness and reduce overfitting.

#### 3.11 Research Hypotheses

The study develops the following hypotheses:

H1:

Macroeconomic variables significantly influence KSE-30 Index performance.

H2:

There is significant volatility clustering in the Pakistan Stock Exchange.

H3:

Machine learning and deep learning models outperform traditional statistical forecasting models.

H4:

Hybrid forecasting models provide more accurate predictions than standalone models.

H5:

Investor behavior and index fund flow patterns significantly affect KSE-30 stock market forecasting and weight allocation.

#### 3.12 Ethical Considerations

The study follows ethical research practices throughout the research process.

Only publicly available secondary data are used.

Data sources are properly acknowledged.

No manipulation or falsification of data is performed.

Statistical analysis is conducted objectively and transparently.

Confidentiality and intellectual property rights are respected.

The research is conducted solely for academic purposes.

#### 3.13 Limitations of the Methodology

Despite adopting advanced forecasting techniques, the methodology has certain limitations.

Financial markets are highly volatile and influenced by unpredictable events.

Historical data may not fully capture future market behavior.

Machine learning models require large computational resources.

Data availability limitations may restrict inclusion of certain variables.

Sudden political, economic, or global shocks may reduce forecasting accuracy.

These limitations should be considered while interpreting the study results.

#### 3.14 Chapter Summary

This chapter explained the research methodology adopted for analyzing KSE-30 index fund flow patterns, stock market forecasting and volatility behavior in the Pakistan Stock Exchange. The chapter discussed the research philosophy, design, population, sample, variables, data collection methods, preprocessing techniques, and forecasting models used in the study. Both econometric approaches and machine learning techniques including ARIMA, GARCH, ANN, LSTM, Random Forest, and hybrid models were described in detail.

The chapter also explained model evaluation methods, research hypotheses, ethical considerations, and methodological limitations. The next chapter presents the empirical analysis, model estimation, and interpretation of results.

## CHAPTER 04

### RESULTS AND ANALYSIS

#### 4.1 Introduction

This chapter presents the empirical findings of the study, derived from the KSE-30 focused workflow. The analysis follows a logical progression from initial data exploration to complex predictive and structural modelling. We first examine the descriptive characteristics of the dataset before moving into fund-flow forecasting, volatility dynamics, and formal tests of market efficiency.

The results are interpreted within the context of the final sample structure: a daily KSE-30 dataset comprising 1,300 observations and an aggregate monthly fund-flow dataset of 60 observations. Given the relatively small monthly sample size, our analysis prioritizes directional accuracy and comparative model performance over absolute R-squared values in the forecasting block.

#### 4.2 Exploratory Data Analysis Results

The exploratory phase confirms that the merged KSE-30 dataset captures distinct and economically significant patterns. The daily reconstructed KSE-30 index return series exhibits a mean of 0.0736% and a standard deviation of 1.3945%. Extreme market movements were observed, with a minimum daily return of -10.2414% (March 2, 2026) and a maximum of 9.3245% (May 12, 2025). The Jarque-Bera test yields a p-value of effectively zero, indicating that the returns are non-normal and characterized by "fat tails," which justifies the subsequent application of GARCH-family models.

Figure 4.1: Reconstructed cumulative KSE-30 return over the final analysis period.

4.2.1 AUM Trends

The analysis of Assets Under Management (AUM) reveals a highly concentrated fund universe. AKD serves as the dominant component of the aggregate sector AUM, while NBP and NTI/NIT remain significantly smaller. Specifically, peak AUM levels reached approximately 2107.24 for AKD, compared to 94.00 for NBP and 212.86 for NTI/NIT. By January 30, 2026, the aggregate sector AUM stood at 2414.10 million PKR. This concentration implies that aggregate sector-flow movements are predominantly driven by the behavior of the largest fund.

Figure 4.2: AUM trend and aggregate KSE-30 sector total.

4.2.2 NAV Return Distributions

Daily Net Asset Value (NAV) return distributions across the three funds are wide and leptokurtic. Standard deviations vary significantly: 2.3716% for AKD, 8.2404% for NBP, and 3.4015% for NIT. The extreme kurtosis and Jarque-Bera results (p-value ≈ 0) across all series necessitate the use of robust volatility modelling rather than Gaussian assumptions.

Figure 4.3: Distribution of daily NAV returns for the KSE-30 related funds.

4.2.3 Fund Flow Behavior

The aggregate fund-flow series is characterized by episodic shocks rather than stability. Within the 60-month sample, 38 months recorded positive net flows while 22 months were negative. While the mean monthly flow is 13.6109 million PKR, the series is heavily skewed by outliers, such as the peak inflow of 234.8581 million PKR in December 2025 and a sharp outflow of -44.0068 million PKR in December 2024.

Figure 4.4: Aggregate monthly KSE-30 sector fund flows showing alternating inflow and outflow episodes.

4.2.4 Correlation Analysis

Contemporaneous linear relationships between variables appear weak. The correlation between aggregate flow and monthly KSE-30 returns is a negligible 0.0095. Macroeconomic factors like CPI (-0.0262) and Oil returns (-0.0783) also show minimal direct association. The most notable, albeit mild, relationship is a negative correlation with interest rates (-0.2092). This suggests that predictive power is more likely to reside in lagged or system-based dynamics.

Figure 4.5: Monthly correlation matrix for KSE-30 flow, macroeconomic variables, and index measures.

#### 4.3 Fund-Flow Forecasting Results

Forecasting was conducted using a 35-month training window and a 25-month out-of-sample testing window. The objective was to determine if parsimonious time-series models could outperform a naive random-walk benchmark.

Table 4.1: Aggregate KSE-30 Fund-Flow Forecasting Results

4.3.1 Stationarity and Granger Causality

ADF tests confirm that total_fund_flow (p=0.0000), Oil returns, and USD/PKR returns are stationary. Conversely, Interest Rates and CPI remain non-stationary in level form. Lag-1 Granger causality tests proved weak across the board, with CPI (p=0.0639) being the only variable approaching conventional significance. This indicates that predictive gains in our models stem from combined dynamic structures rather than a single dominant macro driver.

Figure 4.6: Lag-1 Granger causality p-values for macroeconomic variables against aggregate KSE-30 fund flow.

4.3.2 ARIMAX and VAR Results

The ARIMAX(1,0,1) model significantly improved upon the naive benchmark, which had a directional accuracy of only 37.5%. The ARIMAX model increased this accuracy to 75.0% and reduced the MAE from 51.54 to 37.35. While the out-of-sample R-squared remained negative (-0.0935), the improvement in error metrics and sign classification suggests that lagged flow and macro controls provide valuable, albeit subtle, predictive signals.

The VAR(1) system also achieved 75.0% directional accuracy but yielded slightly higher forecast errors (MAE of 39.41) compared to ARIMAX. Nevertheless, the VAR model remains a robust benchmark as it accounts for the joint evolution of flows and macroeconomic conditions.

Figure 4.7: Actual versus predicted aggregate KSE-30 sector flow under ARIMAX and VAR.

#### 4.4 Volatility Modelling Results

This section focuses on the daily volatility of the reconstructed KSE-30 index.

Table 4.2: KSE-30 GARCH and EGARCH Results

4.4.1 GARCH and EGARCH Analysis

The GARCH(1,1) model reveals high volatility persistence ($\alpha + \beta = 0.9667$), indicating that market shocks decay slowly. However, the EGARCH(1,1) model is the preferred specification due to its lower AIC (4209.68 vs 4243.43). The EGARCH results show a significant negative gamma term, confirming a "leverage effect" where negative news triggers a stronger volatility response than positive news of the same magnitude.

Figure 4.8: KSE-30 daily returns and model-implied conditional volatility.

4.4.2 VaR Backtesting

Using the EGARCH(1,1) model for a 5% Value-at-Risk (VaR) backtest, we observed 58 exceedances out of 1,300 observations (4.46%). This close alignment with the nominal 5% level suggests that the model provides a reliable downside risk envelope for the KSE-30.

Figure 4.9: 5 percent VaR backtesting for the preferred KSE-30 volatility model.

#### 4.5 Market Efficiency Results

The efficiency results are mixed rather than one-sided. This is one of the most important substantive findings of the project. Different tests do not all point in the same direction, which means the KSE-30 cannot be described too simply as either fully efficient or fully inefficient.

Table 4.3: KSE-30 Market Efficiency Test Results

4.5.1 Runs Test

With a p-value of 0.0561, the result is borderline. While it technically fails to reject efficiency at the 5% level, the proximity to the threshold suggests the randomness of return signs is not robust.

4.5.2 Variance Ratio Test

VR(2) and VR(4) results (p=0.8752) do not reject the random-walk hypothesis, providing the strongest evidence in favor of efficiency.

Figure 4.10: Variance-ratio evidence for the KSE-30 across alternative lag horizons.

4.5.3 Ljung-Box Test

Conversely, a p-value of 0.0000 indicates significant serial dependence, strongly contradicting the random-walk assumption.

4.5.4 Hurst Exponent Results

The full-series Hurst exponent of 0.6559 points toward "persistence." This reinforces the Ljung-Box findings, indicating that the KSE-30 possesses a "memory" that departs from pure efficiency.

Figure 4.11: Autocorrelation structure of reconstructed KSE-30 returns.

4.6 Summary of Empirical Findings

The empirical analysis yields four key conclusions. First, the KSE-30 fund universe is highly concentrated, making sector-level flows sensitive to the largest players. Second, while absolute PKR magnitudes are difficult to forecast, ARIMAX and VAR models provide significant gains in directional accuracy for fund flows. Third, KSE-30 volatility is persistent and asymmetric, with negative shocks having a disproportionate impact. Finally, the mixed efficiency results, specifically the high Hurst exponent and significant Ljung-Box statistics, suggest that the KSE-30 is not perfectly weak-form efficient. These findings provide the empirical justification for the portfolio rebalancing strategies explored in Chapter 5

## CHAPTER 05

### PORTFOLIO TILT AND REBALANCING APPLICATION

#### 5.1 Introduction

This chapter converts the statistical findings of the previous chapters into a practical KSE-30 application. While Chapter 4 establishes the characteristics of fund flows, volatility, and market efficiency, this section extends the analysis into a stock-level rebalancing framework. By doing so, the project moves from purely descriptive modeling to an operational tool designed to anticipate changes in KSE-30 membership and constituent weights. The objective is to build a predictive framework that uses historical stock characteristics to estimate future index inclusion and weight outcomes, providing a data-driven basis for portfolio positioning.

#### 5.2 Motivation for Portfolio Tilt Using Fund-Flow Signals

The motivation for this application stems directly from the evidence of non-randomness in fund flows and persistence in KSE-30 volatility. Since return dynamics exhibit serial dependence, it is logical to investigate whether these signals can support an "index-tilt" strategy, adjusting portfolio weights ahead of formal index reviews. Figure 5.1 outlines the conceptual framework for this rebalancing application, highlighting the transition from raw market signals to predictive outcomes.

Figure 5.1: Integrated rebalancing and portfolio tilt framework.

#### 5.3 Data and Sample for Rebalancing

The rebalancing analysis is based on a panel of 422 stock-window observations across 16 semi-annual rebalancing periods. The dataset incorporates features such as lagged market capitalization, average daily turnover, and existing index weights. This rich feature set allows the models to capture the fundamental and liquidity-based drivers that the Pakistan Stock Exchange (PSX) considers during its official index reviews.

#### 5.4 Predicting Index Retention and Inclusion

A critical task for any index-tracking or tilt strategy is predicting which stocks will remain in the index. We evaluated multiple models, including a Naive Persistence benchmark, Random Forest, and Logistic Regression.

5.4.1 Retention Probability Results

Logistic Regression emerged as the superior model for predicting index retention, achieving an Area Under the Curve (AUC) that exceeded both the Naive rule and the Random Forest classifier. Figure 6.2 visualizes the retention probabilities, showing a clear distinction between core "safe" constituents and those at risk of exclusion.

Figure 5.2: Estimated retention probabilities for current KSE-30 constituents.

5.4.2 Feature Importance

To understand what drives these predictions, we analyzed the feature importances (Figure 6.3). Unsurprisingly, lagged weights and market capitalization are the most dominant predictors. However, liquidity measures also play a vital role, confirming that the KSE-30 is as much a measure of tradability as it is of size.

Figure 5.3: Feature importance ranking for the rebalancing prediction model.

#### 5.5 Predicting Constituent Weights

Beyond simple inclusion, predicting the exact weight of a stock is essential for portfolio optimization. We compared several regression techniques against a naive benchmark (using the previous period's weight).

Ridge Regression emerged as the preferred model. While the naive benchmark is already strong due to index stability, Ridge Regression provided a modest improvement in Mean Squared Error (MSE) by better handling the "churn" at the bottom of the index. Figure 6.4 illustrates the relationship between actual and predicted weights, showing high alignment for large-cap names but more variance among smaller constituents.

Figure 5.4: Scatter plot of actual versus predicted constituent weights.

#### 5.6 Forecasted Weight Changes

The final stage of the application involves identifying "high-conviction" weight changes. By comparing model-predicted weights with current actual weights, we can identify stocks likely to see an increase in their index representation. Figure 6.5 displays these forecasted changes, which serve as the primary signal for a portfolio tilt strategy.

Figure 5.5: Forecasted direction and magnitude of KSE-30 weight changes.

#### 5.7 Implementation Considerations and Limitations

While the results are promising, it is important to note that the pipeline does not currently include a transaction-cost model or a live-trading backtest. Therefore, this framework should be viewed as a decision-support tool rather than a fully validated algorithmic trading strategy. The evidence suggests that while the models can provide a probabilistic edge in anticipating index changes, they should be used as one component of a broader investment process.

#### 5.8 Chapter Summary

This chapter has demonstrated that the earlier empirical insights into fund flows and market efficiency can be successfully translated into a stock-level rebalancing application. By utilizing Ridge Regression for weight prediction and Logistic Regression for inclusion probability, the pipeline offers a systematic way to anticipate KSE-30 index shifts. The findings suggest that the index will remain concentrated in established large-cap names, though identifying the "at-risk" low-weight stocks provides significant tactical value for portfolio managers looking to tilt their exposure ahead of official announcements.

## CHAPTER 06

### DISCUSSION

#### 6.1 Interpretation of Fund-Flow Prediction Results

The fund-flow forecasting results demonstrate that aggregate KSE-30 sector flow is not perfectly unpredictable, yet it remains a variable that cannot be forecast with high precision in absolute PKR terms. A balanced interpretation is essential: while the naive benchmark performs poorly, particularly in directional terms, the ARIMAX and VAR models produce clear improvements. However, the out-of-sample R-squared values remain slightly negative even for these superior models.

This suggests that the underlying monthly flow process contains usable dynamic information, but the signal is weak relative to the magnitude of shock-driven variation in the series. Consequently, the final models are more reliable as directional tools than as exact point estimators. The strongest evidence for this is found in the directional accuracy results, where both retained dynamic models achieve a 75.0% hit rate compared to only 37.5% for the naive benchmark.

Figure 6.1: Fund-flow model performance scorecard.

#### 6.2 Discussion of Market Efficiency Evidence

The efficiency results are mixed, preventing a simple categorization of the KSE-30 as either fully efficient or fully inefficient. The variance ratio and runs tests suggest that the index frequently behaves like a random walk, yet the Ljung-Box and Hurst exponent results point toward significant serial dependence.

This "mixed-efficiency" finding is consistent with the nature of an emerging market. While large-cap, liquid stocks might be priced relatively efficiently, the overall index still exhibits "pockets" of memory and exploitable structure. Figure 7.2 summarizes this evidence, illustrating how different statistical lenses can lead to varying conclusions about the randomness of KSE-30 returns.

Figure 6.2: Summary of market efficiency evidence for the KSE-30.

#### 6.3 Volatility Dynamics and Leverage Effects

The presence of high volatility persistence and a significant leverage effect in the EGARCH model highlights the vulnerability of the KSE-30 to downside shocks. As visualized in the volatility regime analysis (Figure 7.3), periods of market stress are not isolated incidents but tend to cluster, creating prolonged windows of elevated risk. The leverage effect confirms that investors in the Pakistani market react more aggressively to negative news, which is a critical consideration for risk management and Value-at-Risk (VaR) calibration.

Figure 6.3: KSE-30 realized volatility regimes over the study period.

#### 6.4 Choice of Modeling Techniques

Throughout the study, a deliberate choice was made to favor parsimonious econometric models for monthly flow prediction while utilizing machine learning for the richer rebalancing dataset. This decision was driven by the "sample-to-complexity" ratio. For the 60-observation monthly flow series, complex machine learning models risk overfitting. However, for the stock-level rebalancing task—which provides hundreds of observations—machine learning techniques like Logistic Regression and Random Forest offer the depth needed to capture non-linear relationships.

Figure 6.4: Performance comparison across model families (Econometric vs. ML).

#### 6.5 Practical Utility of the Rebalancing Application

The rebalancing application serves as a bridge between theory and practice. By identifying stocks with high exclusion risk and predicting weight changes, the pipeline offers a decision-support framework for institutional investors. Figure 7.5 maps these risks, providing a visual guide for tactical positioning. It is important to emphasize that this framework is a probabilistic aid; without accounting for transaction costs and slippage, it should not be treated as a fully validated live trading strategy.

Figure 6.5: KSE-30 rebalancing risk and opportunity map.

#### 6.6 Study Limitations

While the results provide meaningful insights, they are bounded by certain limitations. The monthly flow sample is relatively small, which limits the ability to make high-precision PKR forecasts. Additionally, the macroeconomic feature set is intentionally compact; drivers such as political events, foreign participation, and specific regulatory announcements are not explicitly modeled. These constraints define the scope of the findings: a careful, KSE-30 focused research study with bounded but actionable predictive claims.

#### 6.7 Chapter Summary

The discussion demonstrates that the KSE-30 environment is neither fully random nor fully predictable. Instead, it contains pockets of exploitable structure strong enough to support disciplined forecasting and rebalancing analysis. By favoring interpretable econometric models for flows and machine learning for rebalancing, the study provides a balanced approach to navigating the complexities of the Pakistani equity market.

## CHAPTER 07

### CONCLUSION AND RECOMMENDATIONS

#### 7.1 Conclusion

This study set out to analyze mutual fund flow patterns and market efficiency in Pakistan through a final workflow centered on the KSE-30. Using cleaned stock, fund, and macroeconomic data, the project successfully constructed aggregate monthly KSE-30 sector flows, reconstructed a daily KSE-30 index series, modelled market volatility, and tested weak-form efficiency. Furthermore, the empirical findings were extended into a practical rebalancing and portfolio-tilt application.

The results demonstrate that KSE-30 related fund flows are not fully random. While exact monthly point prediction remains a challenge due to market noise, parsimonious time-series models, specifically ARIMAX and VAR, improved meaningfully on the naive benchmark in directional terms. In the final cleaned analysis, the ARIMAX model provided the strongest forecast-error performance.

The investigation into market risk revealed strong evidence of volatility clustering and persistence. The EGARCH model outperformed the symmetric GARCH specification, confirming the presence of a leverage effect where negative shocks exert a more significant impact on future volatility than positive shocks of a similar magnitude.

Evidence regarding market efficiency remained mixed. While the runs and variance-ratio diagnostics failed to strongly reject weak-form efficiency, the Ljung-Box and Hurst exponent results indicated clear persistence and serial dependence. Consequently, the KSE-30 cannot be characterized as perfectly weak-form efficient over the study period. Finally, the stock-level rebalancing framework demonstrated that future retention and constituent weights can be modeled with a probabilistic edge, offering a structured foundation for tactical portfolio positioning ahead of official index reviews.

#### 7.2 Key Contributions of the Study

This research makes several distinct contributions to the study of the Pakistani capital market:

Integrated Analytical Pipeline: The study developed a comprehensive end-to-end workflow that bridges raw data extraction from fund sheets and market indices with advanced predictive modeling and risk analysis.

Reconstruction of KSE-30 Dynamics: By reconstructing the daily KSE-30 index and aggregate fund flows, the project provided a unique, high-resolution view of the interplay between capital movement and market performance in Pakistan.

Empirical Validation of Asymmetric Volatility: The study provides formal evidence of the leverage effect in the KSE-30, highlighting that market participants in Pakistan react more intensely to downside news.

Practical Application of Predictive Models: Unlike purely theoretical studies, this research translated statistical signals into an operational rebalancing tool, demonstrating how Logistic and Ridge regression can aid in anticipating index membership shifts.

#### 7.3 Recommendations

Based on the findings of this study, the following recommendations are proposed:

For Institutional Investors: Fund managers should incorporate asymmetric volatility models like EGARCH into their risk management frameworks to better calibrate downside risk envelopes (VaR) during market stress.

For Portfolio Managers: The use of directional fund-flow signals and predictive rebalancing models can be adopted as a decision-support tool to enable gradual portfolio tilts ahead of formal index reviews, potentially reducing execution costs and slippage.

For Market Regulators: The evidence of serial dependence and persistence suggests that the KSE-30 still exhibits informational friction. Continued efforts to improve transparency and data accessibility could help move the market closer to weak-form efficiency.

#### 7.4 Future Research Directions

To build upon the foundation established in this study, the following areas are suggested for future exploration:

Sample Expansion and Feature Enrichment: Future research should expand the monthly fund-flow dataset as more history becomes available and incorporate additional drivers such as foreign portfolio investment (FPI), policy surprises, and political event indicators.

Advanced Model Architectures: With a larger dataset, researchers could explore non-linear modeling families, including regime-switching models, Bayesian time-series approaches, or deep learning sequence models like LSTMs.

Realized Portfolio Backtesting: The rebalancing application should be extended into a full backtest that accounts for turnover constraints, transaction costs, and commission fees to evaluate the net-of-cost performance of an index-tilt strategy.

Multi-Frequency and Sectoral Analysis: Comparing monthly and weekly forecasting horizons, or applying the flow-efficiency framework to specific sectors (e.g., Banking or Energy), could reveal deeper structural insights into the Pakistani equity market.

In summary, this project provides a strong applied foundation for future KSE-30 research. Its primary value lies in the structured framework it leaves behind for richer data integration, broader modelling, and more realistic investment testing.

## REFERENCES

[1] Yamani, E. (2023). The informational role of fund flow in the profitable predictability of mutual funds. Finance Research Letters, 51, 103445.

[2] Larsson, E., & Wergeland, J. (2020). Market efficiency and index fund flow: An empirical study of the relationship between passive investment and broad-market efficiency [Master's thesis, University of Gothenburg]. GUPEA.

[3] Lim, K., & Yoon, S.-J. (2018). Validity of fund flows as a measure of investor sentiment. Journal of Management and Economics, 40(4), 115–134.

[4] Huang, J., Wei, K. D., & Yan, H. (2022). Investor learning and mutual fund flows. Financial Management, 51(3), 739–765.

[5] Barber, B. M., Huang, X., & Odean, T. (2016). Which factors matter to investors? Evidence from mutual fund flows. Review of Financial Studies, 29(10), 2600–2642.

[6] Iqbal, Z., & Shoaib, M. (2025). Long short-term memory method: A case study of Pakistan stock market volatility. Journal of Business and Social Review in Emerging Economies, 11(1), 129–140.

[7] Zahid, S., & Saleem, H. M. N. (2023). Stock volatility prediction using machine learning during Covid-19. Statistics, Computing and Interdisciplinary Research, 5(2), 99–119.

[8] Iqbal, Z., & Naz, L. (2025). Modelling volatility of Pakistan stock market using family of GARCH models. Journal of Economic Impact, 7(1), 48–54.

[9] Jadoon, A. K., Mahmood, T., Sarwar, A., Javaid, M. F., & Iqbal, M. (2024). Prediction of stock market movement using Long Short-Term Memory (LSTM) artificial neural network: Analysis of KSE 100 Index. Pakistan Journal of Life and Social Sciences, 22(1), 107–119.

[10] Fraz, T. R., Fatima, S., & Uddin, M. (2022). Modeling and forecasting stock market volatility of CPEC founding countries: Using nonlinear time series and machine learning models. JISR Management and Social Sciences & Economics, 20(1), 1–20.

[11] Ali, M., Khan, D. M., Alshanbari, H. M., & El-Bagoury, A. A.-A. H. (2023). Prediction of complex stock market data using an improved hybrid EMD-LSTM model. Applied Sciences, 13(3), 1429.

[12] Batool, K., Ahmed, M. F., & Ismail, M. A. (2022). A hybrid model of machine learning model and econometrics' model to predict volatility of KSE-100 index. Reviews of Management Sciences, 4(1), 225–239.

[13] Sabri, R., & Iqbal, S. (2024). Comparative analysis of univariate and deep learning models for stock market prediction in frontier markets: A case study of Pakistan, Bangladesh, and Sri Lanka. Business Review, 19 (2), 54–73.

[14] Rouf, N., Malik, M. B., Arif, T., Sharma, S., Singh, S., Aich, S., & Kim, H.-C. (2021). Stock market prediction using machine learning techniques: A decade survey on methodologies, recent developments, and future directions. Electronics, 10(21), 2717.

[15] Asif, M., & Aziz, A. (2016). Equity market volatility using garch models- evidence from Pakistan stock exchange (kse-100 index). International Journal of Accounting and Economics Studies, 4(2), 125–130.

[16] Fatima, U., Zafar, R., Shah, S. A., & Samad, A. (2025). Blending econometric and deep learning approaches for enhanced volatility forecasting of the KSE-100 index. Journal of Finance and Data Science, 11(1), 88–104.

[17] Rasheed, M. U., Fareed, T., Ahmed, B., & Badar, M. (2024). Determining the efficacy of GARCH type models for estimating VaR in case of equities enlisted in PSX. Journal of Management and Social Sciences, 12(3), 442–458.

[18] Jadoon, A. K., Mahmood, T., Sarwar, A., Javaid, M. F., & Iqbal, M. (2024). Prediction of stock market movement using Long Short-Term Memory (LSTM) artificial neural network: [19] Analysis of KSE 100 index. Global Strategic & Management Review, 6(1), 17–28.

Munawar, T., Mushtaq, L., & Siddiqui, M. H. (2025). Interactive GARCH-based volatility predictor for financial markets. Journal of Financial Technology and Analysis, 3(2), 142–159.

[20] Bukhari, K., Jadoon, A. K., Iqbal, M., & Arshad, A. (2023). Predicting stock market trends based on macroeconomic indicators through machine learning approach: A case study of KSE 100 index. Journal of Finance and Economics Research, 8(1), 42–61.

[21] Javid, I., Ghazali, R., Syed, I., Zulqarnain, M., & Husaini, N. A. (2022). Study on the Pakistan stock market using a new stock crisis prediction method. PLOS ONE, 17(10), e0275022.

[22] Zaffar, A., & Hussain, S. M. A. (2022). Modeling and prediction of KSE – 100 index closing based on news sentiments: An applications of machine learning model and ARMA (p, q) model. Multimedia Tools and Applications, 81(23), 33311–33333.

[23] Hassan, H., Niaz, A., Rooh, S., & Qureshi, J. A. (2025). Emerging stock market performance and macro economic fundamentals: Evidence from Pakistan Stock Exchange. International Journal of Social Sciences Bulletin, 3(4), 812–826.

[24] Latif, S., Javaid, N., Aslam, F., Aldegheishem, A., Alrajeh, N., & Bouk, S. H. (2024). Enhanced prediction of stock markets using a novel deep learning model PLSTM-TAL in urbanized smart cities. Heliyon, 10(6), e27747.

[25] Khan, S. N., Khan, S. U., Shafique, O., Ansar, Z., Imran, P., Altamish, M. H., & Hamza, A. (2025). Comparative analysis of stock market price behavior through machine learning approaches. Quantum Journal of Social Sciences and Humanities, 6(2), 29–41.

[26] Munir, T., Mamlook, R. E., Rahman, A. R., Alrashidi, A., & Yaseen, A. M. (2024). COVID-19's influence on Karachi stock exchange: A comparative machine learning algorithms study for forecasting. Heliyon, 10(13), e33190.

[27] Qambrani, S. A., Ahmed, I., & Basit, A. (2025). The predictive accuracy of ARIMA models for stock market indices: A case study of KSE-100. Journal for Business Education and Management, 5(1), 1–22.

[28] Ali, M. S., & Javed, A. (2020). Modeling and forecasting volatility in Pakistan stock exchange. International Journal of Sciences: Basic and Applied Research (IJSBAR), 54(1), 234–241.

[29] Khan, I., & Khattak, A. A. (2026). Dynamics of weak-form market efficiency in an emerging market: Evidence from Pakistan’s KSE-100 index using GARCH and random walk approaches. Research Consortium Archive, 10(4), 373–388.

[30] Akber, U., & Muhammad, N. (2014). Is Pakistan stock market moving towards weak-form efficiency? Evidence from the Karachi stock exchange and the random walk nature of free-float of shares of KSE 30 index. Asian Economic and Financial Review, 4(6), 808–836.

[31] Fausch, J., Frigg, M., Ruenzi, S., & Weigert, F. (2025). Machine learning mutual fund flows. Review of Financial Studies, 38(4), 512–548.

[32] Mohsin, M., Naiwen, L., Zia-UR-Rehman, M., Naseem, S., & Baig, S. A. (2024). The volatility of bank stock prices and macroeconomic fundamentals in the Pakistani context: An application of GARCH and EGARCH models. Journal of Public Affairs, 24(2), e2244.

[33] Iqbal, J., Sandhu, M. A., Amin, S., & Manzoor, A. (2019). Portfolio selection and optimization through neural networks and Markowitz model: A case of Pakistan stock exchange listed companies. Review of Economics and Development Studies, 5(1), 185–196.

[34] Chu, N., Dao, B., Pham, N., Nguyen, H., & Tran, H. (2023). Predicting mutual funds’ performance using deep learning and ensemble techniques. arXiv preprint arXiv:2209.09649 (Revised July 2023).

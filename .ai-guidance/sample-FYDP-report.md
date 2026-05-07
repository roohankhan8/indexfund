# Assessing Dynamic Hedge Effectiveness and Statistical Arbitrage with Conventional and Deep Learning Models in Equity Futures

Source PDF: `FYDP.pdf`
Pages: 134

This Markdown file is a text extraction of the source PDF with repeated headers and footers lightly cleaned.

## Page 1

Assessing Dynamic Hedge Effectiveness
and Statistical Arbitrage with
Conventional and Deep Learning Models
in Equity Futures

Group Number: 01 Batch: 2021– 2025

Group Member Names:

Ammar Sheikh CF-21034

Hussain Aftab CF-21030

Zaara Asim Mirza CF-21056

Hanadi Sabir CF-21058

Approved by

……………………………………………………………………………………………...

Ms. Komal Batool
Lecturer, NED UET
Project Advisor

© NED University of Engineering & Technology. All Rights Reserved – May 2025

## Page 2

Author’s Declaration

We declare that we are the sole authors of this project. It is the actual copy of the project that
was accepted by our advisor(s) including any necessary revisions. We also grant NED
University of Engineering and Technology permission to reproduce and distribute electronic
or paper copies of this project.

Signature and Date

Signature and Date

Signature and Date

Signature and Date

...............................

...............................

...............................

...............................
15th June 2025 15th June 2025 15th June 2025 15th June 2025
Ammar Sheikh Hussain Aftab Zaara Asim Mirza Hanadi Sabir
CF-21034 CF-21030 CF-21056 CF-21058
sheikh4405575
@cloud.neduet.edu.pk
aftab4401690
@cloud.neduet.edu.pk
mirza4431125
@cloud.neduet.edu.pk
sabir4400920@
cloud.neduet.pk

## Page 3

Statement of Contributions

Throughout this project, our team embraced both unity and healthy specialization to achieve a
robust and insightful analysis of minimum variance hedge ratios for the KSE -30 Index and its
futures contracts. From the outset, every member —Ms. Hanadi Sabir, Ms. Zaara Asim Mirza,
Mr. Hussain Aftab, and Mr. Ammar Sheikh —came together to collect and adjust futures data,
diligently filling missing values and ensuring data continuity. This foundational work was carried
out as a cohesive unit, reflecting our shared commitment and collective effort without which
none of our subsequent achievements would have been possible.
As we progressed to the core modeling phase, we honored individual strengths and interests by
dividing tasks between mathematical/statistical and deep learning approaches. Ms. Hanadi Sabir
and Ms. Zaara Asim Mirza applied their exceptional analytical skills to develop and refine the
conventional DCC -GARCH and Copula -GARCH models, demonstrating both precision and
creativity in their statistical formulations. Meanwhile, Mr. Hussain Aftab and Mr. Ammar Sheikh
harnessed their deep learning expertise to build and optimize our LSTM -CNN and FT -Net
Hybrid architectures, driving innovation in self -supervised learning for dynamic hedge ratio
estimation.
When we reached the stage of eva luating Fuzzy TOPSIS and Statistical Arbitrage, we
deliberately structured our work to maximize shared learning. First, Ms. Zaara Asim Mirza and
Mr. Hussain Aftab conducted a thorough study of the Fuzzy TOPSIS method and presented their
findings to the gro up, while Mr. Ammar Sheikh and Ms. Hanadi Sabir simultaneously
investigated statistical arbitrage opportunities and explained their conclusions. After these initial
presentations, we held a group discussion to ensure everyone fully understood both techniqu es.
To reinforce this collective knowledge and test our grasp of each other’s insights, we then
swapped implementation responsibilities: Ms. Zaara Asim Mirza and Mr. Hussain Aftab took on
the statistical arbitrage coding based on the group’s shared underst anding, and Mr. Ammar
Sheikh together with Ms. Hanadi Sabir implemented the Fuzzy TOPSIS ranking procedure using
the insights provided by their teammates. This deliberate role reversal guaranteed that each team
member not only taught but also applied both methodologies, fostering deeper comprehension
and stronger collaboration across the entire project.

## Page 4

Finally, inspired by our model comparisons, we collectively decided to extend the FT-Net Hybrid
model by incorporating macroeconomic variables, with Mr. Amm ar Sheikh and Mr. Hussain
Aftab leading its advanced development and Ms. Hanadi Sabir and Ms. Zaara Asim Mirza
meticulously managing the economic data collection and transformation. Throughout the report
writing, all four of us collaborated seamlessly, mer ging our analyses, interpretations, and
reflections into a coherent and polished document. It has been an honor to work alongside such
dedicated and talented colleagues, whose enthusiasm, expertise, and mutual respect have driven
this project to its successful conclusion.

## Page 5

Executive Summary
This project addresses the pressing challenge of managing risk and uncovering arbitrage
opportunities in the Pakistani equity futures market, particularly for the KSE -30 index. In
emerging economies like Pakistan, financial markets are often marked by heightened volatility,
limited liquidity, and frequent inefficiencies that undermine the effectiveness of conventional
static hedging strategies. Against this backdrop, our study aims to devise and asses s dynamic
hedging methodologies that adapt in real time to market conditions, thereby delivering superior
risk mitigation and the potential for statistical arbitrage.
To achieve these objectives, we utilized daily data for the KSE -30 index and its correspo nding
futures, supplemented by weekly macroeconomic variables including GDP growth, SPI -based
inflation, and SBP policy rates. This comprehensive dataset, sourced from the Pakistan Stock
Exchange and the Pakistan Bureau of Statistics, enabled the development and rigorous testing of
both econometric and deep learning models for dynamic hedge ratio estimation.
The methodology centers on four advanced frameworks: the Dynamic Conditional Correlation
GARCH model, an enhanced Copula -DCC–GARCH model incorporating Student-t copula for
improved tail dependence, a hybrid LSTM –CNN deep learning model adept at capturing both
short-term and sequential dependencies in returns, and the novel FT-Net Hybrid, which uniquely
combines Fourier-based spectral analysis with temporal convolution to recognize complex, non-
stationary market patterns. Each model’s hedging effectiveness was assessed using variance
reduction, RMSE, MAD, directional accuracy, Sharpe ratio, average hedged return, VaR and
CVaR reductions via Extreme Value T heory, and time complexity. These measures were
synthesized through the Fuzzy TOPSIS multi -criteria decision -making process to provide an
integrated model ranking.
Empirical findings consistently highlight the FT-Net Hybrid as the most effective model for risk
minimization, excelling in variance reduction, tail -risk control, and predictive accuracy.
Moreover, the model exhibited mean -reversion and statistical arbitrage properties,
outperforming traditional methods in exploiting market inefficiencies. When further extended to
incorporate macroeconomic variables, the FT-Net Hybrid demonstrated enhanced hedge stability
and adaptability to macro shocks, underscoring its practical value.

## Page 6

Acknowledgments

We would like to express our most sincere gratitude to our Project Advisor, Miss Komal Batool,
for her invaluable guidance, insightful suggestions, and constant encouragement throughout the
course of our project, “Assessing Dynamic Hedge Effectiveness and Statistical Arbitrage with
Conventional and Deep Learning Models in Equity Futures.” Her unwavering support was
instrumental in shaping the direction and successful completion of our research. We are deeply
thankful to our Co -Supervisor, Miss Ubaidah Fatima, for her constructive feedback and
continuous motivation, which played a significant role in refining our work and deepening our
understanding of the subject. We also extend our heartfelt thanks to our External Supervisor, Mr.
Asad Khan, whose industry expertise and practical insights provide d us with a much broader
perspective and helped us connect our academic work to real-world challenges.

We would also like to thank the FYDP Committee for giving us the opportunity to complete this
project and for their guidance at every stage. Their valuable advice has been a tremendous help
throughout our research journey.

## Page 7

Table of Contents
Contents
1 Introduction ....................................................................................................................... 1
1.1 Background Information ................................................................................................................1
1.2 Significance and Motivation ..........................................................................................................3
1.3 Aims and Objectives ......................................................................................................................3
1.4 Methodology ...................................................................................................................................4
1.5 Contributions of the Study .............................................................................................................5
1.6 Report Outline ................................................................................................................................6
2 Literature Review .............................................................................................................. 7
2.1 Introduction ....................................................................................................................................7
2.2 Bibliometric Analysis .....................................................................................................................7
2.2.1 Keyword Analysis ................................................................................................................8
2.2.2 Yearly Research Trends and Publication Volume ................................................................9
2.2.3 Most Influential Countries ..................................................................................................10
2.2.4 Three Factor Analysis .........................................................................................................11
2.3 Systematic Review of Key Research Streams ..............................................................................12
3 Data Collection and Processing ...................................................................................... 16
3.1 Data Processing ............................................................................................................................17
3.1.1 Filling the Missing Values ..................................................................................................17
3.1.2 Adjustments of the Future Contracts ..................................................................................19
4 Research Methodology .................................................................................................... 22
4.1 Statistical Tests Pre-Model Implementation ................................................................................23
4.1.1 Augmented Dickey-Fuller (ADF) Test for Stationarity ......................................................23
4.1.2 Engle’s ARCH Test ............................................................................................................23
4.1.3 Modeling Approaches for Hedge Ratio Estimation ............................................................25
4.1.4 Dynamic Conditional Correlation (DCC) GARCH ............................................................26
4.1.5 Dynamic Copula DCC GARCH Model ..............................................................................29
4.1.6 Long-Short Term Memory – Convolutional Neural Network (LSTM–CNN) Hybrid Model
4.1.7 Fourier Transform Network (FT-Net) Hybrid Model .........................................................42
4.1.8 Model Evaluation and Diagnostics .....................................................................................51
4.2 Post Model Performance Evaluation ............................................................................................53
4.2.1 Statistical Tests Post-Model Implementation .....................................................................53
4.2.2 ARCH LM Test ..................................................................................................................54
4.2.3 Ljung Box Q Test ...............................................................................................................55
4.2.4 KPSS Test ...........................................................................................................................55
4.2.5 Theil U Statistic ..................................................................................................................56
4.2.6 Performance Evaluation ......................................................................................................56
4.2.7 Hedged Returns Estimation ................................................................................................60

## Page 8

5 Comparative Analysis using Fuzzy TOPSIS .................................................................. 64
5.1 Introduction ..................................................................................................................................64
5.2 Implementation of the TOPSIS methodology ..............................................................................64
5.2.1 Define the decision-making problem ..................................................................................64
5.2.2 Determination of AHP weights ...........................................................................................66
5.2.3 Fuzzification of decision criteria ........................................................................................68
5.2.4 Normalize the decision matrix ............................................................................................69
5.2.5 Calculate weighted normalized matrix ...............................................................................69
5.2.6 Determine positive and negative ideal solutions ................................................................69
5.2.7 Calculate Euclidean distance from ideal solutions .............................................................70
5.2.8 Rank the alternatives ...........................................................................................................70
5.3 Summary ......................................................................................................................................71
6 Statistical Arbitrage Analysis .......................................................................................... 73
6.1 Introduction ..................................................................................................................................73
6.2 Relevance in Equity Futures Markets ..........................................................................................73
6.3 Motivation for Statistical Arbitrage in Hedging Context .............................................................74
6.4 Statistical Arbitrage in the Context of the Project ........................................................................75
6.5 Methodological Framework for Statistical Arbitrage ...................................................................75
6.6 Construction of Hedged Return Series .........................................................................................76
6.7 Rolling Statistics and Z-Score Normalization ..............................................................................77
6.8 Stationarity Testing and Justification ...........................................................................................78
6.9 Signal Generation and Trade Execution Logic ............................................................................79
6.10 Mean-Reversion Signal Design ....................................................................................................79
6.11 Trade Position Management .........................................................................................................81
6.12 Parameter Optimization ................................................................................................................81
6.13 Empirical Results and Visualization ............................................................................................82
6.13.1 Model-by-Model Strategy Performance .............................................................................83
6.13.2 Visualization of Hedged Spread Dynamics ........................................................................84
6.13.3 Equity Curve and Comparative Performance .....................................................................87
6.14 Interpretation and Practical Implications......................................................................................90
6.14.1 Model Selection and Arbitrage Exploitation ......................................................................90
6.14.2 Comparative Assessment ....................................................................................................90
6.14.3 Robustness and Sensitivity Checks .....................................................................................91
6.14.4 Real-Time Exploitation Strategy ........................................................................................92
6.15 Discussion and Implications .........................................................................................................93
6.15.1 Interpretation of Arbitrage Profitability ..............................................................................93
6.15.2 Limitations and Caveats......................................................................................................94
6.16 Summary ......................................................................................................................................95
7 Implementing FT-Net Hybrid in a Macroeconomic Context .......................................... 96
7.1 Introduction and Rationale for Macroeconomic Integration ........................................................96
7.2 Data Foundation: Weekly Market and Macroeconomic Features ................................................96
7.3 Feature Relationships: Empirical Insights from the Data .............................................................97
7.4 Exploratory Visualization of Macroeconomic Variables .............................................................98

## Page 9

7.5 FT-Net Hybrid Architecture: Integrating Macroeconomic Information ......................................99
7.6 Mathematical Formulation of the FT-Net Hybrid ......................................................................100
7.7 Model Training, Filtering, and Statistical Cleanliness ...............................................................101
7.8 Performance Assessment and Statistical Validation ..................................................................101
7.8.1 Training and Validation Loss Curves ...............................................................................101
7.8.2 MVHR Time Series ..........................................................................................................102
7.8.3 Cumulative Returns: Hedged vs. Unhedged .....................................................................102
7.8.4 Economic and Practical Interpretation ..............................................................................104
7.9 Summary ....................................................................................................................................104
8 Conclusion..................................................................................................................... 105
8.1 Summary ....................................................................................................................................105
8.2 Recommendations for Future Work ...........................................................................................109
9 References ..................................................................................................................... 111
Appendix A: Statistical Diagnostic Tests on Residuals Post Model Implementation ............. 114
Appendix B: Data Collection and Processing of Macro Economic Variables ......................... 115

## Page 10

List of Figures

Figure 1: Cartography Results obtained from VoS Viewer ......................................................... 9
Figure 2: Yearly Research Trends .............................................................................................. 10
Figure 3: Number of publications in every country ................................................................... 11
Figure 4: Three Factor Analysis ................................................................................................. 12
Figure 5: Architecture and Pipeline of the Random Forest Regressor ....................................... 18
Figure 6: ACF plots of residuals and squared residuals of KSE 30 index return....................... 24
Figure 7: ACF plots of residuals and squared residuals of KSE 30 index futures return .......... 25
Figure 8: Standardized Returns of KSE 30 Index and its Future Contracts ............................... 29
Figure 9: ACF and PACF of Index and Future Returns ............................................................. 36
Figure 10: Architecture and Pipeline of LSTM-CNN Hybrid Model ........................................ 39
Figure 11: Architecture and Pipeline of FT-Net Hybrid Model ................................................. 46
Figure 12: Dynamic MVHRs obtained using DCC GARCH .................................................... 58
Figure 13: Dynamic MVHRs obtained using Copula DCC GARCH ........................................ 58
Figure 14: Dynamic MVHRs obtained using LSTM-CNN ....................................................... 59
Figure 15: Dynamic MVHRs obtained using FT-Net Hybrid .................................................... 59
Figure 16: Cumulative Hedged and Unhedged Returns – DCC GARCH ................................. 60
Figure 17: Cumulative Hedged and Unhedged Returns – Copula DCC GARCH ..................... 61
Figure 18: Cumulative Hedged and Unhedged Returns – LSTM-CNN .................................... 62
Figure 19: Cumulative Hedged and Unhedged Returns – FT-NET Hybrid............................... 63
Figure 20: Fuzzified AHP Criteria Weights ............................................................................... 68
Figure 21: Model Ranking by Fuzzy TOPSIS ........................................................................... 70
Figure 22: Graphical User Interface (GUI) for Statistical Arbitrage ......................................... 83
Figure 23: Spreads and Bands – Copula DCC GARCH ............................................................ 85
Figure 24: Spreads and Bands – DCC GARCH ......................................................................... 86
Figure 25: Spread and Bands – FTNET Hybrid ......................................................................... 86
Figure 26: Spreads and Bands – LSTM-CNN............................................................................ 86
Figure 27: Strategy Equity Curve – DCC GARCH ................................................................... 87
Figure 28: Strategy Equity Curve – Copula DCC GARCH ....................................................... 88
Figure 29: Strategy Equity Curve – LSTM-CNN ...................................................................... 88
Figure 30: Strategy Equity Curve – FTNET Hybrid .................................................................. 88
Figure 31: Cumulative PnL Comparison Across Models .......................................................... 89
Figure 32: Correlation Matrix .................................................................................................... 97
Figure 33: Weekly Series of GDP, Inflation, and SBP Policy Rate ........................................... 98
Figure 34: FT-Net Hybrid Model Architecture .......................................................................... 99
Figure 35: Convergence of Training and Validation Loss ....................................................... 101
Figure 36: Dynamic MVHRs Using FT-Net Hybrid Model with Macroeconomic Variables . 102
Figure 37: Hedged Returns in Comparison with Unhedged Returns ....................................... 102
Figure 38: Weekly Interpolated GDP and M2 Money Supply ................................................. 117

## Page 11

List of Tables

Table 1: Performance Metrics of Hedging Models – Training Model ....................................... 66
Table 2: Performance Metrics of Hedging Models – Testing Model ......................................... 66
Table 3: Computation and Fuzzification of AHP Pairwise Comparison Matrix ....................... 67
Table 4: Statistical Arbitrage Performance Metrics by Model. .................................................. 84
Table 5: Statistical Tests for Residuals on Training Data – Mathematical Model................... 114
Table 6: Statistical Tests for Residuals on Testing Data – Mathematical Model .................... 114
Table 7: Statistical Tests for Residuals – LSTM-CNN ............................................................ 114
Table 8: Statistical Tests for Residual – Ft-Net Hybrid ........................................................... 114

## Page 12

List of Abbreviations

Abbreviation Full Form Abbreviation Full Form
ACF Autocorrelation Function HE Hedge Effectiveness
ADCC Asymmetric Dynamic
Conditional Correlation HFT High Frequency Trading
ADF Augmented Dickey-Fuller Test IGC Integrated GARCH
AHP Analytic Hierarchy Process IUP Index Underlying Portfolio
API Application Programming
Interface KIBOR Karachi Interbank Offered Rate
ARCH Autoregressive Conditional
Heteroskedasticity KPSS Kwiatkowski-Phillips-Schmidt-
Shin Test
ARCHT
Autoregressive Conditional
Heteroskedasticity with t-
distribution
KSE Karachi Stock Exchange
ARMA Autoregressive Moving Average LM Lagrange Multiplier
BEKK Baba-Engle-Kraft-Kroner Model LSTM Long Short-Term Memory
CAPM Capital Asset Pricing Model M2 Money Supply (Broad)
CCC Constant Conditional
Correlation MAD Mean Absolute Deviation
CDF Cumulative Distribution
Function MAE Mean Absolute Error
CMES Chicago Mercantile Exchange
Swap MAPE Mean Absolute Percentage
Error
CNN Convolutional Neural Network MCDM Multi-Criteria Decision
Making
CNX CNX Nifty Index MLE Maximum Likelihood
Estimation
CNY Chinese Yuan MSE Mean Squared Error
CSI China Securities Index MVHR Minimum Variance Hedge
Ratio

## Page 13

Abbreviation Full Form Abbreviation Full Form
CV Coefficient of Variation NCCPL National Clearing Company
of Pakistan Limited
CVAR Conditional Value at Risk NIS Net Investment Strategy
DCC Dynamic Conditional
Correlation OLS Ordinary Least Squares
DFT Discrete Fourier Transform PACF Partial Autocorrelation
Function
DL Deep Learning PI Price Index
ECM Error Correction Model PIS Positive Ideal Solution
EVT Extreme Value Theory PSX Pakistan Stock Exchange
EWMA Exponentially Weighted
Moving Average RMSE Root Mean Squared Error
FFT Fast Fourier Transform RNN Recurrent Neural Network
FTNET Fourier Transform-based
Neural Network SBP State Bank of Pakistan
FTS Financial Time Series SIF Statistical Information
Function
GARCH
Generalized Autoregressive
Conditional
Heteroskedasticity
SPI Sensitive Price Indicator
GBM Geometric Brownian
Motion TFN Triangular Fuzzy Number
GDP Gross Domestic Product TFT Temporal Fusion
Transformer
GRU Gated Recurrent Unit TOPSIS
Technique for Order
Preference by Similarity to
Ideal Solution
GUI Graphical User Interface VAR Vector Autoregression
XGB Extreme Gradient Boosting

## Page 14

United Nations Sustainable Development Goals

The Sustainable Development Goals (SDGs) are the blueprint to achieve a better and
more sustainable future for all. They address the global challenges we face, including
poverty, inequality, climate change, environmental degradation, peace and justice. There is
a total of 17 SDGs as mentioned below. Check the appropriate SDGs related to the project.

□ No Poverty

□ Zero Hunger

□ Good Health and Wellbeing

□ Quality Education

□ Gender Equality

□ Clean Water and Sanitation

□ Affordable and Clean Energy

□ Decent Work and Economic
Growth

□ Industry, Innovation and Infrastructure

□ Reduced Inequalities

□ Sustainable Cities and Communities

□ Responsible Consumption and Production

□ Climate Action

□ Life Below Water

□ Life on Land

□ Peace and Justice and Strong Institutions

□ Partnerships to Achieve the Goals

## Page 15

Similarity Index Report

Following students have compiled the final year report on the topic given below
for partial fulfillment of the requirement for Bachelor’s degree in Computational
Finance.

Project Title
Assessing Dynamic Hedge Effectiveness and Statistical Arbitrage with Conventional and
Deep Learning Models in Equity Futures

S. No. Student Name Seat Number
1. Ammar Sheikh CF-21034
2. Hussain Aftab CF-21030
3. Zaara Asim Mirza CF-21056
4. Hanadi Sabir CF-21058

This is to certify that Plagiarism test was conducted on complete report, and overall
similarity index was found to be less than 20%, with maximum 5% from single source, as
required.

Signature and Date

..................................
Ms. Komal Batool

## Page 16

_No extractable text on this page._

## Page 17

1 Introduction

1.1 Background Information

The management of portfolio risk is a cardinal concern of investors and financial institutions
operating within the stock and derivative markets. Especially in the context of emerging
economies such as Pakistan, the equity markets pose unique challenges and opportunities driven
by market volatility, liquidity, and the evolution of regulatory landscapes. As these challenges
pertain, the literature present on the Pakistani Markets also focuses on the impact of derivatives,
especially the future contracts, on mar ket volatility, efficiency, systematic risk of underlying
stocks, and spot price discovery ( (Khan , 2006); (Malik & Shah, 2016); (Khan & Abbas, 2013))
with scarce literature on effective hedging strategies using future contracts.
From a global perspective, a lot of research has focused on minimum variance hedging strategies
derived from foundational models like the Ordinary Least Squares (OLS) techniq ue
(Ederington, 1979) and the Error Correction Models (ECM) (Ghosh, 1993) to estimate optimal
static hedge ratios to offset the risk exposure in underlying stock portfolios. However, advanced
research has identified that these methodologies fail to hedge risk effectively, due to their
inability to capture autocorrelation, heteroskedasticity, and the time -varying relationship of
variances and covariances of futures and spot return series. Therefore, academic research shifted
from traditional econometric models to more sophisticated models such as the Autoregressive
Conditional Heteroskedasticity (ARCH) (Engle R. F., Autoregressive Conditional
Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation, 1982) and
Generalized Autoregressive Conditional Heteroskedasticity (GARCH) (Bollerslev T. , 1986)
frameworks and their multiple variants. These models, although proven beneficial in capturing
time-series data characteristics such as heteroskedasticity and autocorrelation, may lead to
misspecification errors in the computation of conditional volatilities due to the underlying
model assumptions such as normality, linearity, and symmetry of the time-series data, resulting
in less efficient hedge ratios.
With recent advancements in computational finance, the risk management landscape has entered
a novel era of deep learning and machine learning techniques to develop optimal hedg ing
strategies. Early studies on machine learning models explored simple neural networks to

## Page 18

approximate option sensitivities, followed by advancements in deep learning led to the adoption
of LSTM, hybrid LSTM –CNN models, and Fourier Transform -based feature modules with
temporal convolutional blocks – the FT -Net Hybrid model capable of capturing complex,
nonlinear patterns and explicit spectral analysis components in options. (Chen, Hussain,
Cauteruccio, & Zhang, 2023) Although t he majority of research lays emphasis on option
hedging strategies, the implications of these techniques are immense in hedging through future
contracts as well.
While there is a considerable volume of international research dedicated to the comparative
analysis of various econometric models to ascertain which best estimates minimum -variance
hedge ratios (Floros & Vougas, 2007) , few studies emphasize the comparability between
mathematical and machine learning models. The study by (François, Gauthier, Godin, &
Mendoza, 2025) combines the mathematical delta hedging technique with deep hedging. Rather
than evaluating the models through various performance metrics, it examines a distinctive
proposition suggested by (Horikawa & Nakagawa, 2024) . This proposition states that under a
complete market admitting statistical arbitrage, the discrepancy between the hedging position
derived by a deep hedging model and that of a traditional replicating portfolio, may provide a
statistical arbitrage opportunity. These studies explore a unique perspective on the application
of statistical arbitrage beyond asset mispricing. Conventionally, statistical arbitrage is primarily
defined as the temporary pricing mismatches of an asset or related asset with strategies requiring
buying undervalued assets and selling overvalued ones to profit from the anticipated price
convergence. On the contrary, the concept of “statistical arbitrage between models” focuses on
identifying and exploiting the consistent, predictable profit arising from how one model
performs differently from the other based on underlying assumptions and mechanisms,
potentially when turned into a zero -cost strategy. This evolvin g landscape accentuates the
ongoing pursuit of more refined and adaptive hedging methodologies.
Therefore, in this project, we develop and evaluate dynamic hedging strategies using the KSE
30 Index future contracts by employing both econometric and cutting -edge machine learning
frameworks. Specifically, we have used the DCC GARCH framework and extended it by
layering it with a copula function using the Copula GARCH framework to model dynamic
correlations and tail dependencies. Furthermore, we have leveraged the LSTM-CNN and FT-

## Page 19

Net Hybrid models to capture complex, non -linear, and cyclical patterns in Pakistani equity
markets.
1.2 Significance and Motivation

The motivation for this research stems from critical gaps in the existing literature and the
dynamic evolution of financial risk management practices. Firstly, despite the significant role
derivatives play in managing risks in the equity markets, especiall y in the case of developing
markets like Pakistan, specific research in this context is notably limited. Therefore, we focus
our methodology on the KSE 30 index and its future contracts to fill this crucial regional void
and offer practical insights to local investors and institutions.
Secondly, while there is significant research on developing sophisticated econometric models,
their underlying assumptions often inhibit their effectiveness, especially in capturing non-linear
and complex relationships in re al-world data. The shift to machine learning models is a
promising alternative as it models intricate relationships in data without fitting the data to a lot
of the model's underlying assumptions. However, there is a need to study these methodologies
together to gauge the relative strengths and weaknesses of the models. There is a distinct
literature gap in this aspect, and our research focuses on this analytical gap by providing
empirical evidence on the relative strengths and weaknesses of each framework.
Thirdly, this study is inspired by the pioneering work of (François, Gauthier, Godin, &
Mendoza, 2025) and (Horikawa & Nakagawa, 2024) , which introduces the concept of
"statistical arbitrage between models." They test the proposition that the difference in the
hedging performance between models can lead to exploitable arbitrage opportunities. This is a
unique perspective that extends the traditional concept of statistical arbitrage and presen ts a
novel research area. By investigating this phenomenon, we layer our objective with an added
perspective on our models’ efficiency and market equilibrium.
1.3 Aims and Objectives

In this research, we aim to:
(i) formulate hedging strategies for a KSE 30 index portfolio using the KSE 30 futures
contract through both sophisticated econometric models and advanced deep learning

## Page 20

models to dynamically minimize risk in the equity futures market,
(ii) utilize key comparison m etrics to evaluate and contrast the effectiveness of hedging
strategies derived from both econometric and machine learning models in minimizing
risk, and
(iii) conduct an in -depth investigation into potential statistical arbitrage trading
opportunities by all t he models and analyze which model presents the best
opportunities.

1.4 Methodology

To incorporate the time-varying nature of optimal hedge ratios, we have focused on models that
dynamically compute the hedge ratios. In this research, we first use the (Engle R. , 2002) DCC-
GARCH framework. This framework models each return series through an ARMA –eGARCH
margin and updates the conditional correlation matrix, effectively capturing the time -varying
dynamics of covariance. Acknowledging that linear correlation might underest imate joint
extreme movements, we further enhance the DCC -GARCH model with a student -t copula,
tailored to the probability integral transformations of the standardized residuals. This adjustment
allows for symmetric tail dependence and improves hedging eff ectiveness during periods of
market stress (Lai Y. -S., 2018). Concurrently, we also evaluate machine learning models that
can capture intricate nonlinear patterns and long -term dependencies. The LSTM –CNN hybrid
model integrates convolutional layers to extract local temporal features from returns and futures
spreads, followed by Long Short -Term Memory (LSTM) layers that model the dynamics of
sequential hedge ratios. Variants of this hybrid model have demonstrated ability to learn optimal
hedging strategies directly from market data, efficiently adapting to transaction costs and market
frictions in contexts involving illiquid assets (Wang et al., 2024) (Zhang & Huang, 2021). The
innovative FT -Net Hybrid model combines Fourier Transform -based feature modules wi th
temporal convolutional blocks. The spectral modules are designed to capture cyclical market
patterns and non-stationarities that may be overlooked by networks operating solely in the time
domain. Meanwhile, the convolutional blocks, along with a final recurrent or transformer layer,
focus on learning dynamic relationships within the transformed data space. This hybrid
approach enhances recent deep-hedging frameworks by incorporating explicit spectral analysis
components.
Furthermore, we evaluate the hedg ing effectiveness of our econometric and deep learning

## Page 21

models through a total of nine performance metrics which include variance reduction
percentages, Root Mean Square Errors (RMSE), average hedged returns, Sharpe ratios,
directional accuracy, Value at Ri sk (VaR) reduction, Conditional Value at Risk (CvaR)
reduction, mean absolute deviation, and time complexity as used in prior research. To assess
which model performs the best given all these criteria, we use the Fuzzy TOPSIS multicriteria
decision-making technique. Furthermore, to gauge how well a model performs, we analyze the
hedged portfolio’s Profit and Loss series for statistical arbitrage opportunities.
1.5 Contributions of the Study
This study makes several substantial and novel contributions to the fields of financial
engineering and quantitative risk management, particularly within the context of emerging
markets like Pakistan. The most significant contribution lies in the pioneering application of
advanced machine learning models, along with system atically designed statistical arbitrage
trading strategies, directly to equity futures contracts. While the existing global literature on
dynamic hedging and statistical arbitrage has predominantly centered on options markets, this
research extends these i nnovative methodologies to the relatively underexplored domain of
index futures. Hence, it addresses a critical gap in both practical implementation and academic
coverage.
Moreover, this work stands out for its specific focus on the KSE 30 Index, and its futures traded
on the Pakistan Stock Exchange. There is a notable scarcity of quantitative research and deep
learning applications targeting the Pakistani capital markets within existing academic literature.
Through empirical evaluati on, contrasting sophist icated econometric models (such as DCC -
GARCH with Student -t copula) and cutting -edge deep learning models (notably the LSTM –
CNN hybrid and FT-Net Hybrid), this study not only introduces methodological innovation but
also serves as a foundational reference for subsequent research in the region. The research
demonstrates how deep learning models can efficiently extract complex, nonlinear, and time -
varying patterns in hedge ratios, which are often missed by traditional techniques.
A further methodological cont ribution is the integration of Fourier Transform -based spectral
analysis into the deep learning pipeline through the FT -Net Hybrid model. This approach
enhances the ability to capture cyclical and non-stationary characteristics in financial time series
data features particularly relevant to the volatility and market structure of emerging economies.

## Page 22

Additionally, this research incorporates a multi-criteria decision-making framework based on
fuzzy TOPSIS, allowing for a transparent comparison across multiple dimensions of hedge
effectiveness, predictive accuracy, profitability, and computational efficiency.
The beneficiaries of this project include investors such as institutional portfolio managers and
professional traders operating in the Pakistani or similar emerging markets to benefit from
enhanced risk management tools and empirically validated trading strategies that adapt to
market dynamics and exploit inefficiencies. Second, academic researchers gain a region-specific
empirical foundation and a set of advanced modeling frameworks that can be extended to further
research on financial innovation in underexplored markets.
1.6 Report Outline
This thesis is structured i nto seven chapters, each addressing a critical component of the
research. Chapter 1 introduces the study, outlining the background, significance, research
objectives, and the methodology employed. Chapter 2 offers a comprehensive review of relevant
literature and is divided into two sections: the first presents a bibliometric analysis of existing
research, while the second provides a descriptive assessment of the academic landscape. Chapter
3 details the processes of data collection and preprocessing, inclu ding the sources, selection
criteria, and preparation techniques applied to the dataset. Chapter 4 elaborates on the research
methodology, describing the theoretical framework, model architecture, and analytical
techniques utilized throughout the study. Chapter 5 presents and discusses the results of model
evaluation, including performance metrics and comparative analyses. Chapter 6 explores the
statistical arbitrage trading opportunities uncovered by each model and discusses their practical
implications. Furthermore, Chapter 7 explores the integration of macroeconomic variables into
our highest-ranked model, as determined by the fuzzy TOPSIS decision-making criteria. Finally,
Chapter 8 concludes the thesis by summarizing the key findings, highlighting the contributions
of the research, and suggesting potential directions for future work.

## Page 23

2 Literature Review

2.1 Introduction

Hedging risk exposure in equity markets using derivatives, particularly stock index futures, is
a fundamental concep t in financial risk management. The development of an effective and
robust hedging strategy is predicated on a comprehensive understanding of the theoretical
grounds and practical advancements regarding the hedging framework. In consideration of the
extreme importance of a thorough understanding, we contextualize our research with existing
literature. This chapter presents a systematic literature review of the existing literature on the
development and assessment of hedging strategies in equity futures markets.
To effectively traverse the diverse and rapidly evolving research landscape, we have divided
our literature review into two mutually supportive sections. The first section is devoted to a
bibliometric analysis of the literature mapping the major research streams, influential authors,
publication trends, and thematic concentrations within the broader domain of equity market
hedging and statistical arbitrage. This approach leads to an in-depth view of scholarly activity
and key research streams and cont ributions. Expanding on these insights further, the second
section delivers a systematic qualitative review, delving deeper into methodologies, findings,
and limitations of relevant studies and research streams.
The primary objective of structuring the literature review in two sections is to achieve a refined
understanding of the research landscape of this project. This chapter outlines the development
of research on hedging strategies, especially those using advanced econometric and machine
learning models, clarifying the bases of our study. Additionally, it aims to pinpoint essential
research gaps and methodological flaws that our thesis intends to tackle, highlighting the
importance and uniqueness of our research contributions.

2.2 Bibliometric Analysis

The bibliometric analysis is conducted on a sample of 588 articles imported from the Web of
Science (WoS) database. The Clarivate’s Web of Science database is one of the oldest and most
reputable databases of research papers and citations. An article search technique was formulated
based on author keywords to gather the most relevant sample of articles. To refine the search, a

## Page 24

keyword search criterion was defined which included the following keywords: “futures
contracts”, “hedging effectiveness”, “minimum v ariance hedging”, “optimal hedge ratio”,
“stock index futures”. The publication years were limited to 2014 to 2024. Furthermore, the
most relevant WoS core collection subject categories were chosen, which were “Business
Finance”, “Mathematics Applied”, “St atistics Probability”, “Mathematics”, and “Economics”.
The meta literature review was conducted using VoS Viewer, Biblioshiny R Package, and
Microsoft Power BI.
2.2.1 Keyword Analysis
The result of the keyword analysis performed using VoS Viewer is presented in Figure 1. The
figure shows a cartography map that highlights clusters of different research streams within the
broader literature of optimal dynamic hedging using stock index futures. These clusters are
identified by VoS Viewer based on the close associat ion of keywords depicting thematic
groupings within the literature and are represented by different colors. The lines interconnecting
the clusters amongst themselves depict the strength and frequency of the co -occurrence of
various keywords within the same publications to capture the interdisciplinary and
interconnected nature of research. The VoS Viewer software has a scale of co -occurrence that
sets a lower limit to the number of co -occurrences of keywords. For this analysis, it was
configured to a minimum of five co-occurrences of a keyword. This results in five main clusters
in the cartography map, which represent five major research streams in the literature.
The red cluster is the most prominent and densely populated, with keywords “volatility”,
“model”, “risk management”, and “hedging effectiveness”. This cluster represents the literature
focusing on volatility modelling, assessment of hedge effectiveness, and development of
frameworks for risk mitigation in financial markets. The second most prominent cluster is the
blue one, which majorly shows keywords with the theme of “hedge effectiveness”; however, it
includes other related topics such as “volatility spillovers”, “returns”, and “uncertainty”. The
third cluster in green is centered on the research stream “stock index futures”. It represents the
part of literature that focuses on price discovery, information transformation, lead -lag
relationships, and the econometric modelling of future contracts.
Furthermore, we have two other clusters in purple an d yellow. The purple cluster is
characterized by keywords such as “volatility spillover”, “conditional heteroskedasticity”, and
“structural breaks”, emphasizing the body of literature on advanced time series modelling and

## Page 25

structural changes in financial ma rkets. Meanwhile, the yellow cluster is dominated by
keywords such as “gold”, “bitcoin”, “diversification”, and “COVID -19”. This cluster
encapsulates the body of literature that emphasizes alternative investments and their impact on
periods of heightened volatility.
The cartography results on an overall level depicted the major research streams in our literature
sample and highlighted the interconnected nature of the themes.
2.2.2 Yearly Research Trends and Publication Volume

Figure 2 below shows the yearly analysis of publication volume and citation trends of the
literature from 2014 to 2024. The graph presents two key metrics: the number of papers
published each year, represen ted by blue bars, and the cumulative number of times these
works have been cited across all databases, depicted by the solid line.
The overall trend in the publication volume and the times cited score shows an increasing
trend throughout the years. However, a notable increase in the volume of literature is evident
from the graph after 2018, which reflects a positive development in the academic interest
and engagement in research relevant to our project’s themes. The growth in research is
particularly high after 2020, corresponding with increasing market uncertainty and the rapid
development of new computational methodologies in finance.
The trend in citation counts simultaneously exhibits a positive growth pattern, although it is
Figure 1: Cartography Results obtained from VoS Viewer

## Page 26

somewhat more variable. A sharp increase in citations from 2018 onwards suggests that
recent research has garnered significant attention and has played an instrumental role in
influencing the current academic and research landscape. The steady upward movem ent in
both the volume of published papers and citation scores in recent years is not only indicative
of the vitality of this research field but also of its growing relevance to financial practitioners
and policymakers.
2.2.3 Most Influential Countries
Figure 3 presents the country-wise publication analysis. The map shows every country in a
distinctive color and its corresponding publication volume through the size of the marker.
This figure visualizes the regional distribution of literature and helps id entify the leading
countries in terms of publication in the context of the literature.
The map reveals a greater concentration of literature in developed markets such as China
and the United States of America (USA). Moreover, other significant contributors to the risk
management through stock index futures literature include the United Kingdom (UK), South
Korea, Australia, and Germany.
In the South and Southeastern regions, India, Taiwan, and Vietnam dominate most of the
publication volume. Moreover, there is some prominent research activity in France, Canada,
Turkey, and Saudi Arabia which emphasizes the widespread growth of the literature in
various parts of the world. There are smaller volumes of research in countries such as
Figure 2: Y early Research Trends

## Page 27

Pakistan, Greece, Oman, Tun isia, and New Zealand; however, despite the lower volumes,
they contribute important regional perspectives in the literature.
2.2.4 Three Factor Analysis
Figure 4 shows the three-factor analysis diagram. The diagram combines three factors crucial
in understanding the dynamics of the literature: the principal research keywords, most
frequently occurring terms extracted from articles, abstracts, and titles, and the top 10
countries in terms of volume of research.
Previously, we viewed these factors separately; now we put them together to gauge how
different countries focus on particular aspects of the broader research landscape. For
instance, the linkage between volatility -focused res earch and the Chinese, UK, and US
researchers shows that these markets are central to research in quantitative risk modelling.
Moreover, countries like Tunisia and Vietnam contribute more research on themes such as
market efficiency or commodity risk.
In the overall aspect, we can observe from the three-factor analysis the dynamic and diverse
interplay between research themes and their global distribution.
Figure 3: Number of publications in every country

## Page 28

2.3 Systematic Review of Key Research Streams

The conceptual framework of this research is inspired from the work authored by (François,
Gauthier, Godin, & Mendoza, 2025) which is based on option contracts that test the
proposition made by (Horikawa & Nakagawa, 2024) . This proposition states that under a
complete market admitting statistical arbitrage, the difference between the hedging position
provided by deep hedging and that of the replicating portfolio, i s a statistical arbitrage.
(François, Gauthier, Godin, & Mendoza, 2025) tests this proposition using GARCH based
models for delta hedging and deep hedging. Therefore, in this paper, we test the same
proposition in the context of hedging with future contracts, where, instead of delta hedging,
we use minimum variance hedging, which is a popular choice in research. To estimate
minimum variance hedge ratios (MVHRs), research has used multiple econometric models
such as traditional Ordinary Least Squares (OLS) and Error Correction Models (ECM), only
as in (Lin, 2002) or along with sophisticated models like Autoregressive Conditional
Heteroscedasticity (ARCH) and Generalized Conditional Autoregressive Heteroscedasticity
(GARCH) models and their variants (eGARCH, mGARCH, DCC, ADCC, GO GARCH,
GARCH COPULA, BEKK GARCH, etc.) (Alvarez-Diez & Gonzalez, 2006) (Bodla &
Jindal, 2006) (Floros & Vougas, 2007) (Wang & Zhang, 2019).
The conventional strategies for estimating MVHRs, such as the OLS method, estimate
constant MVHRs and are criticized for their inability to account for cointegration, which
Figure 4: Three Factor Analysis

## Page 29

results in under-hedging. To incorporate cointegration, ECM methods were introduced; they
capture short-term dynamics and the long -term equilibrium between the spot and futures
returns, resulting in better performance. However, like the OLS technique, they also assume
that the relationship between spot and futures is timeless. It is evident from various studies
(Koutmos & Tucker, 1996) that there is a strong causal relationship between the spot and
futures return that is time va rying; therefore, dynamic hedge ratios are more effective for
risk management. (Baillie & Myers, 1991) To compute dynamic hedge ratios, various
extensions of the GARCH method (Bollerslev T. , 1986) are used in prior research. Some
prominent variants used in research were BEKK GARCH (Baba, Engle , Kraft, & Kroner,
1990) which is computationally intensive as it allows for complex interaction between asset
volatilities, CCC GARCH (Bollerslev T. , 1990) is simpler than BEKK GARCH. It assumes
that although the variances are time varying, the conditional correlation between assets
remains constant. DCC GARCH is an extension of this model by (Engle R. , 2002) which
allows conditional correlations to vary with time. Furthermore, Copula GARCH models as
in (Patton A. J., 2006), (Lai Y. S., 2018), (Lai, Chen, & Gerlach, 2009) estimate the marginal
distribution using univariate GARCH models and then link them to a copula function to
capture asymmetric and non -linear dependencies. These GARCH -type models, although
outperform static mode ls, might overestimate the persistence in volatilities since regime
switches and unexpected changes in variance are often ignored. (Qu, Wang, Zhang, & Sun,
2019)
The recent developments in deep learning have led the estimation of minimum variance
hedge ratios (MVHRs) in stock index futures markets to a new paradigm which overcomes
the model assumption and specification challenges in traditional and sophisticated models
by capturing complex relationships between time series thro ugh the power of neural
networks. Many researchers have concluded that deep learning-based hedging strategies can
understand complex, nonlinear features in high frequency data significantly better than
conventional approaches while yielding higher economic benefits (Hu & Ni, 2024) . The
literature is especially dominated with Long Short-Term Memory (LSTM) architectures that
have demonstrated superior performance in capturing time -varying dynamics in financial
markets, outperforming static OLS and parametric GARCH models in both simu lated and
real-time data (Adams, Asemota, & Ibrahim, 2024) . Hybrid models that integrate

## Page 30

Convolutional Neural Networks (CNNs) with LSTMs or more novel Frequency -Temporal
(FT-Net) Hybrid architectures have greater potential to e nhance hedge effectiveness by
combining local feature extraction with long-term dependency modeling. It is evident from
research that these deep and hybrid models produce lower hedged return variance and higher
risk adjusted performance metrics across dive rse asset classes in contrast with the hedge
produced through dynamic mathematical models.
The theoretical basis of “deep hedging” was initialized in (Beuhler, Gonon, & Wood,
2019).In this research, it was highlighted that the deep learning framework does not require
conventional finance basis such as the Greeks in finance or underlying assumptions
regarding price behavior, however, the neural networks can be trained to directly minimize
risk. Later studies including (François, Gauthier, Godin, & Mendoza, 2025) and (Hu & Ni,
2024) build upon this methodology and formulated a Deep Learning based Financial
Hedging (DL-HE) strategy. This strategy was first implemented on commodity futures to
learn complex patterns through nonlinear feature extraction for inventory and market risk
management. The annualized gains reported from the use of this strategy were greater than
traditional hedges by 1.2 million Chinese Yen (CNY). Subsequently, other researches
explored implications of these similar deep learning models in illiquid markets. However, in
illiquid markets LSTM based models outperform significantly than prior buy and hold
strategies judged on the basis of improved Sharpe ratios and draw down control when
forecasting futures prices for hedging purposes. Moreover, comprehensive comparis ons of
traditional versus deep MVHRs exhibited that deep models not only reduce hedged‐position
variance but also adapt more swiftly to regime shifts and volatility spikes than static OLS or
ECM methodologies.
(Knoll, 2023) illustrate a “Deep Mean-Variance Hedging” approach wherein a quadratic risk
objective is used to train an LSTM hedging agent, which then exhibits superior hedging
performance across both exponential Lévy and stochastic volatility models in simulation
experiments. This work is further extended by (Agram, Oksendal, & Rems , 2024) , who
propose a self -supervised learning approach for quadratic hedging in incomplete jump
markets. The self -supervised learning network learns and adjusts f or discontinuities and
jumps in data, thereby performing better than GARCH -based hedges in synthetic and
empirical tests.

## Page 31

Furthermore, hybrid deep learning models that layer two types of networks, CNN and
LSTM, are being used in research due to their ability to capture spatial and temporal patterns
in futures price series. Their application on EU Emissions Trading Scheme (EU FTS) futures
realized up to a 43% reduction in mean absolute percentage error (MAPE) compared to pure
LSTMS or Vector Autoregressive (VAR) models, leading to more efficient intraday trading
and hedging strategies. Hybrid models such as CNN-LSTM and CNN-LSTM-AM proved to
outperform single model baselines as they are able to focus on short term, sudden price
changes, long term trends and patterns along with attention mechanisms to help the model
concentrate on the most important data. This proven success of hybrid models led to the
emergence of the FT -NET Hybrid architectures in 2024. These models consist of Fourier
transformers that help detect cyclical patterns in the market and respond to sudden shifts in
market behavior. Although FT -Net is a novel approach that has yet to be used widely in
research some preliminary works have highlighted its potential to reduce hedged return
variance by an additional 5-8% comparative to CNN-LSTM hybrids in the context of stock
index futures.

## Page 32

3 Data Collection and Processing

This study hedges the risk of a spot equity portfolio of the KSE 30 index by conventional
and deep learning hedging techniques using the KSE 30 stock index futures contract. It uses
historical data of stock index and stock index futures contracts from January 2019 to June
2024. The frequency of the data recorded is daily to ensure accuracy. All data was collected
from the PSX Data Portal (Daily Downloads page).
The data obtained includes the following fields:
• Index cash price: The current or spot price of the underlying index on the date of
report.
• Interest/ Financing Rate: This is the interest rate used for calculating the fair value of
the futures contract. It reflects the cost of carrying or financing the position.
• Index dividend yield: The expected annual dividend yield. of the index. This is used
to adjust the futures price for the expected dividends that would be paid out over the
life of the contract.
• Days to expire: The number of days remaining until the futures contract expires
• Indicative fair value: The Exchange calculates a reference price as fair value of the
SIF Contract on which all outstanding SIF Contracts are settle d. The fair value is a
function of cash or underlying index value plus financing charges
• (r, determined as a function of KIBOR rates) less any dividends (d) that would accrue
with the purchase and carry of all Index constituent until the final settlement date.
Equation 1
𝐼𝑛𝑑𝑖𝑐𝑎𝑡𝑖𝑣𝑒 𝑓𝑎𝑖𝑟 𝑣𝑎𝑙𝑢𝑒= 𝑈𝑛𝑑𝑒𝑟𝑙𝑦𝑖𝑛𝑔 𝐼𝑛𝑑𝑒𝑥∗ (1 + 𝑟( 𝑥
365)) − 𝑑
The Pakistan Stock Exchange offers three stock index futures contracts from which we
selected the KSE 30 Index Future contracts. The futures contracts have a maturity of 3
months. We have selected the contracts with the closest days to expire which will re quire
rolling the contract every month for continuity.

## Page 33

3.1 Data Processing
After the data collection, the next step was to process the data, which included interpolating
missing data. Our data had missing Indicative fair value of contracts for multiple days . To
estimate the indicative fair value of those contracts, we used the equation provided in the
PSX rule book (equation 1). However, a major problem was the absence of interest/
financing rate data for multiple days. Therefore, the random forest model was used to
interpolate missing values of the interest \financing rate. Furthermore, to ensure continuity
and integrity of the futures price data as the nearest to expiry contracts were rolled over, we
performed mean back adjustments.
3.1.1 Filling the Missing Values
One of the critical challenges encountered was the presence of missing data in our data set.
Missing values are a pervasive issue in the financial time series, often arising due to data
recording errors, market holidays, or irregular reporting. In our c ase, the most significant
gaps were observed in the daily futures data. This posed a substantial barrier because, as
stipulated by the Pakistan Stock Exchange (PSX) rulebook, the indicative fair value of the
KSE 30 futures contract is derived from a precise mathematical formula (refer to formula 1).
As mentioned earlier too, the Underlying Index is the spot value of the KSE -30 index on a
particular date, 𝑟 represents the annualized interest or financing rate, 𝑥 denotes the number
of days remaining until the futures contract expires, and 𝑑 is the value of the expected
dividends over the contract period. The historical data of price of underlying asset and
dividend rate were readily available on PSX website, and the number of days to expiry was
directly computable from the date fields. The only true ambiguity was the missing financing
or interest rate, which is pivotal for pricing futures accurately and, by extension, for all
subsequent econometric and machine learning analysis that depend on continuous, hi gh-
quality price data.
To address this, we implemented a robust data imputation approach. Our first step was to
gather the required auxiliary data from authoritative sources, particularly the 1 -month
KIBOR rates from the State Bank of Pakistan (SBP), since the financing rate, as defined in
the NCCPL rulebook, is fundamentally a function of this short -term interbank offered rate.
This step ensured that we had a reliable explanatory variable for predicting the missing

## Page 34

interest rates.
Given the non-linear and potentially complex relationship between the 1-month KIBOR and
the futures contract’s financing rate, we opted for a machine learning approach rather than a
simple linear interpolation. We selected the Random Forest Regressor, an ensemble method
capable of capturing intricate, non -linear dependencies without requiring explicit model
specification or parametric assumptions. The Random Forest Regressor builds a number of
decision trees during training. The prediction of each tree is averaged to get the final
prediction output, which provides strong predictive accuracy and robustness against
overfitting. Figure 5 below shows the architecture and pipeline used.
The implementation involved splitting the available data into two subsets: one containing
rows where the interest rate was known (used for training), and one where it was missing
(for prediction). Using the 1-month KIBOR as the sole predictor, the Random Forest model
was repeatedly trained (across 1000 different random seeds) to maximize out -of-sample
performance, assessed by metrics such as R-Squared (𝑅2), Mean Squared Error (MSE), and
Pearson correlation coefficient between actual and predicted in terest rates. This iterative
process ensured that we identified the optimal model configuration, providing the highest
explanatory power and the lowest prediction error. Ultimately, the best -performing model
Figure 5: Architecture and Pipeline of the Random Forest Regressor

## Page 35

achieved an exceptionally high 𝑅2 value of 0.9092 and a remarkably low MSE of 0.0003, as
highlighted in the output. This indicates that the model explained more than 90% of the
variance in the observed interest rate values, instilling high confidence in the fidelity of the
imputed values.
Once the missing interest rates were filled using the trained model, we were able to compute
the indicative fair values for all missing entries using the aforementioned PSX formula. This
seamless integration of machine learning -driven imputation with domain -specific financial
formulas ensured the consistency, accuracy, and completeness of our futures dataset.
Consequently, the resultant data was rendered suitable for rigorous econometric modeling
and deep learning experiments, preserving the integrity of our research findings and allowing
us to confidently pursue the subsequent stages of dynamic hedging and arbitrage analysis.
3.1.2 Adjustments of the Future Contracts
In the context of our research, we encountered a significant issue relating to the continuity
and integrity of futures contract price data. The fundamental problem arose from the nature
of futures contracts themselves: each contract has a defined expiration date, typically at the
end of the month, after which trading transitions or "rolls over" to a new contract with a
future expiration. Theoretically, as a futures contract approaches its expiry, its price is
expected to converge with the spot price of the underlying asset, a principle rooted in the
cost-of-carry model. However, in practice, and particularly in th e dataset we obtained, the
way continuous futures price series are constructed leads to artificial "jumps" at rollover
points. These jumps occur because, on the final trading day, the expiring contract is priced
very close to the spot (due to convergence), but the next available contract, being further
from its own expiry, incorporates additional risk premia, interest rate differentials, and
market expectations, resulting in a different price level. When the dataset is stitched together
contract by contract in a continuous series, these price jumps at each monthly expiry can
introduce significant distortions, disrupting any time series analysis, volatility modeling, or
hedge ratio estimation.
Addressing this discontinuity is crucial for the validity of any e mpirical analysis involving
futures prices. Several methods exist to resolve these jumps, such as forward adjustments,
calendar spread methods, and the more popular back-adjustment technique. In our study, we
selected the mean-back adjustment method due to its conceptual simplicity and effectiveness

## Page 36

in producing a smooth, artifact -free price series, which is particularly suitable for
econometric and deep learning models that are sensitive to structural breaks or artificial
volatility spikes.
The mean -back adjustment processes we applied can be described as, for each contract
rollover, we first identified the specific day on which the expiring contract ended, and the
new contract began trading as the active series. Let us denote the price of th e expiring
contract on its last day as 𝑃𝑡𝑟𝑜𝑙𝑙
(exp) and the price of the new contract on the same day as 𝑃𝑡𝑟𝑜𝑙𝑙
(new).
To avoid an abrupt shift, we computed the mean of these two prices.
Equation 2
𝑀𝑟𝑜𝑙𝑙 =
𝑃𝑡𝑟𝑜𝑙𝑙
(exp)+𝑃𝑡𝑟𝑜𝑙𝑙
(new)
Next, we determined the difference between this mean and the expiring contract's price on
the rollover day.
Equation 3
𝐷𝑟𝑜𝑙𝑙 = 𝑃𝑡𝑟𝑜𝑙𝑙
(exp) − 𝑀𝑟𝑜𝑙𝑙
This difference, which can be posi tive or negative depending on the direction of the jump,
quantifies the artificial gap that needs to be adjusted.
The key step in the mean -back adjustment is to uniformly apply this difference 𝐷𝑟𝑜𝑙𝑙 as an
adjustment factor to all price points within t he expiring contract's life, i.e., all daily prices
for that contract month. If 𝑃𝑡𝑟𝑜𝑙𝑙
(exp) is the price of the expiring contract at time 𝑡 within its
active month, the adjusted price becomes,
Equation 4
𝑃𝑡𝑟𝑜𝑙𝑙
(exp,adj) = 𝑃𝑡
(exp) ± 𝐷𝑟𝑜𝑙𝑙
This means that each daily price in that contract month is shifted up or down by exactly the
same amount, ensuring the price at the rollover aligns seamlessly with the start of the new
contract, while preserving the intra -month price dynamics and volatility. The same
procedure is repeated for every rollover month, systematically removing jumps at all points
of contract expiration throughout the historical dataset.

## Page 37

Mathematically, if there are 𝑛 contracts spanning the full data period, the adjustment process
for each contract 𝑖 at rollover point 𝑡𝑟𝑜𝑙𝑙,𝑖 can be generalized as follows:
1. Compute the mean at rollover (refer equation 2)
2. Calculate the difference (refer equation 3)
3. Adjust all prices in contract 𝑖 (from start 𝑡𝑠𝑡𝑎𝑟𝑡,𝑖 to 𝑡𝑟𝑜𝑙𝑙,𝑖 ) (refer equation 4)
Through this process, we obtained a continuous time series of futures prices free from any
spurious jumps or artificial volatility. Such fluctuations would otherwise distort the
reliability of any subs equent econometric modelling or deep learning -based hedge ratio
estimation.
This adjustment means that both the statistics of returns and detection of structural market
dynamics reflect the real market and are not just a result of how contracts are rolled over.
Thus, data processing of futures price through a mean -back adjustment gave us a rigorous
and transparent methodology to use as a groundwork for all further analyses of our project.

## Page 38

4 Research Methodology
This chapter is divided into three distinct sections in order to outline the methodological
framework adopted in this research systematically.
The first section is dedicated to preliminary data diagnostics and pre -estimation testing. In this
section, we p erform a comprehensive investigation of the time series data characteristics, with
the aim of assessing whether the underlying statistical assumptions required by each modeling
approach are satisfied. Especially focusing on statistical tests for stationarity, autocorrelation, and
heteroscedasticity, which are fundamental prerequisites for time series modelling and are
essential specifically for conventional econometric models. By rigorously interrogating the data
at this initial stage, we ensure that the in tegrity of the modeling process is upheld and that the
results derived later in the study are built on a sound empirical foundation.
The second section gives in -depth insights into the core modeling methodologies employed to
address the research objectives . Particularly, we provide an extensive exposition of the four
principal hedging frameworks utilized: the Dynamic Conditional Correlation Generalized
Autoregressive Conditional Heteroskedasticity (DCC -GARCH) model, the DCC Copula -
GARCH extension, the Long Short-Term Memory Convolutional Neural Network (LSTM –
CNN) hybrid, and the Fourier Transform – Network (FT -Net) Hybrid model. Each of these
models is discussed in detail highlighting its mathematical underpinnings, structural
specifications, and theoretical motivation, as well as the rationale for its inclusion in the context
of dynamic hedging and risk minimization in the equity futures market.
The final section of the chapter is concerned with post -estimation testing and model validation.
In this segment, we outline the statistical tests and performance metrics used to evaluate the
adequacy and robustness of each model. This includes both in -sample and out -of-sample
diagnostic checks and the model results providing an empirical basis for the comparative analysis
and model selection that follows in subsequent chapters.

## Page 39

4.1 Statistical Tests Pre-Model Implementation

4.1.1 Augmented Dickey-Fuller (ADF) Test for Stationarity
The ADF test for stationarity, as given by (Dicky & Fuller , 1979), is a statistical test used
to determine whether a time series has a unit root, indicating that the series is non-stationary.
The test involves estimating an autoregressive model and testing the null hypothesis that the
series contains a unit root, w hich would imply non -stationarity. If the null hypothesis is
rejected, it suggests that the time series is stationary.
We conducted the ADF Test using the R programming language and the tseries package on
the futures contract returns, and KSE 30 returns t ime series. The adf.test() function in R
applies the general regression equation, incorporating both a constant and a linear trend, and
computes the t-statistic to test whether the first -order autoregressive coefficient equals one
(indicating non-stationarity). The number of lags used in the regression is denoted by k,
where the default value is determined using equation 5 below.
Equation 5
𝑡𝑟𝑢𝑛𝑐((𝑙𝑒𝑛𝑔𝑡ℎ(𝑥) − 1) 1/3,
It follows the suggested upper bound for the ARMA(p,q) framework. Therefore, the default
lag value for our time series was 6. The p -values are interpolated from (Banerjee, Dolado,
Galbraith, & Hendry, 1993) . If the computed test statisti c falls outside the table of critical
values, a warning message is generated.
The null hypothesis, which tests for the presence of a unit root (indicating non-stationarity),
for all three series was rejected at both the 5% and 1% confidence levels for all time series.
As the p -values were smaller than 0.01, and R displays the lowest recorded value, all p -
values were presented as 0.01.
4.1.2 Engle’s ARCH Test
The Engle’s ARCH Test introduced in his paper in (Engle R. F., 1982) is conducted by
estimating a regression model, where a set of independent variables explains the dependent
variable and the residuals (𝜀𝑡̂ ) are extracted. The residuals must be checked before fitting the
model because the DCC-GARCH and Copula DCC-GARCH are built to model and account

## Page 40

for the presence of heteroscedasticity in the time series. To assess the presence of conditional
heteroscedasticity, the squared residuals (𝜀𝑡
2̂ ) are used as a proxy for variance. A secondary
regression is then performed, where the squared residuals are regressed on their own lagged
values up to a chosen number of lags (𝑞). The null hypothesis (𝐻0) assumes that there are no
ARCH effects, meaning the coefficients of the lagged squared residuals are jointly equal to
zero. To test this, an LM (Lagrange Multiplier) test statistic is computed as 𝐿𝑀 = 𝑇𝑅2,
where 𝑇 is the sample size and 𝑅2 is the coefficient of determination from the auxiliary
regression. This statistic follows a chi -square (𝜒2 ) distribution with 𝑞 degrees of freedom.
If the LM statistic exceeds the critical chi -square value, the null hypothesis is rejected,
indicating the presence of ARCH effects.
To analyze this dataset, we utilized the ARCHTest() function from the FinTS library in R.
The test was conducted with up to 10 lags. The results indicated that both return series,
namely KSE 30 Futures and the underlying index KSE 30, exhibited ARCH effects. To
further analyze the volatility clustering on our dataset, we used residual and squared residual
plots along with Auto Correlation Function (ACF) plots of residuals and squared residuals,
which showed significance at lag 1 (see figures 6 and 7).

Figure 6: ACF plots of residuals and squared residuals of KSE 30 index return

## Page 41

4.1.3 Modeling Approaches for Hedge Ratio Estimation
This paper focuses on dynamically hedging the risk of a KSE 30 index portfolio using KSE
30 stock index futures. Hedging through future contracts is a widely known method to offset
price risk. The most commonly used framework for hedging through stock index futures is
minimum variance hedging, which targets optimizing the portfolio's total variance. The
minimum variance hedge ratio can be calculated using the framework below:
Let 𝑅𝑠 be the return of an equity portfolio underlying the KSE 30 index to hedge the risk of
this portfolio, we short N KSE 30 future contracts whose return is represented by 𝑅𝑓.
Equation 6 below defines the return of the hedged portfolio.
Equation 6
𝑅𝑣 = 𝑅𝑠 − 𝑁𝑅𝑓
And we define the variance of the portfolio V in equation 7.

Equation 7
𝜎𝑣
2 = 𝜎𝑠
2 + 𝑁2𝜎𝑓
2 − 2𝑁𝜎𝑠,𝑓
Since we aim to find a hedge that reduces the risk to a minimum level, we differentiate
Figure 7: ACF plots of residuals and squared residuals of KSE 30 index futures return

## Page 42

equation 7 with respect to N;
Equation 8
𝜕𝜎𝑣
𝜕𝑁
= −2𝑁𝜎𝑓
2 + 2𝜎𝑠,𝑓
Equating equation 8 to zero, gives us 𝑁∗ (refer equation 9(.
Equation 9
𝑁∗ = 𝜎𝑠𝑓
𝜎𝑓
Here 𝑁∗is the minimum variance hedge ratio, 𝜎𝑠𝑓 is the covariance between spot and future
returns and 𝜎𝑓
2 is the variance of future contracts’ returns. There are various methods used
in prior research to estimate the MVHR. The direct methods, i.e., methods that estimate the
hedge ratio directly, include the Ordinary Least Square (OLS) technique, Regression
Coefficient (RC) technique, Error Correction Models (ECM), etc. However, these direct
methods mostly do not estimate time -varying hedge ratios. Indirect methods include
sophisticated models such as Generalized Autoregressive Conditional Heteroskedasticity
(GARCH) models and their variants, namely, Dynamic Conditional Correlation (DCC)
GARCH, Constant Conditional Correlation (CCC) GARCH, Copula GARCH, and Baba,
Engle, Kraft, and Kroner (BEKK) GARCH, etc.
In this research, however, we used the econometric mode ls DCC eGARCH and Copula
GARCH and the machine learning models Long Short – Term Memory – Convolutional
Neural Network (LSTM – CNN) model and FT-Net Hybrid model to compute time-varying
dynamic hedge ratios.
4.1.4 Dynamic Conditional Correlation (DCC) GARCH
Inspired from (Xu & Li, 2017), which highlighted that for enhanced volatility modelling we
can combine the Exponentially Weighted Moving Average (EWMA) with the GARCH
model. The EWMA models volatility while assigning greater weig ht to recent events and
the GARCH models account for volatility clustering therefore, combined they can portray a
more accurate representation of market dynamics. So, before fitting the DCC GARCH model

## Page 43

on the training data we standardize the returns of the KSE 30 index and its future contracts.
The standardized returns were calculated using the equation 10 below.
Equation 10
𝑍𝑖,𝑡 = 𝑅𝑖,𝑡 − 𝜇
𝜎𝑡

Where, 𝑅𝑖,𝑡 is the raw log return of the asset i, and 𝜇 is the mean of the returns. The standard
deviation of the returns was estimated using the EWMA method. The EWMA method is a
powerful technique for modelling conditional volatility because it captures volatility
clustering without requiring a full GARCH specifications. The EWMA variance is estimated
using the following equation 11:
Equation 11
𝜎𝑖,𝑡
2 = 𝜆𝜎𝑖,𝑡−1
2 + (1 − 𝜆)𝑅𝑖,𝑡−1
Here 𝜎𝑖,𝑡
2 is the variance of the raw log returns of asset i. Since we convert the raw returns of
the training data to standardized returns while forecasting the volatilities of the testing data,
we convert them to standardized forms as well using standard deviation s estimated in the
training data. (see figure 8)
The DCC GARCH model is a multivariate GARCH model proposed by (Engle R. , 2002)
which is a generalization of the (Bollerslev T. , 1990) constant conditional correlation (CCC)
GARCH estimators. The DCC framework is a two-step methodology as follows:
4.1.4.1 Univariate modelling using GARCH
The DCC model proposed by (Engle R. , 2002) requires the use of univariate GARCH
models to estimate the conditional variance of individual assets' returns, specifically the KSE
30 index returns and KSE 30 futures contracts returns in our case. To model the conditional
variances, we used the Exponential GARCH (eGARCH) model introd uced by (Nelson,
1991), which, unlike standard GARCH models, allows for asymmetry in the impact of
positive and negative shocks on volatility. This is an essential feature in financial time series
data, such as the one used in this research, where negative news trends tend to increase
volatility more than positive news of the same magnitude.

## Page 44

This research defines the return process using an ARMA (1,1) model represented by equation
12 below.
Equation 12
𝑍𝑖,𝑡 = 𝜇𝑖 + ∅1𝑍𝑖,𝑡−1 + 𝜃1𝜖𝑖,𝑡−1 + 𝜖𝑖,𝑡
where 𝑍𝑖,𝑡 is the standardized return of asset i, 𝜇𝑠 is the constant mean and 𝜖𝑖,𝑡 is a white
noise error term with zero mean and constant variance. Then we use the eGARCH (2,1)
equation from (Nelson, 1991) (equation 13) to estimate the volatilities of KSE 30 index and
KSE 30 index future contracts returns.
Equation 13
log ℎ𝑖,𝑡 = 𝜔𝑖 + 𝛽𝑖,1𝑙𝑜𝑔ℎ𝑖,𝑡−1 + 𝛽𝑖,2𝑙𝑜𝑔ℎ𝑖,𝑡−2 + ∝𝑖 ( 𝜖𝑖,𝑡−1
√ℎ𝑖,𝑡−1
)
+ 𝛾𝑖 (| 𝜖𝑖,𝑡−1
√ℎ𝑖,𝑡−1
| − 𝐸 | 𝜖𝑖,𝑡−1
√ℎ𝑖,𝑡−1
|)
Where ℎ𝑖,𝑡, is the variance of asset i. The model parameters were estimated using Maximum
Likelihood Estimation (MLE) in R via the ugarchspec() and ugarchfit() functions from the
rugarch package.
4.1.4.2 Dynamic Conditional Correlation Estimation
After estimating conditional volatilities, the second step was to estimate the dynamic
conditional correlation matrices. To do so, the methodology presented in (Engle R. , 2002)
where let 𝜖𝑡 = 𝐻𝑡
1/2𝑧𝑡 be the vector of residuals and 𝑧𝑡 ~ 𝑁(0,1) and 𝐻𝑡 is the conditional
covariance matrix as presented in equation 14.
Equation 14
𝐻𝑡 = 𝐷𝑡𝑅𝑡𝐷𝑡
Where 𝐷𝑡 (equation 15) is a diagonal matrix of time -varying standard deviation from
univariate eGARCH models:

## Page 45

Equation 15
𝐷𝑡 = 𝑑𝑖𝑎𝑔(√ℎ1,𝑡, √ℎ2,𝑡, … , √ℎ3,𝑡 )
The DCC model assumes that the standardized residuals 𝜂𝑡 = 𝐷𝑡
−1𝜖𝑡 have a time-varying
correlation structure governed by a GARCH-type equation (equation 16):
Equation 16
𝑄𝑡 = (1 − 𝑎 − 𝑏)𝑆 + 𝑎𝜂𝑡−1𝜂𝑡−1 + 𝑏𝑄𝑡−1
Where 𝑄𝑡 is the time -varying covariance matrix of standardized residuals, 𝑆 is the
unconditional variance of 𝜂𝑡. The dynamic correlation matrix is then obtained by:
Equation 17
𝑅𝑡 = 𝑑𝑖𝑎𝑔(𝑄𝑡)−1/2𝑄𝑡𝑑𝑖𝑎𝑔(𝑄𝑡)−1/2
The DCC model was estimated using the dccspec() and dccfit() functions from the rmgarch
package in R, with univariate eGARCH models used for the marginal distributions.
4.1.5 Dynamic Copula DCC GARCH Model
Copula-DCC-GARCH models combine (Engle R. , 2002) DCC-GARCH with copula theory,
developed further by (Patton A. J., 2006) , (Jondeau & Rockinger, 2006) , and (Hafner &
Figure 8: Standardized Returns of KSE 30 Index and its Future Contracts

## Page 46

Manner , 2012) . Despite the fact that the DCC -GARCH model presents a comprehensive
approach for modelling time-varying correlation between assets, it relies on the assumption
that the joint distribution of the standardized residuals is multivariate normal. However,
financial time series often exhibit non -linear dependencies, especially in the tails of the
distribution (tail dependence), which the normal distribution may not adequately capture. To
address this limitation, we employ a Copula -based DCC-GARCH approach that allows us
to capture complex dependencies and nonlinear co-movements, by focusing on the collective
and individual behavior of the assets returns. Therefore, to optimize our hedge ratios and
reduce the variance, we combine ARMA-eGARCH margins with DCC correlation structure
and a student -t Copula that adequately captures changing volatilities and tail dependence.
(Demarta & McNeil, 2005) (Hsu, Tseng, & Wang, 2008)
4.1.5.1 Pre Copula Fitting Procedure
As mentioned in the DCC -GARCH methodology; to remove the noise and stabilize
volatility, an Exponentially Weighted Moving A verage (EWMA) filter is applied for
standardized returns in training and testing data. After this, the first step in the Copula-DCC
GARCH framework involves modeling the marginal distributions of each return series to
account for volatility clustering and asymmetry. Similar to the earlier section titled
“Univariate modelling using GARCH”, the KSE 30 spot index and futures returns are
modelled using Exponential GARCH (eGARCH) processes. The residuals 𝜀𝑖,𝑡 from the
eGARCH models are standardized to obtain standardized residuals (see equation 18):
Equation 18
𝑧𝑖,𝑡 =
𝜀𝑖,𝑡
√ℎ𝑖,𝑡

Assuming normality of standardized residuals, we transform them to uniform margins using
the cumulative distribution function (CDF) of the standard normal distribution, represented
in equation 19 below.
Equation 19
𝑢𝑖,𝑡 = Φ(𝑧𝑖,𝑡)

## Page 47

This transformation ensures the marginal uniformity required for copula estimation. The
resulting 𝑢𝑖,𝑡 𝜖 (0, 1) are then used to model the joint distribution of returns. The output of
this step serves as the input for both the DCC estimation and the copula transformation steps
that follow.
The estimation of the dynamic conditional correlation (DCC) model follows the
methodology already outlined in the preceding section, based on the framework introduced
by (Engle R. , 2002) . To avoid repetition, we refer the reader to the earlier section titled
“Dynamic Conditional Correlation Estimation”, where the equations governing the DCC
model (Equations14-17) are presented in detail. These equations define how the conditional
correlation matrix 𝑅𝑡 evolves over time using past standardized residuals and a weighted
moving average structure.
4.1.5.2 Copula Estimation
Once the residuals are transformed into uniform margins, a copula function is used to model
the dependence structure. To flexibly model the joint distribution between the standardized
residuals of spot and futures returns beyond linear correlation, we fit a bivariate Student -t
copula to the standardized residuals. The student’s t copula was selected for its ability to
capture tail dependence, which is crucial for modeling co-movements during extreme market
conditions (Demarta & McNeil, 2005).
Given the fitted marginal models, let equation 20 define the probability integral
transformation of each standardi zed residuals into uniform variables, then the student -t
copula can be defined by its correlation coefficient 𝜌 and degrees of freedom 𝑣 as:
Equation 20
𝐶(𝑢1, 𝑢2; 𝜌, 𝑣) = 𝑡𝑣, 𝑝(𝑡𝑣−1 (𝑢1),𝑡𝑣−1 (𝑢2))
Where, 𝐶(𝑢1, 𝑢2; 𝜌, 𝑣) is the copula function, 𝑡𝑣, 𝑝 is the CDF of the bivariate Student's t -
distribution with 𝑣 degrees of freedom and correlation 𝜌, 𝑡𝑣
−1 is the quantile function of the
univariate Student's t -distribution with 𝑣 degrees of freedom (Alshenawy, 2024) . These
parameters are estimated using Maximum Likelihood Estimation via the fitCopula() function
in R.

## Page 48

According to (Demarta & McNeil, 2005), the density of t-copula for Maximum Likelihood
Estimation, can be estimated as:
Equation 21
𝐶𝑣,𝑃
𝑡 =
𝑓𝑣,𝑃 (𝑡𝑣−1 (𝑢1),……..,𝑡𝑣−1 (𝑢𝑑)
∏ 𝑓𝑣 (𝑡𝑣−1 (𝑢𝑖))𝑑
𝑖=1
,𝑢 𝜖 (0,1)𝑑
Where, 𝑓𝑣,𝑃 is the joint density of a multivariate t -distributed random vector and 𝑓𝑣 is the
density of the univariate standard t-distribution with 𝑣 degrees of freedom.
To simulate the t -copula, firstly we generate a multivariate t -distributed random vector X,
which means a random value that follows a t -distribution, with degrees of freedom 𝑣, a
constant mean everywhere, and a correlation matrix 𝑃. A normal mixture technique
explained by (Demarta & McNeil, 2005) is used to generate the vector X. Then, we apply
the standard t-distribution’s Cumulative Distribution Function (CDF) to each element of X,
so that the values fall between 0 and 1. This way, we obtain our sample from the t -copula.
Secondly, to estimate the density, we first map the sample values back into the original t -
distribution space, using inverse CDF, then apply the multivariate t -distribution’s density
function at those points, and divided by the product of the marginal t -densities. Using the
formula at equation 20, we obtain the density of the t-copula, which is useful for estimation.
It is important to note that the t -copula remains invariant under strictly increasing
transformations of the marginals, ensuring that the dependence structure is purely modelled
by the copula, independent of the marginals themselves.
4.1.6 Long-Short Term Memory – Convolutional Neural Network (LSTM–
CNN) Hybrid Model
The financial time series data is characterized by complex features such as nonlinearities,
volatility cl ustering, regime shifts, and structural breaks. These present significant
challenges to classical econometrics model that assume linear relationships or stationary
properties. To overcome the deficiencies of the previously employed models, we make use
of a hybrid deep learning model which is a combination of the Convolutional Neural
Networks (CNN) and Long Short -Term Memory (LSTM) networks from the Recurrent
Neural Network (RNN) family (Kumar, Rao, & Dhochak, 2025) . The model wa s used to
create a dynamic hedge ratio estimate for the KSE 30 index portfolio.

## Page 49

The rationale for using LSTM-CNN Hybrid is because they complement each other. Using
convolutional filters, CNNs can identify short -lived anomalies like bursts of volatility o r
sudden co-movement in spot and futures prices as they slide over the temporal dimension.
This is particularly important in financial data where a short -term increase or shock to the
price can hinder hedging. On the other hand, LSTMs are proven for modeli ng sequential
dependencies and long -range temporal correlations by letting the network remember or
forget things through gating. Therefore, these characteristics are relevant to market regime,
persistence volatility, and other long-lasting effects that may alter hedge ratios over time.
Through using a combination of these two architectures, the model gains a better
understanding of the market by capturing short-term anomalies and long-term dependencies.
Accordingly, it provides a more sophisticated data -driven alternative to static or linear
hedging schemes. The hybrid form is consistent with equity futures since optimal hedge
ratios are not constant but also change through time depending on the market condition.
Recent evidence show that these hybrid deep l earning models outperform traditional
econometric and pure deep learning ones, especially in environment having nonlinear and
regime shifts behavior.
This approach matches the project’s objective that dynamically minimizes portfolio risk
based on estimates of optimal hedge ratios based on market data. The hedging objective is
included in the model design and loss function to allow the model to predict minimum
variance hedge ratios (MVHR) while minimizing hedged portfolio variance to enhance the
real-world applicability of predictive hedge strategies.
4.1.6.1 Theoretical Foundations
To fully comprehend the working and advantages of the LSTM –CNN hybrid model,
examining the theory behind the constituent architectures is highly imperative.
4.1.6.1.1 Convolutional Neural Networks (CNNs)
CNNs were originally developed for spatial data such as images, but their use in time series
analysis has grown rapidly. A CNN applies convolutional filters that act as localized pattern
detectors over input data. Formally, a convolution operation f or a one -dimensional time
series input 𝑥 and filter 𝑤 of size 𝐾 is defined in equation 22 below:

## Page 50

Equation 22
(𝑡) = (𝑥 ∗ 𝑤)(𝑡) = ∑ 𝑥𝑡+𝑘𝑤𝑘
𝐾−1
𝑘=0

where 𝑡 indexes the time steps. In the context of the financial return series, this operation
allows the model to scan over a sequence of returns, identifying local features such as sudden
spikes, dips, or short-lived correlations between the spot index and its futures contract. CNN
filters learn these patterns during training, automatically adapting to the specific structures
present in the data.
The use of “causal” padding is crucial in this setup. Causal convolutions ensure that the
output at any time step depe nds only on the present and past inputs, preserving the time -
ordering and preventing “future” information leakage, which would invalidate the predictive
framework. This is particularly important for financial applications where causality and
chronological order must be maintained.
CNNs excel at feature extraction because they transform the original high -dimensional raw
data into more compact and informative representations called feature maps. These maps
highlight the presence of local temporal features tha t can be critical for downstream
modeling.
4.1.6.1.2 Long Short-Term Memory Networks (LSTMs)
Recurrent Neural Networks (RNNs) are natural candidates for sequential data modeling
because of their inherent temporal feedback loops, but standard RNNs suffer from vanishing
or exploding gradients, limiting their ability to capture long -term dependencies . LSTMs,
introduced by (Hochreiter & Schmidhuber, 1997) , overcome these limitations by using a
gated memory cell architecture that selectively retains relevant information over time.
An LSTM cell contains several gates, i.e. input, forget, and output gates, that regulate the
flow of information. Mathematically, for an input vector 𝑥𝑡, previous hidden state ℎ𝑡−1, and
previous cell state 𝑐𝑡−1, the operations are:
Equation 23
𝑖𝑡 = 𝜎(𝑊𝑖𝑥𝑡 + 𝑈𝑖ℎ𝑡−1 + 𝑏𝑖)

## Page 51

Equation 24
𝑓𝑡 = 𝜎(𝑊𝑓𝑥𝑡 + 𝑈𝑓ℎ𝑡−1 + 𝑏𝑓)
Equation 25
𝑜𝑡 = 𝜎(𝑊𝑜𝑥𝑡 + 𝑈𝑜ℎ𝑡−1 + 𝑏𝑜)
Equation 26
𝑐̃𝑡 = tanh(𝑊𝑐𝑥𝑡 + 𝑈𝑐ℎ𝑡−1 + 𝑏𝑐)
Equation 27
𝑐𝑡 = 𝑓𝑡 ⨀𝑐𝑡−1 + 𝑖𝑡⨀ 𝑐̃𝑡
Equation 28
ℎ𝑡 = 𝑜𝑡 ⨀ 𝑡𝑎𝑛ℎ(𝑐𝑡)
Here, 𝜎 is the sigmoid function that constrains gate activations between 0 and 1, effectively
deciding how much information passes through. The memory cell 𝑐𝑡 is updated by forgetting
part of the old cell state and adding new candidate information modulated by the input gate.
The hidden state ℎ𝑡 is computed as a gated version of the cell state, controlling exposure to
the next layer or output.
LSTMs are highly effective for capturing volatility clustering and long -range temporal
dependencies in financial return s, which are crucial for accurate dynamic hedge ratio
prediction.
4.1.6.1.3 Hybridizing LSTM and CNN
The LSTM and CNN architectures complementing each other. CNNs are powerful local
feature extractors that can detect short, but informative temporal patterns within a moving
window. LSTMs, on the other hand, are used to capture dependencies that take place over
many steps in time, modeling the structure of the whole sequence.
When applying CNN in the beginning, it reduces the input dimensionality by compressing
the raw returns into meaningful local features, which are then processed by LSTM to learn

## Page 52

deeper sequential relationships and long-term interdependencies. This lessens the difficulty
the LSTM has to deal with, improving training efficiency and generalization.
In hedging, this translates to the ability to respond both to immediate market shocks and to
adjust for longer-term regime changes, volatility persistence, and nonlinearities, which are
paramount in minimizing portfolio risk in equity futures markets.
4.1.6.2 Model Architecture and Implementation
The LSTM–CNN hybrid model is implemented using the TensorFlow Keras framework,
with design choices informed by both statistical diagnostics and domain knowledge in
financial time series modeling.
4.1.6.2.1 Data Input and Feature Engineering
The raw dataset contains daily spot and futures prices for the KSE 30 index. These prices
are converted into continuously compounded logarithmic returns for both the spot and
futures series as shown in equation 29:
Equation 29
𝑅𝑆,𝑡 = ln(
𝑃𝑆,𝑡
𝑃𝑆,𝑡−1
) , 𝑅𝐹,𝑡 = ln(
𝑃𝐹,𝑡
𝑃𝐹,𝑡−1
)
where 𝑃𝑆,𝑡 and 𝑃𝐹,𝑡 are the spot and futures prices at time 𝑡.
Figure 9 presents the autocorrelation functions (ACF) for the daily returns of the KSE 30 index
and its corresponding futures contract, revealing important characteristics of short -term and
Figure 9: ACF and PACF of Index and Future Returns

## Page 53

long-term dependencies in these financial time series. Notably, both plots show a significant
spike at the first lag, which stands above the confidence interval, indicating the presence of
short-term autocorrelation at lag 1 for both the index and futures returns. This suggests that the
return for a given day is not entirely ind ependent of the previous day's return, likely due to
factors such as market microstructure effects, delayed information flow, or transient
momentum and reversal patterns commonly observed in real-world markets. Beyond this initial
lag, however, all subsequ ent autocorrelation coefficients fall within the bounds of statistical
insignificance, confirming that neither series displays persistent linear dependencies at longer
horizons. This pattern of strong first-lag autocorrelation followed by rapid decay validates the
modeling approach adopted in this study, specifically the inclusion of both contemporaneous
and one-period lagged returns as input features in the LSTM–CNN architecture.
Hence, to enrich the temporal context, the first lag of each return series is included as features:
𝑅𝑆,𝑡−1, 𝑅𝐹,𝑡−1
Thus, at each time 𝑡, the input features include contemporaneous and lagged returns for both
spot and futures, forming a four-dimensional feature vector:
Equation 30
𝑥𝑡 = [𝑅𝑆,𝑡, 𝑅𝐹,𝑡, 𝑅𝑆,𝑡−1, 𝑅𝐹,𝑡−1]
This choice stems from empirical observations that immediate past returns influence current
price dynamics, enhancing model predictive power.
These features are organized into sequential data tensors using a rolling window appro ach
with a sequence length 𝑝 = 20, a hyperparameter chosen based on autocorrelation analyses
(ACF and PACF). This window length captures sufficient market history to model relevant
dependencies without introducing excessive noise or complexity.
Formally, the input tensor at time 𝑡 is:
Equation 31
𝑋𝑡 = [𝑥𝑡−𝑝, 𝑥𝑡−𝑝+1, … , 𝑥𝑡−1] ∈ ℝ𝑝×4
The corresponding target outputs are the next-day spot and futures returns 𝑅𝑆,𝑡, 𝑅𝐹,𝑡.

## Page 54

4.1.6.2.2 Neural Network Architecture
The model begins with an input layer accepting the 𝑝 × 4 tensor. This is followed by a one-
dimensional convolutional layer with 32 filters and kernel size 3, applying causal padding to
preserve the temporal ordering and prevent information leakage from the future. The
convolution operation transforms the input into 32 parallel feature maps of length 𝑝, where
each map encodes a specific temporal pattern learned during training.
ReLU activation is applied after convolution to introduce nonlinearity and mainta in
computational efficiency while mitigating gradient vanishing.
Next, these feature maps feed into an LSTM layer with 32 hidden units. The LSTM processes
the sequence of length 𝑝 and dimensionality 32, learning to retain or discard information
across time steps to form a fixed-length embedding vector summarizing the entire sequence.
A fully connected dense layer with a single linear unit then maps this embedding to a scalar,
interpreted as the predicted hedge ratio 𝛽̂𝑡.
Finally, a custom Lambda layer calculates the hedged portfolio return at time 𝑡 using
equation 32.
Equation 32
𝑅̂ ℎ𝑒𝑑𝑔𝑒𝑑,𝑡 = 𝑅𝑆,𝑡 − 𝛽̂𝑡 𝑅𝐹,𝑡
This operation embeds the hedging objective directly into the model structure, allowing end-
to-end learning focused on minimizing the residual variance of the hedged portfolio. The
model architecture pipeline can be seen in figure 10 below.

## Page 55

4.1.6.3 Model Compilation and Training Details
The model is compiled with the Adam optimizer, a stochastic gradient descent variant with
adaptive learning rates and momentum, set at a lear ning rate of 10−3 (Reyad, Sarhan, &
Arafa, 2023) . This optimizer is well -suited to complex neural architectures due to its
robustness and efficient convergence.
The training objective minimizes the mean squared error (MSE) between the hedged return
output and zero, reflecting the ideal of a perfectly hedged portfolio with no residual return:
Equation 33
ℒ = 1
𝑛∑(𝑅̂ ℎ𝑒𝑑𝑔𝑒𝑑,𝑖 − 0)2
𝑛
𝑖=1
= 1
𝑛∑ 𝑅̂ ℎ𝑒𝑑𝑔𝑒𝑑,𝑖
𝑛
𝑖=1

The hedge ratio output has zero weight in the loss, making it an auxiliary prediction driven
indirectly by minimizing portfolio residual risk.
The dataset is split into training and testing sets with an 80/20 ratio (Gholamy, Kreinovich,
& Kosheleva, 2018) , pr eserving chronological order to respect temporal dependencies.
Training proceeds for 50 epochs with batch size 64, using 20% of the training data for
Figure 10: Architecture and Pipeline of LSTM-CNN Hybrid Model

## Page 56

validation. Early stopping based on validation loss and learning rate reduction on plateaus
are implemented to prevent overfitting and stabilize training.
4.1.6.4 Mathematical Formulations and Model Equations
The modeling pipeline is grounded in rigorous mathematical definitions that underpin the
data transformation and neural network computations.
4.1.6.4.1 Input Tensor Construction
The input sequence at time 𝑡 is:
Equation 34
𝑋𝑡 = [
𝑅𝑆,𝑡−𝑝 𝑅𝐹,𝑡−𝑝 𝑅𝑆,𝑡−𝑝−1 𝑅𝐹,𝑡−𝑝−1
⋮ ⋮ ⋮ ⋮
𝑅𝑆,𝑡−1 𝑅𝐹,𝑡−1 𝑅𝑆,𝑡−2 𝑅𝐹,𝑡−2
] ∈ ℝ𝑝×4
This matrix forms the input tensor to the convolutional layer.
4.1.6.4.2 One-Dimensional Convolution Operation
For each of the 32 convolutional filters, the convolution output at time 𝑡, ℎ𝑐
(𝑡) is shown in
equation 35:
Equation 35
ℎ𝑐
(𝑡) = 𝑓(∑ 𝑤𝑐
(𝑘) ∙ 𝑥𝑡−𝑘 + 𝑏(𝑐)
𝐾−1
𝑘=0
)
where 𝐾 = 3 is the kernel size, 𝑤𝑐
(𝑘) the filter weights, 𝑏(𝑐) the bias term, and 𝑓(⋅)the ReLU
activation function defined in equation 36:
Equation 36
𝑓(𝑧)= max(0, 𝑧)
Causal padding ensures that the computation respects temporal ordering: the output at time
𝑡 depends only on inputs at 𝑡𝑖𝑚𝑒 ≤ 𝑡.

## Page 57

4.1.6.5 LSTM State and Output Update Equations
Given convolutional features ℎ𝑡 ∈ ℝ32, the LSTM updates are mentioned in equations 23 –
28.
Hedge Ratio Output
The final hidden state ℎ𝑝 ∈ ℝ32 after processing the full sequence is mapped via a dense
layer:
Equation 37
𝛽̂𝑡 = 𝑊β ℎ𝑝 + 𝑏β
where𝑊β ∈ ℝ1×32 and 𝑏β ∈ ℝ .
Hedged Return Calculation
The residual portfolio return post-hedging is presented in equation 34 which quantifies the
portfolio risk remaining after applying the predicted hedge ratio.
Loss Function
The model training objective is displayed in equation 38.
Equation 38
𝑚𝑖𝑛
θ
𝑛∑(𝑅𝑆,𝑡 − 𝛽̂𝑡 𝑅𝐹,𝑡)2
𝑛
𝑖=1

where θ encapsulates all neural network parameters. This objective corresponds to
minimizing the variance of residual returns, consistent with classical hedging principles
seeking minimum variance hedge ratios.
4.1.6.6 Model Evaluation and Diagnostics
Post-training, the model’s predicted hedge ratios 𝛽̂𝑡 are applied to the testing dataset to
compute hedged returns 𝑅̂ ℎ𝑒𝑑𝑔𝑒𝑑,𝑡. These returns are subjected to rigorous statistical and
financial evaluations.
Statistical tests on estimated hedged returns using KPSS stationarity test, Ljung -Box
autocorrelation test, and ARCH LM. These tests are conducted to ensure that the residuals

## Page 58

are white noise, and that they do not have autoregressive conditional heteroskedasticity. This
would help in indicating effective risk mitigation.
The measure of hedging effectiveness, as mentioned earlier, are defined by some standard
financial performance metrics like variance reductions, RMSE, Sharpe ratio, directional
accuracy, mean absolute deviation, VaR and CVaR reductions based on extreme value
theory. These measures make a good combination as they show a detailed picture of the risk
mitigation and model’s protection against tail risks.
The selected settings of sequence length, architecture depth, and hyperparameters contribute
to effective model generalization. Causal convolutions respect the coherence of temporal
data. The structured CNN –LSTM captures complex non -linear dependencies efficiently
(Ullah, et al.) . Incorporating the hedged return as a loss target, the model directly aligns
training with economic goals, offering transparent and economically relevant output.
4.1.7 Fourier Transform Network (FT-Net) Hybrid Model
The financial markets of today have become highly complicated. This is due to the fact that
many forces today do not just play out through time in serial ways. Also, they have different
periodicities, cyclical things, and persistent non-stationarities. Classical time-domain models
such as autoregressiv e models and even conventional deep learning networks like LSTM
and CNN often fail to capture these subtle features. Existing methods mainly capture time
dependencies. Though these methods have advanced to model complex dependencies, they
are ultimately time-domain approaches. Thus, such models may systematically miss spectral
features, such as cyclical features, business cycles, seasonality, volatility regimes,
fundamentally present in financial time-series.
The FT-Net Hybrid model is being developed with this problem in mind (Volpatti, 2024).
Evidence from financial econometrics and quantitative trading studies indicates that market
returns tend to follow some deterministic cycles and stochastic shocks. Things like
seasonality, day-of-the-week effects, business cycles, regime changes and periodic shocks
often show up as dominant frequencies in the spectrum. It is vital to capture these patterns
in conjunction with their time-domain dynamics for robust risk management and statistically
significant arbitrage opportunities.

## Page 59

FT-Net is a new hybrid deep learning architecture that combine the strengths of the spectral
(frequency-domain) and the temporal (time -domain) analysis. FT -Net combines
sophisticated temporal modeling blocks whic h are sometimes CNN based and other times
recurrent layers like LSTM or GRU with modules that are explicitly based on the Fourier
Transform to exploit the joint information in both domains. This method works best in equity
futures markets, where missed cyc lical risks, regime shifts can affect the performance of
hedging strategies, and price series can be spectrally decomposed for timelier realization of
arbitrage signals only at a later stage.
FT-Net allows the learning of richer and more adaptable hedge ratios and arbitrage signals
by designing a model that natively handles both spectral and temporal features. Our
framework goes beyond existing econometric models and conventional deep learning
models, which rely on some approximation of the joint distribution of returns or may become
non-stationary in power cycles or price cycles. The reasoning for selecting such a hybrid
model in this research is now clear, when we consider both the domains jointly we will
achieve an ideal dynamic hedge and arbitrage dete ction which is the need of the hour in
present-day equity futures trading.
4.1.7.1 Theoretical Foundations
To understand the FT-Net approach, it is significant to understand the theory behind using
spectral analysis modules. The FT-Net is a deep neural network architecture that makes use
of discrete Fourier transform (DFT), which is a mathematical tool that separates a finite
sequence or data point, such as historical returns, into the sum of sinusoidal components with
varying frequencies, amplitudes, and phases (Brigola, 2025).
The DFT for a time series {𝑥𝑛}𝑁 − 1
𝑛 = 0 of length 𝑁 is mathematically defined as:
𝑋𝑘 = ∑ 𝑥𝑛 ∙ 𝑒−2𝜋𝑖𝑘𝑛/𝑁𝑁−1
𝑛=0 , 𝑘 = 0,1,2 … 3
where 𝑋𝑘 is the complex -valued coefficient corresponding to the k -th frequency bin, 𝑥𝑛 is
the value of the time series at time 𝑛, and 𝑖 is the imaginary unit. The magnitude |𝑋𝑘|
represents the strength of the frequency component at a given frequency, while the angle
(argument) of 𝑋𝑘 represents its phase.

## Page 60

In the context of financial time series, the DFT allows researchers and practitioners to reveal
otherwise hidden periodicities, cycles, or regime changes. These spectral features may
correspond to macroeconomic cycles, regular trading patterns (such as day-of-week effects)
or shifts in volatility that occur in a non -stationary but recurrent manner. Notably, the
spectral domain can also provide early warnings of regime shifts —such as transitions from
low to high-volatility states , since changes in the energy or distribution of frequency
components can be detected before corresponding patterns are fully evident in the time
domain (Li, Wang, Wang, Yin, & Yao, 2025).
Several empirica l studies have documented the importance of spectral phenomena in
financial markets. For instance, business cycles (such as 3-5-year economic expansions and
contractions), monthly or quarterly seasonality, and even intraday periodicities all manifest
as peaks in the spectral density of return series. More sophisticated analyses, such as wavelet
or cross-spectral analysis, have further demonstrated that volatility clustering and contagion
effects can also exhibit distinct frequency signatures.
Despite this w ealth of evidence, most deep learning models in finance have continued to
operate in the time domain, missing the opportunity to leverage spectral information for
improved risk management and arbitrage strategies. FT-Net seeks to directly address this by
embedding spectral modules into its neural architecture, thus enabling the joint modeling of
time- and frequency-domain characteristics.
4.1.7.2 Model Architecture and Design of FT-Net Hybrid
The FT -Net Hybrid model represents a modular, end -to-end deep learning pi peline that
seamlessly integrates spectral analysis with advanced temporal modeling. The design
philosophy is rooted in the principle that financial market data contains interdependent
patterns in both domains, and only through their joint modeling can one hope to capture the
full spectrum of risks and opportunities.
The architecture of FT-Net can be conceptually divided into several key modules: the input
and feature extraction layer, the Fourier (spectral) feature module, the temporal
convolutional block, the sequence modeling block (such as an LSTM or transformer), and
the final output layers that compute the dynamic hedge ratio.
At the input stage, the model receives multi -dimensional feature matrices, typically

## Page 61

constructed from a window of recent observations of both the cash index returns and futures
returns, their lags, and optionally rolling statistical features such as moving averages, rolling
standard deviations, or higher-order moments. Each input sample thus consists of a sequence
of feature vectors over a fixed window length (sequence length).
The Fourier feature module forms the distinctive heart of the FT-Net. For each input window,
the model applies a real -valued fast Fourier transform (FFT) or, equivalently, the DFT, to
the time series of retu rns and their lags. This operation transforms the sequence from the
time domain to the frequency domain, yielding a set of complex coefficients that encode the
amplitude and phase of each frequency component. In practical implementation, both the
real and imaginary parts (or, in some variants, the magnitude and phase) of the transformed
coefficients are concatenated and passed as features to downstream layers. This explicit
extraction of spectral features enables the model to capture periodic, quasi -periodic, and
cyclical phenomena that would otherwise remain undetected.
Following the Fourier module, the model employs a temporal convolutional block —
typically composed of one or more one-dimensional convolutional layers. These layers scan
the time-sequenced (or frequency-sequenced) feature maps with learnable filters, allowing
the model to extract local temporal patterns, short -term dependencies, or shape -based
features that often encode meaningful financial events, such as volatility bursts or
microstructural price shocks.
After convolutional processing, the output is fused (via concatenation or learned fusion
mechanisms) with the original spectral features, ensuring that both time -domain and
frequency-domain representations are available for the subsequent seq uence modeling
block. This block usually consists of a recurrent neural network (RNN) layer, such as LSTM
(Long Short-Term Memory) or GRU (Gated Recurrent Unit), although transformer -based
encoders can also be used. The sequence modeling layer captures longer-term dependencies
and adapts to non-linear, non-stationary dynamics that may unfold over multiple time steps
or market regimes.
The penultimate layer of the network is a fully connected (dense) layer, which synthesizes
the outputs of all previous layer s and computes the dynamic hedge ratio (beta) or arbitrage
signal as a function of the fused feature space. Regularization techniques—such as dropout,

## Page 62

batch normalization, or layer normalization —are employed throughout the network to
mitigate overfitting and to stabilize training dynamics.
Activation functions are judiciously chosen to suit the nature of the output variable. For
instance, if the hedge ratio is constrained to the [0,1] interval, a sigmoid activation may be
used. In more general cases, a line ar or bounded tanh activation can be employed to allow
for negative or leverage-constrained hedge ratios.
A schematic block diagram of FT -Net would typically depict the input feature window
feeding into parallel branches: one path for the Fourier transform and extraction of spectral
coefficients, another for convolutional analysis of the raw sequences, both co nverging into
a fusion module, then passing through a sequence modeling block (LSTM/GRU), and finally
yielding the hedge ratio or arbitrage output through a dense readout layer. This modular
structure is instrumental in enabling FT -Net to exploit the full information content of
financial time series. The model architecture and pipeline of FT -Net Hybrid can be seen in
figure 11 below.

Figure 11: Architecture and Pipeline of FT-Net Hybrid Model

## Page 63

4.1.7.3 Feature Engineering and Selection in FT-Net Hybrid
The selection and engineering of input features is a critical determinan t of the FT -Net’s
ability to generalize and effectively learn meaningful patterns. The initial set of features
typically includes raw log returns of the index and future contracts, denoted as 𝑅𝑆,𝑡 and 𝑅𝐹,𝑡,
t, along with their lagged values over a w indow of length 𝑤. In financial theory and past
empirical evidence, the contemporaneous and lagged return frameworks is a good choice as
they capture not only contemporaneous price co-movements but also lead-lag relations and
potential information spillovers.
The defining contribution of FT-Net is its meticulous extraction and use of spectral features.
The application of DFT to rolling windows of returns and their lags induces encoding of
periodicities, of the cycles and of the dominant frequency bands as real inputs. We can use
various statistical diagnostics to empirically justify this approach. For instance, the
autocorrelation functions (ACF) of the return series, (refer to figure 7 ), show persistent
correlations at 1st lag. This pattern often indicates cyclical behavior. Many periodograms -
plots of spectral density versus frequency - show spikes at frequencies that align with
established market cycles and seasonal effects.
4.1.7.4 Mathematical Formulations and Model Equations
The FT-Net Hybrid Model integrates both spectral (frequency-domain) and temporal (time-
domain) information, leveraging the unique strengths of each to capture the full structure of
financial time series data for dynamic hedging and arbitrage signal extraction. To formalize
its workings, we provide a step-by-step mathematical exposition of each component of the
model, explain the rationale for each transformation, and relate every mathematical
operation to financial intuition.
Considering the financial time series, in our case the log return s of a stock index and its
associated futures contract. Denote by 𝑥𝑡 the value of a univariate feature (e.g., the log return
at time 𝑡), and by 𝑥𝑡 ∈ ℝ𝑑the 𝑑-dimensional feature vector at time 𝑡, which include raw
returns and lagged values.
4.1.7.4.1 Construction of the Input Tensor
For dynamic hedging or prediction, the model does not operate on a single time point, but
instead processes a rolling window of 𝑤 past observations. Thus, for each reference time 𝑡,

## Page 64

the input tensor can be represented as in equation 39.
Equation 39
𝑥𝑡 = [
𝑥𝑡−𝑤+1
𝑥𝑡−𝑤+2
⋮
𝑥𝑡
]
Where each row corresponds to a feature vector at a particular lag, and 𝑑 is the total number
of feature channels. This matrix encapsulates both the most recent and historical context the
model may require.
4.1.7.5 Spectral Transformation via Discrete Fourier Transform
The key innovation of the FT-Net model is the explicit transformation of each input feature
channel from the time domain to the frequency domain. For a given feature channel 𝑗, we
extract the corresponding column vector 𝑥𝑡,𝑗 = [𝑥𝑡−𝑤+1,𝑗, 𝑥𝑡−𝑤+2,𝑗, … , 𝑥𝑡,𝑗]
⊤
and apply the
Discrete Fourier Transform (DFT) (equation 40).
Equation 40
𝐹𝑘
(𝑗) = ∑ 𝑥𝑡−𝑤+1+𝑛,𝑗 ∙ 𝑒−2𝜋𝑖𝑘𝑛
𝑤𝑤−1
𝑛=0 , 𝑘 = 0,1,2, … , 𝑤 − 1
Where 𝐹𝑘
(𝑗) is a complex number representing the amplitude and phase of the 𝑘-th frequency
component for channel 𝑗, and 𝑖 is the imaginary unit. This operation projects the time-domain
sequence onto a basis of complex exponentials, revealing any cyclical, periodic, or
oscillatory structure in the input.
Each 𝐹𝑘
(𝑗) can be decomposed into its real and imaginary parts as in equation 41:
Equation 41
𝐹𝑘
(𝑗) = 𝑅𝑒 [𝐹𝑘
(𝑗)] + 𝑖 ∙ 𝐼𝑚[𝐹𝑘
(𝑗)]
Alternatively, the modulus 𝐹𝑘
(𝑗) gives the amplitude of the frequency, and the argument
arg (𝐹𝑘
(𝑗)) provides the phase. For machine learning purposes, the model typically flattens

## Page 65

the real and imaginary components across all 𝑘 and all feature channels, constructing the
frequency-domain feature vector in equation 42.
Equation 42
𝑆𝑡 = [𝑅𝑒(𝐹0
(1)), 𝐼𝑚(𝐹0
(1)), … , 𝑅𝑒(𝐹𝑤−1
(𝑑) ), 𝐼𝑚(𝐹𝑤−1
(𝑑) )]
⊤
∈ ℝ2𝑤𝑑
This vector fully captures the spectral energy distribution across the selected window for all
features. In practice, one may optionally select only a subset of 𝑘 (dominant) frequencies,
defined as those with the largest amplitudes (see equation 43).
Equation 43
𝑘∗ = arg𝑚𝑎𝑥
𝑘 |𝐹𝑘
(𝑗) |
Alternatively, the model may employ a soft attention mechanism in which learnable weights
𝛼𝑘 are applied to each frequency, resulting in an adaptive focus on the most predictive
frequency bands (equation 44):
Equation 44
𝑆𝑡
𝑎𝑡𝑡 = ∑ 𝛼𝑘 ∙ [𝑅𝑒(𝐹𝑘
(𝑗)), 𝐼𝑚(𝐹𝑘
(𝑗))]
𝑤−1
𝑘=0

subject to ∑𝛼𝑘 = 1, 𝛼𝑘 ≥ 0𝑘 .
4.1.7.5.1 Temporal Feature Extraction via Convolution
While the spectral branch extracts global cyclical structure, it is also essential to capture
local, short-term patterns in the raw time series. To this end, the input tensor 𝑋𝑡is processed
through one or more one-dimensional convolutional layers, defined in equation 45 below.
Equation 45
𝐶𝑡,𝑠
(𝑚) = ∑ ∑ 𝑊𝑙,𝑗,𝑠
(𝑚) ∙
𝑑
𝑗=1
𝑓−1
𝑙=0
𝑥𝑡−𝑤+1+𝑙,𝑗 + 𝑏𝑠
(𝑚)

## Page 66

where 𝑊𝑙,𝑗,𝑠
(𝑚) are the convolutional filters of width 𝑓, applied to each feature channel 𝑗 and
producing 𝑛𝑓𝑖𝑙𝑡𝑒𝑟𝑠 output channels 𝑠, with bias 𝑏𝑠
(𝑚). Non -linear activations, typically
rectified linear units (ReLU), are then applied (see equation 46).
Equation 46
𝐶𝑡,𝑠,𝑎𝑐𝑡
(𝑚) = max(0, 𝐶𝑡,𝑠
(𝑚))
Stacking multiple convolutional layers increases the receptive field, allowing the model to
detect more complex local and sequential structures such as abrupt price jumps, volatility
spikes, or temporal clusters.
4.1.7.5.2 Fusion of Time and Frequency Domain Representations
The model then concatenates the time -domain features extracted from the convolutional
pipeline with the frequency-domain features produced by the Fourier branch. This operation
creates the fused feature vector displayed in equation 47.
Equation 47
𝑍𝑡 = [𝐹𝑙𝑎𝑡𝑡𝑒𝑛(𝐶𝑡), 𝑆𝑡]⊤
Where 𝐹𝑙𝑎𝑡𝑡𝑒𝑛(𝐶𝑡) denotes the vectorization of the multi -channel output of the last
convolutional layer. The resulting 𝑍𝑡 ∈ ℝ𝑛𝑓𝑙𝑎𝑡+2𝑤𝑑provides a holistic, multi -scale
description of the input window.
4.1.7.5.3 Sequential Modeling with Recurrent Neural Networks
To allow the model to capture long -term dependencies, non-linear interactions, and regime
changes, the fused feature vector 𝑍𝑡 is passed through a recurrent neural network (RNN),
often an LSTM (Long Short -Term Memory) or GRU (Gated Recurrent Unit). The LSTM
update equations for each time step can be summarized in equations 23-28.
4.1.7.6 Computation of the Dynamic Hedge Ratio
The output layer of the FT-Net Hybrid model computes the dynamic hedge ratio for the next
period as a function of the hidden state ℎ𝑡. This is implemented as a fully connected (dense)
layer as shown in equation 48 below.

## Page 67

Equation 48
𝛽𝑡 = 𝜙(𝑤𝑜𝑢𝑡
⊤ ℎ𝑡 + 𝑏𝑜𝑢𝑡)
The predicted one-period-ahead hedged return is then calculated using equation 34.
This formula is derived from the principle of constructing a minimum-variance portfolio by
dynamically adjusting the exposure to the hedging instrument.
4.1.7.7 Loss Function and Training Objective
The model is trained to minimize risk, specifically by minimizing the variance of the hedged
return series. The empirical variance is given by equation 49.
Equation 49
𝑉𝑎𝑟(𝑅̂ ℎ𝑒𝑑𝑔𝑒𝑑,𝑖) = 1
𝑁 ∑(𝑅̂ ℎ𝑒𝑑𝑔𝑒𝑑,𝑖 − 𝑅̅ℎ𝑒𝑑𝑔𝑒𝑑,𝑖)
𝑁
𝑖=1
, 𝑅̅ℎ𝑒𝑑𝑔𝑒𝑑,𝑖 = 1
𝑁 ∑ 𝑅̂ ℎ𝑒𝑑𝑔𝑒𝑑,𝑖
𝑁
𝑖=1

However, for simplicity, and to center the returns around zero, the objective can be
formulated as mean squared error (MSE) relative to zero (equation 50).
Equation 50
ℒ𝑣𝑎𝑟 = 1
𝑁 ∑(𝑅̂ ℎ𝑒𝑑𝑔𝑒𝑑,𝑖)2
𝑛
𝑖=1

In practice, further regularization is introduced through techniques such as L2 weight decay,
dropout, and batch normalization to reduce overfitting. The network is optimized using
variants of stochastic gradient descent, such as Adam, which adaptively tunes the learning
rate for each parameter.
4.1.8 Model Evaluation and Diagnostics
The performance of FT -Net Hybrid model is tested extensively using a diverse range of
statistical as well as financial performance measures to guarantee that all facets of
performance must minimize risk and take arbitrage opportunity.
The raw and hedged return series are tested for stationarity and autocorrelation from a
statistical perspective. The KPSS test is used to test the null hypothesis of stationarity. The

## Page 68

test statistics and the p-values are computed for both in-sample (training) and out-of-sample
(testing) periods. The Ljung-Box Q-test checks for serial correlation in residuals in order to
determine how much the model has traded -off future autocorrelations. ARCH -LM test
basically answers whether the hedged returns have volatility clustering or if the risk has been
diversified adequately.
The MAE, MSE, RMSE and the MAD of the hedged returns compared with the target (which
should ideally be zero for a risk -neutral hedge) are used as the measures of the financial
performance. Variance reduction, which is defined as the percentage by which the variance
of the hedged returns is less than that of the un hedged series, measures how effective the
model is at minimizing risk.
The Theil U statistic measures how accurate the model is compared to a naive model where
a value of less than 1 means that the model is good. The measure of risk -adjusted return of
the hedged portfolio is called the Sharpe ratio. Directional accuracy is defined as the
frequency with which the model forecasts the direction of returns
To assess the model in the tails, extreme value theory (EVT) is applied to derive the value -
at-risk (VAR) and conditional value -at-risk (CVAR) at a certain level of confidence (e.g.
5%). Hedged series VaR and CVaR reductions versus benchmark, evaluate model efficacy.
The overall runtime (or computational cost) for which the model is reported is referred to as
time complexity. A model’s time complexity provides context for its usage in real -time (or
high-frequency) models. We also compute the average hedged return to see if this model has
a systematic bias or drift.
The assessment of FT-Net hinges on diagnostic visualizations. The graphs showing how the
beta value changes with time will tell us how the model adapts itself to things like volatile
condition or regime shifts. This plot compares the cumulative returns of the hedged and
unhedged portfolios and illus trates the actual outcome predicted by the model. Through a
periodogram or heatmap of dominant spectral features over time, we are able to gain further
insight into the ability of the model to capture and exploit the changing cycles present in the
market. In practical use, FT -Net performs positively during cyclical volatility or regime
change periods, as spectral features become informative during these times.

## Page 69

4.1.8.1 Practical Considerations and Model Strengths
The results from the FT-Net Hybrid model, as leveraged in this study to hedge the KSE 30
index futures, show the usefulness of using spectra coupled with advanced temporal forms
in empirical studies. In times of strong cycle behavior (e.g. return volatility bursts, business
cycle turning points, seasonal effects etc.) the FT-Net significantly outperforms econometric
models and standard deep learning models. Achieves greater variance reduction, Sharpe
ratios and directional accuracy.
FT-Net is effective mainly because it recognizes when the market changes. Mo re
specifically, FT -Net can adapt to changes in the frequency content of the market. For
instance, during regime switches (from a low volatility to a high volatility state), the splitting
of spectral energy across frequency bands changes relevantly. The explicit modeling of these
shifts by the FT-Net enables it to dynamically change hedge ratios and thus keep effective
risk management when other models do not.
Nevertheless, FT-Net is not without its limitations. As the model becomes more flexible and
complex, the risk of overfitting increases, especially when data is noisy or non -stationary.
This may affect the generalizability to other assets or market environments. It takes quite a
bit of time and energy to calculate; the application of DFT/FFT as well as the training of big
neural networks takes a lot of time. Any situation where high-frequency trading is going on,
it will use up quite a bit of resources.
Even with the challenges mentioned, the FT -Net Hybrid framework can effectively
withstand these issues . Thus, it offers a reliable avenue for dynamic hedging in equity
futures. Its strengths seem most evident in complex, non-stationary markets where traditional
models are incapable of adapting, and where modelling the time and frequency domain
jointly can provide important gains.
4.2 Post Model Performance Evaluation

4.2.1 Statistical Tests Post-Model Implementation
This study conducts statistical tests after implementing the models to assess the residuals,
evaluate the model’s forecasting quality and check the suit ability for comparative analysis.
To evaluate the accuracy of the predicted values, we conducted Theil U Statistics in the

## Page 70

hedged return series. Moreover, the ARCH-LM Test, Ljung Box Test, and KPSS Test were
performed on the residuals, by splitting up the dataset into training and testing periods. These
tests are performed to check whether the models exhibited autocorrelation, non-stationarity,
or heteroscedasticity in the residuals. In the econometric models, we used log -differenced
spot and futures return series of the KSE -30 index. These were useful for estimating the
structure of the volatilities and the time varying correlations. Moreover, for deep learning
models, hedged return series were used. These returns were based on the hedge ratios
obtained from the models, which are trained and tested using historical return information,
hence facilitating us to prevent overfitting, while dynamically capturing the hedging
effectiveness.
4.2.2 ARCH LM Test
As previously discussed, ARCH LM test was used in the pre -model testing phase of the
mathematical models to detect the presence of autoregressive conditional heteroscedasticity
in the return series, however a synthesis of this diagnostic was necessary for a thorough
model validation as well. In the pre-model phase of our study, the ARCH LM test was used
for providing evidence of conditional heteroscedasticity in the spot and futures return series
of the mathematical models. However, it is essential to evaluate the models' performance in
mitigating these effects afte r having been implemented. That is why the same diagnostic
tool is applied once again on the residuals of the fitted models. We choose not to repeat the
theoretical basis for the test here, as it has been discussed above. Instead, we evaluate how
well our models have absorbed volatility clustering and for checking overall
homoscedasticity in residuals which is required for valid inference and hedge ratio
estimation.
As presented in Appendix A, the p -value for the training data in DCC -GARCH is 0.06452
and 0.05857 which is above the standard 5% threshold. Although these results are marginally
significant, indicating minor heteroscedasticity, yet it is not enough to declare the model
invalid. The testing data with the p -values of 0.1427 and 0.1687 shows even s tronger
evidence for the absence of ARCH effects. Moreover, in Copula DCC GARCH, the p-values
for futures and spot returns of both training and testing data is above 0.1, indicating that the
student-t copula seems to model joint tail dependences and residu al volatility dynamics
well. Although Standard time series diagnostics are not as often applied to machine learning

## Page 71

models but we did apply them to the hedged return series to determine if the model was a
good fit. Therefore, as seen in the Appendix A, the p-values of ARCH LM test for the LSTM-
CNN and FT-Net models were all equal to or close to 1.000. This suggests that the models
are not only mathematically sound but are also useful for comparing the hedging
effectiveness.
4.2.3 Ljung Box Q Test
The Ljung-Box test evaluates the presence of serial correlation in the residuals of the model,
which implies that the past errors are affecting current errors. This contradicts the classical
white noise residuals assumption and questions the relevance of the model for capturing the
temporal dependencies.
In this research, Ljung -Box test for DCC -GARCH, under training and testing data for
futures/spot returns, produce the p -value greater than 0.1. These surprisingly large values
indicate that the residuals do not have any level of autocorrelation. Moreover, similar results
were obtained in Copula DCC -GARCH that reported higher training p -values and very
similar test values (as shown in Appendix A) which further confirmed the absence of
autocorrelation in both mathematical m odels. So, it is important to perform these
misspecification checks, in order to determine whether the model has a correct lag structure
or an absence of autoregressive components. Likewise, in the machine learning model the
p-values are greater than the 0 .5 significance level (see Appendix A) indicating that
uncorrelated residues were produced by the models on training and test data. Despite the fact
that this financial time series was non -linear and complex, these results indicate that the
models don’t have any systematic pattern of residuals remaining.
4.2.4 KPSS Test
The Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test is based on the analysis of stationarity
which is an essential assumption in time series modeling. In this test the null hypothesis
indicates that the time series is stationary, and it is rejected when the p-value is greater than
the standard 5% significance threshold.
In the case of the mathematical models, as shown in Appendix A, both DCC -GARCH and
Copula DCC-GARCH have p-values consistently above 0.1, across spot and future returns
in both training and testing sets. These findings confirm the stationarity of the series, which

## Page 72

is a necessary condition for the mathematical models. If the data was not stationary, our
volatility estimates would have been misleading, and we would have inflated Type I errors.
The same result was achieved in the machine learning model, which indicates that the
hedged return series are stationary again. This means that their learned structures compute
hedge returns that a re stable and mean -reverting. This finding is consistent with recent
works that suggests that deep learning models can approximate non -linear functions that
traditional models cannot represent.
4.2.5 Theil U Statistic
The Theil U statistic measures how the forec asting accuracy of a model compares to a
benchmark model, which in our study is the naïve 1:1 hedging strategy. This test was
performed on the entire data sets for both mathematical and deep learning models. A U
statistic that is less than one means that the model does better than the naïve model while the
reverse is true for the U statistic that is greater than one.
The U-statistic of Copula DCC -GARCH is 0.8528, which is significantly better than DCC
GARCH with U statistic value of 0.91. This demonstrates its stronger ability to produce
better forecasted values than a naive hedge, and therefore its superiority. Therefore, the
mathematical model's goal of minimizing risk is represented quantitatively by a lower Theil
U, which also validates the model structu re. Although machine learning models were
outperformed on residual diagnostics, Theil U statistics of LSTM -CNN (0.9179) and FT -
Net (0.9128) were slightly higher than those of econometric models. It is still less than 1 and
thus provides evidence to a better performance than a naive hedge.
4.2.6 Performance Evaluation
This paper further evaluates the hedging effectiveness of the proposed econometric and deep
learning models by establishing a comprehensive performance evaluation framework. This
framework facilitate s us to assess the market fluctuations and apprehend the varying
efficiencies of each model to mitigate risk. The depiction of the hedging performance and
findings of the statistical analysis for each model helps in identifying which model is reliable
and resilient in the context of Pakistani financial market. Hedge ratios are an important
parameter to minimize the variance in the hedged returns; it indicates the proportion of future
contracts required to hedge the risk of the spot position. In this study, the dynamic models

## Page 73

display moving hedge ratios to account for the change in variance and co -variances of the
spot and future returns. Therefore, these MVHRs are compared to observe the ability of each
model to manage the risk considering the volatility and instability of the Pakistani financial
market.
4.2.6.1 Predicted MVHRs for Mathematical Model

The MVHRs, estimated by Copula DCC -GARCH and DCC -GARCH models, shown in
figure 12 and 13, highlight the in-sample period in light blue series and out-of-sample period
in orange series. The MVHRs in the training sample for both models fluctuate approximately
around 0.8 and 1.12, indicating stable hedge ratios (β) for the 2019 -22 region. However,
despite the EWMA filter to smooth out the noise and stabilize volatility, the hedge ratios in
both models show spike of approximately 1.2 and 1.1 followed by a dip of approximately
0.75 and 0.8 due to the onset of the COVID -19 pandemic, reflecting unpredictable change
in covariance’s. Proceeding to the testing period, increased fluctuations are observed with a
dip of 0.75 and peak of approximately 1.30 for both models in the 2023 -24 region. This
hedging pressure reflects the impact shown by the DCC-GARCH and Copula DCC-GARCH
model to previous stress episodes. While comparing the Cop ula DCC-GARCH and DCC -
GARCH models, it can be observed that Copula DCC -GARCH demonstrates short lived
adjustments due to a slightly lower average MVHRs, combined with amplified oscillations
in hedge ratios for the period of extreme shocks reflecting t -copulas ability to capture tail
dependence and asymmetric co -movements, while DCC -GARCH has marginally higher
average hedge ratios.

## Page 74

4.2.6.2 Predicted MVHRs using Machine Learning Models
Moreover, the MVHRs estimated by LSTM -CNN and FT -Net hybrid models, shown in
figure 14 and 15, highlight the in-sample period in dark blue series and out-of-sample period
in orange series. The hedge ratios (β) in LSTM -CNN model with its peak of 1.05 in the
training period has a subdued response to extreme events, when compared to FT -Net,
reflecting the model’s ability to avoid overstating and understating hedge ratios, suitable for
volatile` Pakistani market. The convolutional layer and temporal dependency of the LSTM
layer of the LSTM -CNN model fluctuates around 0.94 -0.98 range, showing reduced short
noise yet lower and non -reactive hedging performance when compared to FT -Net. On the
Figure 13: Dynamic MVHRs obtained using Copula DCC GARCH
Figure 12: Dynamic MVHRs obtained using DCC GARCH

## Page 75

other hand, FT -Net shows hedge ratios (β) mainly close to 1, suggesting better hedge
performance. In both deep learning models, a sudden spike i.e. 1.10 in FT -Net and 1.05 in
LSTM-CNN, is observed in the early 2020, due to the onset of COVID -19 pandemic,
reflecting uncertainty similar to the mathematical model. However, FT-Net highlighting its
mean-reverting behavior stabilizes the hedge ratios quickly compared to the other models.
The testing period fluctuating around 0.96 and 0.99 shows reliable out of sample
performance and the model’s ability to quickly capture non-linear complexities. Comparing
the hedge ratios of mathematical and deep learning models, it is observed that although all
the models exhibit mean reverting behavior, yet FT-Net has higher suitability in dynamic ss
conditions, as it shows greater responsiveness to the volatile Pakistani market.
Figure 15: Dynamic MVHRs obtained using FT-Net Hybrid
Figure 14: Dynamic MVHRs obtained using LSTM-CNN

## Page 76

4.2.7 Hedged Returns Estimation
4.2.7.1 Hedged Returns for Mathematical Learning Model
After the hedging ratios in the training and testing data, we computed the hedged returns for both
DCC-GARCH and Copula DCC -GARCH using equation 34. For qualitative inspection, we
plotted the time series of predicted hedge ratios 𝛽̂𝑡 on training and test dates, as well as the
cumulative unhedged versus filtered hedged returns . Cumulative hedged and unhedged returns
are calculated using equation 51.
Equation 51
𝐶𝑢𝑚𝑈𝑛ℎ𝑒𝑑𝑔𝑒𝑑(𝑡) = ∑ 𝑅𝑆,𝑖
𝑖≤𝑡
, 𝐶𝑢𝑚𝐻𝑒𝑑𝑔𝑒𝑑(𝑡) = ∑ 𝑅ℎ𝑒𝑑𝑔𝑒𝑑,𝑖
𝑓𝑖𝑛𝑎𝑙
𝑖≤𝑡

These figures 16 and 1 7 below vividly illustrate the risk‐reduction benefits of our dynamic
hedging procedure. As illustrated in the figure 16 below for DCC-GARCH, the hedged returns
(in orange line) show smoother, upward sloping returns as compared to the unhedged returns,
which show clear fluctuations especially during the COVID -19 pandemic period. Whereas
Copula DCC -GARCH depicts slightly higher returns as compared to DCC -GARCH, due to
effective hedging and risk reduction.
Figure 16: Cumulative Hedged and Unhedged Returns – DCC GARCH

## Page 77

4.2.7.2 Hedged Returns for Machine Learning Model
Similarly, in LSTM-CNN model, after training, and applying the model to both training and test
windows to extract 𝛽̂𝑡 and computed hedged returns using the equation 34 for each sample.
Denoting the model’s predicted ratios on train and test as { 𝛽̂𝑡
(𝑡𝑟)
} and { 𝛽̂𝑡
(𝑡𝑒)
}, we concatenated
these to form a full‐period series of unfiltered hedged returns.
To correct systemati c errors in the CNN –LSTM’s hedged returns, we constructed an auxiliary
dataset of residuals 𝑒𝑡 = 𝑅̂ ℎ𝑒𝑑𝑔𝑒𝑑,𝑡 and regressed them against their own recent history and the
raw returns. Specifically, we built a DataFrame with columns shown in equation 52.
Equation 52
{𝑒𝑡, 𝑅𝑆,𝑡, 𝑅𝐹,𝑡, 𝑒𝑡−1, 𝑒𝑡−2, 𝑒𝑡−3, 𝑅𝑆,𝑡−1, 𝑅𝑆,𝑡−2, 𝑅𝑆,𝑡−3, 𝑅𝐹,𝑡−1, 𝑅𝐹,𝑡−2, 𝑅𝐹,𝑡−3}
After dropping initial NaNs, we again split chronologically into training and full sets. We then
trained three gradient-boosting regressors—XGBoost, LightGBM, and CatBoost—with 200 trees
each (CatBoost silent, random_state = 0 for reproducibility). Each mo del learned to predict 𝑒𝑡
from the aforementioned features. We averaged their outputs to obtain an ensemble correction 𝑒̂𝑡
and defined our final filtered hedged return as in equation 53.
Equation 53
𝑅ℎ𝑒𝑑𝑔𝑒𝑑,𝑡
𝑓𝑖𝑛𝑎𝑙 = 𝑅̂ ℎ𝑒𝑑𝑔𝑒𝑑,𝑡 − 𝑒𝑡
Figure 17: Cumulative Hedged and Unhedged Returns – Copula DCC GARCH

## Page 78

Moreover, for FT-Net, after predicting 𝛽̂𝑡 for both train and test windows and compute hedged
returns. This direct application of the model’s output equation obviates the need for separate
covariance/variance calculations, as the network has internalized the mapping from spectral‑time
features to optimal ratio.
Similar to the LSTM -CNN model, to capture any systematic biases remaining in the hedged
returns, we form a residual series𝑒𝑡 = 𝑅̂ ℎ𝑒𝑑𝑔𝑒𝑑,𝑡 and construct lagged regressors up to three days.
We then fit an ensemble of gradient-boosted regressors—XGBoost, LightGBM, and CatBoost—
each with 200 trees, on the training residuals, and average their predictions. The final filtered
hedged return is computed same as equation 53 above, yielding a "double -filtered" series with
further reduced variance.
The depiction of cumulative hedged and unhedged returns below vividly illustrate the risk‐
reduction benefits of our dynamic hedging procedure. While both strategies deliver a very flat,
low-volatility hedged profile until mid-2023, the FT-Net hybrid (figure 19) edges out the LSTM–
CNN model (figure 18) by generating a slightly higher peak cumulative hedged return —around
+3% versus about +2% —and then decaying more gradually. In practice this means FT -Net’s
frequency-temporal layers are a bit quicker at lockin g in gains when the market trends, whereas
the LSTM –CNN’s convolution -plus-recurrent structure produces an equally smooth but
marginally lower total hedge payoff.

Figure 18: Cumulative Hedged and Unhedged Returns – LSTM-CNN

## Page 79

Figure 19: Cumulative Hedged and Unhedged Returns – FT-NET Hybrid

## Page 80

5 Comparative Analysis using Fuzzy TOPSIS
5.1 Introduction
In Chapter 4, we formulated hedging strategies for a KSE 30 index portfolio using the KSE 30
futures contract through both sophisticated econometric models, such as the Dynamic
Conditional Corr elation Generalized Autoregressive Conditional Heteroskedasticity (DCC
GARCH) and its copula -augmented variant, and cutting -edge deep learning architectures,
including the LSTM-CNN and FT-Net Hybrid models, to dynamically minimize risk in the equity
futures market. The primary objective of this chapter is to rigorously assess the hedging
effectiveness of the model by using a comprehensive suite of quantitative performance metrics
and rank them with respect to their hedging performance. In this study, we have employed nine
performance criteria to gauge the hedge effectiveness, which include the variance reduction
percentages, Root Mean Square Errors (RMSE), average hedged returns, Sharpe ratios,
directional accuracy, Value at Risk (VaR) Reduction, Conditional Value at Risk (CVaR)
Reduction, mean absolute deviation, and time complexity. These metrics effectively capture the
risk minimization capability and the operational efficiency of the models. However, the
challenges arise from the multicriteria analysis, a s no single model unequivocally dominates all
criteria, emphasizing the conflicts in the multi -criteria decision-making (MCDM) context. This
predicament created the need for the adoption of a systematic methodology to assess the
performance of each model and rank them accordingly.
To address this problem, we incorporated the traditional Technique for Order Preference by
Similarity to Ideal Solution (TOPSIS) method, a multicriteria decision -making technique. The
conventional TOPSIS method ranks multiple a lternatives based on their closeness to an ideal
solution while maximizing benefits and minimizing costs. The following section provides an in-
depth discussion of the TOPSIS methodology used in this research.
5.2 Implementation of the TOPSIS methodology
5.2.1 Define the decision-making problem
The first step of the TOPSIS methodology is to define the decision -making problem. In this
research, our objective is to identify the hedging model that effectively dynamically hedges the

## Page 81

risk inherent in the KSE 30 index portf olio. The alternative models under consideration include
the Copula GARCH, DCC GARCH, LSTM-CNN Hybrid, and FT-Net Hybrid models.
To obtain fair and multidimensional assessment results, we utilise nine key performance metrics
that gauge the risk minimizatio n capability, predictive accuracy, profitability, and operational
efficiency of the models.
i. Variance reduction helps us assess the extent to which a model reduces the variance of
the hedged portfolio compared to the variance of the unhedged portfolio.
ii. The RMSE evaluates the model's ability to forecast the hedge ratios; lower RMSEs
indicate more precise predictions.
iii. The average hedged return is an indicator of the profitability of the models, as it reflects
the model’s ability to maintain the portfolio returns while minimizing the risk.
iv. Sharpe ratio is a measure of risk -adjusted return as it reflects the ability of a model to
achieve higher returns given the risk minimization criteria.
v. Directional accuracy is also an indicator of the predictive accuracy of the models;
however, it focuses on the frequency with which the model correctly predicts the direction
of market movements.
vi. Value at Risk (VaR) reduction is a measure of how effectively a model reduces potential
losses at a specified confidence level.
vii. Conditional Value at Risk (CVaR) reduction further extends the analysis of downside risk
by assessing the expected average loss beyond the VaR threshold.
viii. Mean absolute deviation (MAD) is an additional measure of predictive accuracy.
ix. Finally, time complexity measures the computational resources and time required for each
model to execute. It is an essential metric to ga uge the operational efficiency of the
models.
Collectively, these criteria provide a robust suite of metrics to gauge the key characteristics of
the models. Tables 1 and 2 below show the criteria for all models on the training and testing
datasets.

## Page 82

Table 1: Performance Metrics of Hedging Models – Training Model

Table 2: Performance Metrics of Hedging Models – Testing Model

5.2.2 Determination of AHP weights
After identifying all decision criteria, the fuzzy TOPSIS multicriteria decision -making
methodology requires weights to be assigned to each of them. Therefore, to assign weights to
every criter ion, we use the Analytic Hierarchy Process (AHP), which converts a complex
problem into a hierarchy of sub -problems (Sood, Pathak, & Gupta, 2025) . We begin by
constructing a pairwise comparison matrix, where each element represents the relative
importance of one metric over the other. We created a 9x9 pairwise comparison matrix (Table
3). Each entry (i, j) in this matrix expresses how much more important criterion i is compared to
criterion j. In the table below, each metr ic is represented as follows: variance reduction (C1),
RMSE (C2), average hedged return (C3), sharpe ratio (C4), directional accuracy (C5), VaR
reduction (C6), CvaR reduction (C7), mean absolute deviation (C8), time complexity (C9).
METRICS COPULA
GARCH
GARCH
LSTM-CNN
HYBRID
FT-NET
HYBRID
Variance reduction 92.54 92.71 99.9 99.9
RMSE 0.01241 0.01234 0.00041 0.00042
Average Return -0.000024 -0.000046 0.000004 0.000004
Sharpe Ratio -0.0067 -0.013 0.0098 0.0102
Directional Accuracy 66.63 70.09 50.81 52.64
VaR Reduction 83.58 83.97 97.66 97.55
CVaR Reduction 86.63 87.02 97.31 97.23
MAD 0.00886 0.0088 0.000239 0.000238
Time Complexity 6.66 4.479 10.79 9.63
METRIC COPULA
GARCH DCC GARCH LSTM-CNN
HYBRID
FT-NET
HYBRID
Variance Reduction 95.09 95.05 96.96 96.56
RMSE 0.01118 0.01121 0.00206 0.00219
Average Return 0.00029 0.00034 -0.00009 -0.00005
Sharpe Ratio 0.1113 0.129 -0.0438 -0.0227
Directional Accuracy 62.07 67.82 51.34 51.34
VaR Reduction 89.39 88.54 94.65 94.38
CVaR Reduction 90.97 90.1 91.74 91.62
MAD 0.00824 0.00817 0.000611 0.000619
Time Complexity 4.48 6.39 10.79 9.63

## Page 83

Table 3: Computation and Fuzzification of AHP Pairwise Comparison Matrix

C1 C2 C3 C4 C5 C6 C7 C8 C9
C1 1 5 5 5 7 7 7 9 9
C2 1/5 1 3 3 5 5 5 7 7
C3 1/5 1/3 1 2 4 4 4 6 6
C4 1/5 1/3 1/2 1 3 3 3 5 5
C5 1/7 1/5 1/4 1/3 1 2 2 4 4
C6 1/7 1/5 1/4 1/3 1/2 1 2 3 3
C7 1/7 1/5 1/4 1/3 1/2 1/2 1 3 3
C8 1/9 1/7 1/6 1/5 1/4 1/3 1/3 1 2
C9 1/9 1/7 1/6 1/5 1/4 1/3 1/3 1/2 1

Once the initial pairwise matrix is constructed, it is normalized to ensure consistency and to allow
for proper weighting. The matrix is normalized by dividing each entry in a given column by the
sum of that column, transforming the original elements into relative proportions. The normalized
matrix reflects the proportional importance of each criterion relative to every other criterion i n
the set.
Then the weight of each criterion is extracted by computing the mean value of each row in the
normalized matrix. This average quantifies the overall relative importance of each criterion,
aggregating its normalized influence across all pairwise comparisons. The vector of these row
means represents the initial set of crisp (precise) weights, which together sum to one, ensuring a
proper probability distribution over the criteria.

## Page 84

Instead of considering each crisp weight, we convert them to fuzzified values by constructing a
Triangular Fuzzy Number (TFN) centered on the crisp value but allowing for a range of
uncertainty. Specifically, for each weight w, a TFN is defined as (0.9w, w, 1.1w), thereby
incorporating a ±10% spread around the nominal value. This fuzzification step captures the
ambiguity and lack of perfect precision, making our criteria weighting process more realistic and
defensible.
The fuzzified AHP weights are presen ted in the figure 20 above from highest to lowest. The
highest weights were assigned to variance reduction and RMSE, while the lowest weights were
assigned to the mean absolute deviation and time complexity.
5.2.3 Fuzzification of decision criteria
After defining and calculating all criteria and their fuzzified weights, the next step was to convert
the criteria to fuzzy numbers by using Triangular Fuzzy Number (TFN), defined by a lower,
modal, and upper bound. The fuzzification of the metrics was performed by adding ambiguity to
each performance criterion, such as variance reduction, RMSE, Sharpe ratio, and others - the
code applies a small, symmetric fuzziness margin around the observed value. Specifically, if the
original metric value is v, it is ma pped to a TFN (l, m, u) where 𝑙 = 𝑣 × (1 − 𝛿), 𝑚 = 𝑣, and
𝑢 = 𝑣 × (1 + 𝛿), with δ representing the fuzziness factor. This approach means that the lower
bound 𝑙 models a pessimistic scenario, the modal value m represents the most likely estimate,
and the upper bound u captures an optimistic possibility for that metric. By constructing the entire
Figure 20: Fuzzified AHP Criteria Weights

## Page 85

decision matrix with TFNs instead of single values, the fuzzy TOPSIS process can rigorously
propagate and analyze the uncertainty throughout the normalization, weighting , and ranking
stages. Ultimately, this robust fuzzification ensures that the final model selection and ranking are
not sensitive to minor inaccuracies or subjectivity in the original data, yielding results that more
faithfully reflect real-world ambiguity and imprecision.
5.2.4 Normalize the decision matrix
With the fuzzified decision matrix and weights in place, the next procedural step is normalization.
Here, each criterion is scaled such that its values are directly comparable across models,
regardless of the ir original units or scales. For benefit -type criteria (where higher values are
preferable, such as variance reduction or Sharpe ratio), normalization rescales all values to fall
within a range of zero to one, with the best observed value mapped to one. Conversely, for cost-
type criteria (where lower values are better, such as RMSE, MAD, and time complexity), the
transformation is reversed to ensure consistency in the evaluation framework. This ensures that
the subsequent aggregation of criteria is both meaningful and equitable.
5.2.5 Calculate weighted normalized matrix
Following normalization, each entry of the decision matrix is multiplied by its corresponding
fuzzified weight. This step ensures that criteria with higher relative importance exert
proportionally greater influence on the final scores. The resulting weighted normalized matrix
thus provides a comprehensive, uncertainty-aware summary of each model’s performance across
all criteria.
5.2.6 Determine positive and negative ideal solutions
The TOPSIS methodology then identifies the Positive Ideal Solution (PIS) and Negative Ideal
Solution (NIS) for each criterion. The PIS represents the optimal scenario, i.e., the best
(maximum for benefit, minimum for cost) value observed among all alternatives for each
criterion. Conversely, the NIS captures the worst-case scenario (minimum for benefit, maximum
for cost). By comparing each model to these reference points, the methodology quantifies both
how close a model is to the best possible outcome and how far it is from the worst.

## Page 86

5.2.7 Calculate Euclidean distance from ideal solutions
To rigorously quantify the proximity of each alternative to the ideal and anti-ideal solutions, the
code calculates the fuzzy Euclidean distance of each model’s performance from both the PIS and
NIS, carefully accounting for the spread and structure of each TFN. For each model, these
distances are then aggregated across all criteria.
The final step involves computing the closeness coefficient for each model, defined as the ratio
of its distance from the negative ideal solution to the sum of its distances from both the positive
and negative ideals. The closeness coefficient, ranging between zero and one, provides a single
scalar value reflecting how desirable each alternative is relative to the opt imal benchmark.
Models with higher closeness coefficients are considered more effective, as they are closer to the
ideal and farther from the worst-case performance.
5.2.8 Rank the alternatives
Based on the calculated closeness coefficients, all alternative hedging models are ranked from
most to least effective. This fuzzy TOPSIS framework thus ensures a transparent, uncertainty -
aware, and multidimensional evaluation of hedging models, robustly incorporating both data -
driven performance metrics and subjective judgments about their importance. The result is
presented in the figure 21 below.
Figure 21: Model Ranking by Fuzzy TOPSIS

## Page 87

As per the Fuzzy TOPSIS criteria, the FT-NET Hybrid model achieved the highest score of 0.444,
closely fol lowed by the LSTM -CNN Hybrid model at 0.425. These results suggest that both
advanced deep learning -based approaches outperform the conventional econometric models
DCC GARCH (0.336) and Copula GARCH (0.309) when evaluated on the integrated criteria of
risk minimization, predictive accuracy, profitability, tail risk management, and computational
efficiency.
The marginal lead of the FT-NET Hybrid over the LSTM-CNN Hybrid highlights the added value
of incorporating temporal features into the hedging strategy, as opposed to relying solely on
sequential modeling. On the other hand, the Copula GARCH and DCC GARCH models, while
competent in certain aspects, lag in overall performance when all criteria and uncertainties are
considered simultaneously.
This ranking, derived from the Fuzzy TOPSIS methodology, provides clear quantitative evidence
for the superiority of deep learning hybrid models in the context of dynamic hedging for the KSE
30 index in Pakistan. Moreover, the scores indicate not just raw performance, b ut a
comprehensive and robust assessment that accounts for the inherent ambiguity and imprecision
present in real-world financial modeling and decision-making.
5.3 Summary
The Fuzzy TOPSIS methodology, as implemented in this research, provides a sophisticated and
robust approach to evaluating and ranking the dynamic hedging models across multiple, often
conflicting, criteria while explicitly acknowledging the presence of real -world uncertainty and
subjectivity. The process started with the careful definition o f the decision -making problem,
which centered on selecting the most effective hedging model for minimizing risk and optimizing
performance for the KSE 30 index futures market. This requires us to synthesize nine
comprehensive metrics to form a multidimensional assessment framework.
The assignment of criteria weights using the Analytic Hierarchy Process (AHP) was a crucial
step, translating expert judgment into a structured, quantifiable form. The resulting weights were
further fuzzified to reflect the uncertainty and imprecision that naturally accompany subjective
decision-making. Fuzzification was also applied to the performance metrics, converting each

## Page 88

observed value into a Triangular Fuzzy Number (TFN) that encompasses a range of estimates.
This dual layer fuzzification ensured tha t all subsequent analyses accounted for the ambiguity
found in the time series data.
Through this, the fuzzy decision matrix was normalized so that all metrics could be directly and
fairly compared, regardless of their units or scales. Each normalized, fuz zified metric was then
weighted according to its fuzzified AHP weightages, resulting in a weighted normalized decision
matrix. Following the weights determination the positive and negative ideal solutions were
identified. Then, the fuzzy Euclidean distance s of each model to these ideal solutions were
computed, which then led to the closeness coefficients for each model. These closeness
coefficients integrated all performance dimensions into a single interpretable score.
The results revealed the FT-NET Hybrid model as the top performer, achieving the highest Fuzzy
TOPSIS score and thus ranking closest to the ideal solution. The LSTM -CNN Hybrid model
ranked second, also demonstrated strong overall effectiveness across all criteria. Both DCC
GARCH and Copula G ARCH models, while still viable, received lower overall scores,
highlighting the comparative advantage of advanced deep learning architectures over traditional
econometric methods in the context of dynamic hedging.

## Page 89

6 Statistical Arbitrage Analysis

6.1 Introduction
Statistical arbitrage refers to a class of trading strategies that use statistical, mathematical and
computational methods to exploit pricing inefficiencies between financial instruments. Statistical
arbitrage is widely understood as a high-volume, short-term trading strategy. Unlike traditional
arbitrage, which exploits mispricing between identical or related securities for risk -free profits,
statistical arbitrage distinguishes itself by assuming such mispricing is subtle, short -lived, and
revealed only by complex analysis. Statistical arbitrage involves taking advantage of mutual
relationships, for example, mean reversion, cointegration or other statistical properties. These
relationships must last long enough to be traded systematicall y, regardless of the noise of the
market or frictions.
Statistical arbitrage is based on the idea that the prices of related financial assets are not always
perfectly in sync or adjust instantaneously, thus giving rise to a temporary deviation from their
equilibrium relation. These deviations arise due to various reasons like liquidity shock, investor
reaction and information lag and so on. They open up arbitrage opportunities if they can be
detected and acted on before prices revert to normal co -movement. It’s worthwhile noting that
signals based on statistical patterns can affect prices themselves. A similar point can be made
regarding time -series based strategies. Statistical arbitrageurs use quantitative techniques for
creating trading signals, managing risk, and executing trades either at high frequency or in a
disciplined, repeatable manner. They rely on the law of large numbers, and diversification across
many trades to secure consistent returns.
6.2 Relevance in Equity Futures Markets
The use of statistical arbitrage strategies has become relevant in equity futures markets due to the
high liquidity, transparency and standardization of contracts making systematic trading possible.
Equity futures like the ones on broad -based indices exhibit strong statistic al relationships with
their underlying cash indices as they are arbitraged to link the spot and futures price.
Nevertheless, due to market microstructure effects, varying levels of liquidity, rolling contract
dynamics, and short-term supply-demand imbalances, temporary mispricing between the futures
and their respective underlying indices do occur. These transitory departures from theoretical

## Page 90

pricing models, such as the cost -of-carry model, create an environment in which statistical
arbitrage strategies can thrive.
In the equity futures market, the search for statistical arbitrage opportunities often revolves
around the mean-reverting behavior of spreads, price differences, or constructed portfolios that
are theoretically expected to maintain a certain equil ibrium. For instance, as the expiration of a
futures contract approaches, its price is expected to converge to the spot price of the underlying
asset. However, the path to convergence is rarely smooth due to market frictions and investor
behavior, thus giv ing rise to repeated, albeit short -lived, deviations. Advanced statistical
techniques can be employed to model these deviations, generate trading signals, and capture
profits from the anticipated reversion to equilibrium, provided that the statistical relationships are
robust, stationary, and economically exploitable after accounting for transaction costs and
slippage.
6.3 Motivation for Statistical Arbitrage in Hedging Context
The motivation to incorporate statistical arbitrage within a hedging framework stems from the
inherent imperfections and dynamic nature of financial markets, particularly in emerging market
contexts such as Pakistan. Hedging with futures contracts is designed to minimize risk by
offsetting exposures in the spot market with opposing positions in the futures market. However,
in practice, the effectiveness of any hedging strategy depends not only on the strength of the
statistical relationship between the spot and futures but also on the ability to dynamically adjust
hedge ratios in response to evolving market conditions.
Traditional hedging strategies, such as static or rolling hedge ratios estimated by ordinary least
squares, are limited in their capacity to adapt to non -stationarities, sudden market shocks, and
changes in volatility regimes . This creates opportunities for sophisticated statistical arbitrage
approaches that explicitly model the time-varying nature of hedge ratios and the spread between
the hedged and unhedged positions. By systematically exploiting periods when the hedge is
temporarily “ineffective” or when the spread between the hedged portfolio and its theoretical
value widens abnormally, statistical arbitrage becomes a complementary tool for both profit
generation and risk minimization. The incorporation of mean -reverting s ignals, stationarity
testing, and dynamic trading rules allows for a more flexible, adaptive, and potentially more

## Page 91

profitable hedging process that not only manages risk but also actively seeks to exploit market
inefficiencies.
6.4 Statistical Arbitrage in the Context of the Project
Within the framework of this project, statistical arbitrage is operationalized as a rigorous,
quantitative methodology for detecting and exploiting short -term pricing inefficiencies arising
from the dynamic hedging of a KSE 30 index portfolio using its corresponding futures contract.
The core focus is on the hedged return series, which is constructed by applying model -derived
hedge ratios estimated from both traditional econometric and advanced deep learning models —
to the spot and fut ures returns. The residual return, after applying the hedge, represents the
“spread” or divergence from perfect hedging effectiveness. If this spread is stationary and mean-
reverting, it becomes a candidate for statistical arbitrage, as systematic deviations from the mean
are expected to be temporary and hence tradable.
To operationalize statistical arbitrage in this context, the project deploys a framework that
includes rolling calculation of statistical parameters (such as mean and standard deviation of the
hedged return), transformation into standardized z-scores, and generation of trading signals based
on deviations from the mean. These signals dictate when to enter or exit long and short positions
on the spread, effectively betting on its reversion to t he equilibrium implied by the underlying
statistical model. The analysis is further enriched by rigorous stationarity testing using the
Augmented Dickey-Fuller test to ensure that the trading strategy is grounded in robust statistical
properties, rather th an random or spurious correlations. The project benchmarks the arbitrage
performances of various models, including DCC -GARCH, Copula-GARCH, and LSTM -CNN
and FT -Net Hybrid, to ascertain which model provides the maximum exploitable arbitrage
opportunities. This will assist in operationalizing statistical arbitrage strategies in the growing
derivatives markets in Pakistan. Statistical arbitrage thus is both an indicator of hedge
effectiveness, and an intelligent, adaptive quantitative trading strategy which is part of the overall
research agenda.
6.5 Methodological Framework for Statistical Arbitrage
The practical execution of statistical arbitrage strategies makes a powerful analytical technique
that converts the raw data provided by the market into powerful trad ing signals. In the project,

## Page 92

statistical arbitrage is grounded on the dynamic modelling of hedge ratios, calculation of hedged
returns, generation of signals using rolling statistics and statistical properties like stationarity
along with others are useful ly checked to make inference. Each stage in this framework is
designed to maximize both the interpretability and the effectiveness of the arbitrage signals,
ensuring that the trading opportunities identified are both statistically valid and economically
meaningful.
6.6 Construction of Hedged Return Series

At the heart of the statistical arbitrage methodology is the construction of the hedged return series,
which forms the primary spread to be analyzed and traded. In financial markets, the concept of a
“spread” refers to the difference between two related financial quantities —most commonly, the
prices or returns of assets that are expected to move together due to fundamental or statistical
relationships. In the context of this research, the spread is spe cifically defined as the return on a
dynamically hedged portfolio comprising a position in the KSE 30 index (the spot asset) and an
offsetting position in the corresponding futures contract. The central idea is that, by carefully
calibrating the exposure t o the futures contract, it is possible to reduce or neutralize the risk
arising from movements in the spot market.
The formula used to calculate the hedged return at each time step t is mentioned in equation 34.
The hedge ratio 𝛽̂𝑡 quantifies the degree to which the futures position offsets the risk of the spot
position. The value of 𝛽̂𝑡 is not static but is recalculated at each time point using sophisticated
models, such as DCC -GARCH, Copula -GARCH, LSTM -CNN, or FT -Net Hybrid, so as to
capture changing market conditions and correlations.
The resulting hedged return series, 𝑅̂ ℎ𝑒𝑑𝑔𝑒𝑑,𝑡, can be interpreted as the “residual risk” or “spread”
that remains after applying the hedge. Ideally, if the hedge were perfect and all market
movements were fully a nticipated by the model, the hedged return would be close to zero.
However, in reality, due to estimation errors, market frictions, and the inherent unpredictability
of financial markets, the spread will exhibit variability and, importantly for statistical arbitrage,
may display patterns of mean reversion or deviation from equilibrium. By focusing on this spread,
the statistical arbitrage methodology is able to detect and systematically exploit these temporary
dislocations for profit.

## Page 93

6.7 Rolling Statistics and Z-Score Normalization
Next steps after computing the hedged return series, or spread, are to track how this spread
behaves over time to get actionable trading signals. The primary tool used for this analysis is the
calculation of rolling statistics, which ensure that abnormal deviations are detected with respect
to most of the recent history rather than a fixed mean or standard deviation. This is essential in
financial markets where statistical properties may evolve over time due to regime changes,
volatility clustering, or shifts in investor sentiment.
The rolling mean and rolling standard deviation are defined mathematically as follows (equation
54) for a window of size 𝑤:
Equation 54
𝜇𝑡 = 1
𝑤 ∑ 𝑅̂ ℎ𝑒𝑑𝑔𝑒𝑑,𝑡
𝑡
𝑖=𝑡−𝑤+1

Equation 55
𝜎 = √
𝑤−1 ∑ (𝑅̂ ℎ𝑒𝑑𝑔𝑒𝑑,𝑡 − 𝜇𝑡)𝑡
𝑖=𝑡−𝑤+1
where 𝜇𝑡 represents the rolling mean of the hedged return at time 𝑡, and 𝜎 denotes the rolling
standard deviation over the same window. By recalculating these statistics at each time step, the
method adapts to evolving market conditions and ensures that the detection of anomalies is
contextually relevant.
To further standardize the detection of trading opportunities, the spread is normalized into a z -
score (see equation 56), which expresses the current hedged return in terms of its distance from
the rolling mean, scaled by the rolling standard deviation:
Equation 56
𝑧𝑡 = 𝑅̂ ℎ𝑒𝑑𝑔𝑒𝑑,𝑡 − 𝜇𝑡
𝜎𝑡

The z-score, 𝑧𝑡, provides a dimensionless measure of extremity: values close to zero indicate that
the spread is near its recent average, while large positive or negative values signify abnormal

## Page 94

deviations. In the context of statistical arbitrage, these z -score threshol ds form the basis for
generating entry and exit signals. For example, when the z -score exceeds a certain positive
threshold, the spread is considered “overbought” and likely to mean-revert downwards, triggering
a short position. Conversely, when the z -score falls below a negative threshold, the spread is
“oversold” and expected to revert upwards, prompting a long position. This approach leverages
the inherent tendency of mean -reverting series to oscillate around a stable equilibrium, thus
enabling the systematic exploitation of temporary mispricing.
6.8 Stationarity Testing and Justification

The effectiveness and validity of any statistical arbitrage strategy is largely dependent on whether
the hedged return series is stationary. Statistical features that enable time series analysis are often
gathered at regular intervals, making the time -series analysis of stationary data important.
Stationarity implies that history matters for what may happen in the future. This is exactly the
assumption behind repeated arbitrage opportunities. If the spread is not constant over time, then
any movement away from the average could be permanent, which would render any trading
strategy based on the idea of moving back towards the average too erratic to implement or too
dangerous to implement.
To formally test for stationarity, the Augmented Dickey-Fuller (ADF) test is employed. The ADF
test is a statistical hypothesis test in which the null hypothesis is that a unit root is present in the
time series, indicating non -stationarity. The test is based on estimating th e regression shown in
equation 57:
Equation 57
𝛥𝑦𝑡 = 𝛼 + 𝛽𝑡 + 𝛾𝑦𝑡−1 + ∑ 𝛿𝛥𝑦𝑡−𝑖 + 𝜀𝑡
𝑝
𝑖=1

where 𝑦𝑡 is the value of the spread (here, 𝑅̂ ℎ𝑒𝑑𝑔𝑒𝑑,𝑡), 𝛥𝑦𝑡 denotes the first difference of 𝑦𝑡, 𝑡 is a
time trend, 𝑝 is the number of lagged differences included to account for autocorrelation, and 𝜀𝑡
is the error term. The key parameter of interest is 𝛾 is significantly less than zero, the null
hypothesis of a unit root is rejected, indicating stationarity.

## Page 95

The outcome of the ADF test includes the test statistic, critical values at standard significance
levels, and the p -value. A sufficiently negative test statistic, or a p -value below the chosen
significance threshold (typically 0.05), provides statistical evidence that the series is stationary.
In the context of this project, the application of the ADF test to the hedged return series ensures
that the mean -reversion signals generated by the z -score normalization are grounded in sound
statistical properties, rather than being artifacts of a trending or random -walk process. This will
not only make the arbitrage signals more reliable but also protect against model failure or false
positive signal in the live trading environment.
This project combines the building of hedged returns with rolling statistical analysis, z -score
normalization and stringent stationarity testing to create a methodological framework for
statistical arbitrage. As a result, it assures that the strategies proposed are both theoretically sound
and empirically proven for the KSE 30 equity futures market.
6.9 Signal Generation and Trade Execution Logic
A disciplined and rule -based approach is required for identifying and systematically exploiting
arbitrage opportunities for signals and execution. The aim is to capture short -term price
inefficiencies using a repeatable process that minimizes discretion and boosts statistical power.
This part describes how to design the signals that enable mean -reversion, how to manage the
trading positions, and how to ensure that key parameters are calibrated in the right way so as not
to adversely impact the performance of the strategy.
6.10 Mean-Reversion Signal Design
Statistical arbitrage strategy is essentially b ased on mean -reversion phenomenon. This is a
tendency for a financial time series such as a spread, or hedged return to drift away from the
common value. However, it eventually reverts back towards the mean. The wider or the narrower
the spread becomes com pared to recent history, the more likely it will move back towards the
mean or the equilibrium. The detection of such deviations forms the basis for generating
actionable trading signals.
In implementation, the strategy uses the previously calculated z-score. This z-score tells us how
many standard deviations the current value of the spread is from its rolling mean.

## Page 96

Hedged return also called a spread at time taken equal to the rolling mean over rolling standard
deviation over a window equal 𝜇𝑡 and 𝜎𝑡 respectively. A z -score converts the spread into a
standardized measure, which allows comparing offsets today directly, regardless of changes in
its volatility or average level over time.
Trading signals are produced by determining an entry threshold ( 𝑧𝑒𝑛𝑡𝑟𝑦) and an exit threshold
(𝑧𝑒𝑥𝑖𝑡). The thresholds are statistical cutoffs that determine what deviations away from the mean
are significant and what level of reversion will be sufficient to close a trade. The entry and exit
conditions for trading is mathematically defined as follows.
1. Long Entry (Buy Spread): When the z -score falls below the negative of the entry
threshold, i.e., 𝑧𝑡< −𝑧𝑒𝑛𝑡𝑟𝑦 , this signals that the spread is abnormally low and likely to
mean-revert upwards. A long position is initiated, betting on an increase in the spread.
2. Short Entry (Sell Spread): Conversely, when the z -score rises above the positive entry
threshold, i.e., 𝑧𝑡 > 𝑧𝑒𝑛𝑡𝑟𝑦, the spread is deemed abnormally high and expected to revert
downwards. A short position is initiated.
3. Exit Signal (Close Position): Regardless of the initial direction, when the absolute value
of the z -score returns below the exit threshold, i.e., |𝑧𝑡| < 𝑧𝑒𝑥𝑖𝑡, it indicates that the
spread has normalized, and the trade should be closed to realize profits and mitigate the
risk of reversal.
These rules can be mathematically summarized as in equation 58:
Equation 58
𝑃𝑜𝑠𝑖𝑡𝑖𝑜𝑛= {
1 𝑖𝑓𝑧𝑡 < −𝑧𝑒𝑛𝑡𝑟𝑦
−1 𝑖𝑓𝑧𝑡 > 𝑧𝑒𝑛𝑡𝑟𝑦
0 𝑖𝑓 |𝑧𝑡| < 𝑧𝑒𝑥𝑖𝑡

This systematic approach increases the likelihood of making profitable trades or closing out
losing trades. They are opened once confirmed deviations and closed when the deviation is gone.
Using z -scores rather than absolute values, it provides an adaptive strategy that responds to
market conditions and volatility regimes.

## Page 97

6.11 Trade Position Management
The management of trading positions in a statistical arbitrage strategy is governed by the signals
described above, with explicit rules dictating when to ente r, hold, or exit trades. At any given
time, the strategy can be in one of three possible states: long, short, or flat (no position).
A long position is taken when the spread is judged to be excessively low, based on the z -score
dropping below the negative entry threshold. In practical terms, this involves buying the spot
index and selling the corresponding number of futures contracts as specified by the hedge ratio,
with the expectation that the spread will rise. Conversely, a short position is initiated when the z-
score exceeds the positive entry threshold, indicating the spread is excessively high; this entails
selling the spot index and buying futures contracts, profiting from a decline in the spread.
Trade transitions are managed through a straightforwar d process. When the z -score signal
triggers a new position (long or short) and the strategy is currently flat, the position is opened
accordingly. If a position is already open and the z-score crosses the exit threshold in the direction
of normalization, t he position is closed, returning the strategy to a flat state. Importantly, the
system is designed to avoid simultaneous long and short positions; at most, only one direction is
active at any time. If the signal reverses before the exit threshold is hit (f or example, from long
entry to short entry without normalization), the previous position is first closed before the new
position is opened, ensuring clear transitions and accurate accounting of profits and losses.
This position management logic ensures dis cipline, prevents overtrading, and allows for clean
measurement of individual trade performance. The sequence of position changes is directly
mapped to the time series of z-score signals, making the strategy transparent and auditable.
6.12 Parameter Optimization

The performance of statistical arbitrage hinges significantly on the selection and tuning of key
parameters; namely, the rolling window size used for computing statistics, and the entry and exit
z-score threshold values. How often and how profitably a trading strategy might trade is
determined by these parameters, as is how robust the strategy is to various situations.
The size of the rolling window (𝑤) affects the mean and standard deviation of the spread. When
a window is smaller in size. The strategy will respond to the latest change effectively. But, it can

## Page 98

miss on big trends. On the other hand, if you have a bigger window, you will smooth short-term
variations and perhaps prevent overfitting, but you can be slow to react to real regime changes.
Choosing a window size is, therefore, a trade-off between sensitivity and stability. Using out-of-
sample back testing and validation for empirical testing helps find the optimal window to balance
these competing concerns specifically for the KSE 30 futures market.
Changes to the entry z -score threshold (𝑧𝑒𝑛𝑡𝑟𝑦) and exit z-score threshold (𝑧𝑒𝑥𝑖𝑡) will affect the
responsiveness of the strategy. When entry thresholds are lower, traders trade frequently as
signals are triggered with even small deviations from the mean. Although this may lead to greater
opportunities, it also increases the chances of a false positive and transaction costs. When the
thresholds are high, the strategy will only activate when conditions are extraordinary. The idea is
that the quality of trades will be superior on average. However, it will also activate far less often.
Thus, total profitability might be lowered due to fewer returns and trades. The exit threshold
controls how tightly we control our trades. A tighter exit locks in our profit quickly and limits
our drawdown. A looser exit will allow us to capture more profit but increases the odds of a
reversal.
Usually, the optimization of parameters is done through in -depth back testing of a grid of
parameters. Then the raw retu rn, risk-adjusted return (Sharpe ratio), drawdown, and number of
trades are measured. The choice of final parameters is done to balance profitability and risk and
operational suitability. The optimized strategy should not rely heavily on any specific perio d or
historical regime. Thus, a sensitivity analysis must be conducted to test the results across different
market conditions.
This research incorporates the principle of signal generation, discipline of position management,
and optimization of parameters. This statistical arbitrage framework is methodologically and
practically sound, forming a solid basis for the empirical outcomes illustrated hereafter.
6.13 Empirical Results and Visualization

This part presents the empirical outcomes of the statistical arbitrage framework, consisting of an
in-depth analysis of each hedging model's strategy. The parameter configuration, the dynamics
of the hedged spread, the strategy equity curves, and performance of all models. Each

## Page 99

visualization and table of results is di scussed to clarify the actual world actionability and
statistical significance of the results.
6.13.1 Model-by-Model Strategy Performance

Upon executing the statistical arbitrage strategy testing interface (see Figure 22), the analysis was
conducted with a rolling window of 60 days, an entry z-score threshold of 2.0, and an exit z-score
threshold of 0.5. The empirical evaluation covers four distinct models for hedge ratio estimation:
Copula-GARCH, DCC -GARCH, LSTM –CNN Hybrid, and FT -Net Hybrid. Each model file
encapsulates daily returns data for the KSE 30 index and its futures contract, along with the
dynamically computed hedge ratio and the resulting hedged return series. Figure 20 shows the
GUI developed to assess the statistical arbitrage effectiveness.
A summary of the performance metrics for each model is provided in Table 4 below. These
metrics include total return, annualized return, annualized volatility, Sharpe ratio, maximum
drawdown, number of trades executed, and results of the Augmented Dickey -Fuller (ADF)
stationarity test. The Sharpe ratio, which measures risk -adjusted performance, is used as the
principal comparative criterion for determining the best performing strategy.

Figure 22: Graphical User Interface (GUI) for Statistical Arbitrage

## Page 100

Table 4: Statistical Arbitrage Performance Metrics by Model.
Model
Total
Return
Annual
Return
Annual
Volatility
Sharpe
Ratio
Max
Drawdown
Trades
ADF
Statistic
ADF p-
value
Crit 5%
Copula
GARCH
0.056 0.011 0.012 0.864 -0.015 54 -13.003 0.000 -2.864
GARCH
0.059 0.011 0.012 0.951 -0.014 54 -13.020 0.000 -2.864
LSTM-
CNN
Hybrid
0.066 0.013 0.012 1.074 -0.015 52 -39.123 0.000 -2.864
FT-NET
Hybrid
0.066 0.013 0.012 1.077 -0.015 52 -38.939 0.000 -2.864

The FT-Net Hybrid model marginally outperformed all other models in terms of Sharpe ratio,
closely followed by the LSTM –CNN Hybrid. Both advanced deep learning models produced
higher total and annualized returns than the econometric Copula-GARCH and DCC -GARCH
models, while maintaining comparable volatility and drawdown characteristics. The strong
negative ADF statistics and zero p-values confirm robust stationarity in the hedged return series
for all models, substantiating the statistical basis for mean-reversion-based arbitrage.
6.13.2 Visualization of Hedged Spread Dynamics

The dynamic behavior of the hedged return (spread) for each model is visualized in figures 23-
26. Each graph plots the time series of hedged returns along with its rolling mean, entry bands
(±2.0 standard deviations), and exit bands (±0.5 standard deviations) over the entire sample
period from 2019 to 2024.
The hedged return series produced by the Copula -GARCH model oscillates around the rolling
mean, with most value s contained within the entry bands. Periodic spikes represent significant
short-term dislocations, often coinciding with market stress or contract rollover periods. The
rolling mean remains stable near zero, validating the effectiveness of the hedge. The width of the
entry and exit bands adapts dynamically with volatility, expanding during turbulent periods and
narrowing during tranquil market regimes.
The DCC -GARCH model’s hedged return dynamics shows patterns broadly similar to the
Copula-GARCH model. The frequency and magnitude of excursions beyond the entry bands are
slightly higher during high volatility episodes, yet the spread consistently reverts to the mean.

## Page 101

This mean -reversion property is essential for arbitrage, as it allows repeated entry and exi t
opportunities throughout the sample.
Next comes the LSTM–CNN Hybrid model’s hedged return series. This model exhibits a slightly
tighter clustering of returns around the mean, with fewer extreme outliers. The dynamic entry
and exit bands provide a visual cue for when the arbitrage strategy is likely to activate trading
signals. The persistently mean-reverting behavior is evident, reinforcing the statistical foundation
of the arbitrage approach.
The FT-Net Hybrid model’s spread demonstrates exceptional mea n-reverting tendencies, with
the vast majority of values remaining within the rolling bands. The spectral features incorporated
by this model appear to enhance the stability and predictability of the spread, reducing the
occurrence of unprofitable outliers. The visual compactness and symmetry of the hedged returns
suggest robust risk control.

Figure 23: Spreads and Bands – Copula DCC GARCH

## Page 102

Figure 26: Spreads and Bands – LSTM-CNN
Figure 24: Spreads and Bands –DCC GARCH
:
Figure 25: Spreads and Bands – FTNET Hybrid

## Page 103

6.13.3 Equity Curve and Comparative Performance
To evaluate actual trading performance, Figures 2 7 – 30 plots the accumulated profit -and-loss
(PnL) curves and entries of all the individual trades for the model. Accumulated PnL represents
the total impact of each arbitrage trade, signed based on the s -z-score before its entry and exit
decision.
The equity curve of the Copula -GARCH model demonstrates a stable rising tendency with a
moderate number of drawdowns in the early sample. Each trade is marked to facilitate
understanding, clearly demonstrating that part icipating in long and short trades both contribute
to the increased profitability of both parties.

Figure 27: Strategy Equity Curve – DCC GARCH

## Page 104

Figure 28: Strategy Equity Curve – Copula DCC GARCH
Figure 29: Strategy Equity Curve – LSTM-CNN
Figure 30: Strategy Equity Curve – FTNET Hybrid

## Page 105

The DCC-GARCH model also follows a comparable trend, albeit with a smoother evolution and
lesser drawdown episodes. The gradual changes in the equity curve show that the model can deal
with changing volatility conditions and achieve consistent performance.
The LSTM-CNN Hybrid model has seen an increase in total PnL, with a longer rising equity
curve and small drawdowns. The model’s adaptive learning mechanism determines when to go
long or short and helps the investor earn a higher compounded return than competing econometric
models.
FT-Net Hybrid has the highest cumulative return among all models in terms of equity curve. The
equity curve shows a generally upward direction, especially in the later sample years, with few
reversals. The FT-Net Hybrid model is the number one model by Sharpe ratio. This is due to its
robust and consistent execution of trade.
Finally, figure 31 overlays the cumulative PnL curves for all four models, providing a direct
visual comparison of strategy performance over time.
This comparative equity curve reveals the relative strengths and weaknesses of each model.
While all models demonstrate positiv e growth and mean-reversion-driven profitability, the FT -
Net Hybrid and LSTM –CNN Hybrid models consistently outperform the econometric
alternatives, both in terms of total return and in the smoothness of the equity curve. The absence
of large, persistent drawdowns further confirms the statistical robustness and practical viability
of the deep learning approaches.
Figure 31: Cumulative PnL Comparison Across Models

## Page 106

6.14 Interpretation and Practical Implications

The empirical results clearly demonstrate the superiority of advanced deep learning models,
particularly the FT-Net Hybrid, in capturing and exploiting short-term arbitrage opportunities in
the KSE 30 futures market. All models satisfy the statistical requirements for mean -reversion
trading, as evidenced by highly significant ADF test statistics. The enha nced Sharpe ratios,
combined with stable drawdown and trade frequency metrics, highlight the ability of data-driven,
adaptive models to dynamically adjust to changing market regimes and microstructure dynamics.
From a practical standpoint, these findings suggest that deploying a statistical arbitrage strategy
based on the FT -Net Hybrid model can deliver superior risk -adjusted returns compared to
conventional econometric techniques. The robust stationarity of the hedged spread and the
effective capture of mean-reversion opportunities enable systematic, repeatable profit generation
in Pakistan’s evolving equity derivatives landscape.
6.14.1 Model Selection and Arbitrage Exploitation

The ultimate objective of the statistical arbitrage framework is not only to evalu ate the
comparative effectiveness of different modeling approaches but also to translate these insights
into actionable strategies that can be robustly deployed in live trading environments. This section
synthesizes the empirical evidence, examines the rob ustness of the results across key parameter
configurations, and outlines practical recommendations for real-time exploitation of the optimal
arbitrage signals.
6.14.2 Comparative Assessment

The comparative analysis of model performance is grounded in a comprehe nsive suite of
quantitative metrics and visual diagnostics. Across all four tested models —Copula-GARCH,
DCC-GARCH, LSTM–CNN Hybrid, and FT-Net Hybrid—the results consistently indicate that
advanced deep learning architectures, specifically the FT -Net Hybrid and LSTM–CNN Hybrid,
offer a superior edge in the detection and exploitation of statistical arbitrage opportunities.
The FT-Net Hybrid model emerges as the most effective, as evidenced by its highest Sharpe ratio
of 1.07680 and a total return nearly iden tical to that of the LSTM –CNN Hybrid. Both models
deliver not only higher absolute and risk -adjusted returns but also demonstrate more consistent

## Page 107

equity curve progression, marked by persistent upward momentum and limited drawdowns
throughout the out-of-sample evaluation period. The econometric models, while still profitable
and statistically significant, underperform relative to their deep learning counterparts both in
terms of total returns and risk-adjusted metrics.
Figures 25 - 28 depicting the cumulativ e profit and loss trajectories (see Section 6.13.3)
underscore this conclusion visually: the FT-Net Hybrid’s equity curve is not only smoother and
less volatile but also achieves the highest terminal value over the full testing horizon. The hedged
return spread associated with this model is characterized by a strong mean -reverting tendency,
fewer extreme outliers, and more predictable band crossings, all of which translate to high-quality
arbitrage signals. Furthermore, the ADF test statistics for all models are strongly significant, but
the FT -Net Hybrid and LSTM –CNN Hybrid models exhibit the most pronounced levels of
stationarity in their spreads, further cementing the statistical reliability of the arbitrage process.
6.14.3 Robustness and Sensitivity Checks

A crucial aspect of any empirical trading strategy is its robustness to variations in core parameter
choices. To assess this, the sensitivity of model performance to changes in the rolling window
size, entry z-score threshold, and exit z-score threshold was systematically evaluated (Liu, 2023).
The findings confirm that, while absolute returns and trade frequency do vary with parameter
adjustments, the overall rank ordering of model performance remains stable.
Both the FT-Net Hybrid and LSTM–CNN Hybrid models maintain their outperformance across
a reasonable range of parameterizations. Increasing the rolling window length generally results
in smoother equity curves and fewer false signals, at the cost of reduced responsiveness to sudden
market changes. Adjusting the entry z-score threshold modulates the trade-off between frequency
of trading and the statistical extremity required for entry, but the deep learning models continue
to generate superior Sharpe ratios even under more conservative (higher) thresholds. The exit
threshold also impacts average trade duration and the ability to capture mean -reversion profits,
yet the FT-Net Hybrid’s advantage in rapid and precise identification of mean -reverting moves
is consistently preserved. This robustness to parameter shifts indicates that the performance edge
is not the result of overfitting to a particular regime or a narrow range of market conditions but
rather reflects a genuine improvement in model-driven signal quality.

## Page 108

6.14.4 Real-Time Exploitation Strategy

Having established the FT -Net Hybrid as the optimal model for statistical arbitrage in this
context, it is essential to translate these findings into practical guidelines for real -world trading
implementation. At the core of the real-time strategy is the continuous monitoring of the hedged
return spread, calculation of rolling mean and standard deviation over the optimal window, and
dynamic computation of the z -score to trigger trade signals according to the pre -defined entry
and exit thresholds.
Upon detection of an actionable signal —a z -score exceeding the entry threshold in either
direction—the trader or automated trading system would enter a position in the direction of mean
reversion. This involves either going lo ng on the hedged spread (buying spot, selling futures)
when the spread is unusually low, or short (selling spot, buying futures) when the spread is
unusually high. Position sizing should be determined based on a comprehensive risk management
framework, taking into account both signal strength (the magnitude of the z -score) and overall
portfolio risk exposure. For instance, trade size can be scaled proportionally to the absolute z -
score or set to a fixed fraction of capital, constrained by volatility and value-at-risk limits.
Execution efficiency is also paramount, especially in emerging markets where liquidity may be
episodic and market impact more significant. Real -time arbitrage systems should include
slippage controls and avoid trading during periods of thin liquidity, contract rollovers, or known
market disruptions. Dynamic stop -loss mechanisms, trailing exits, or volatility -adjusted exit
thresholds can further mitigate the risk of large adverse moves, ensuring that the realized profit
distribution closely mirrors the back tested results.
Additionally, given the stationarity of the spread and the high frequency of mean -reverting
opportunities, it is advisable to automate the signal monitoring and order placement process to
minimize latency and eliminate be havioral biases. Post -trade analytics, including attribution of
profit to long and short trades, tracking of trade duration, and periodic recalibration of model
parameters, will help maintain performance and adapt to structural shifts in the market over time.
Hence, the FT -Net Hybrid -based statistical arbitrage strategy offers a highly effective,
empirically validated, and practically robust approach to exploiting short -term pricing
inefficiencies in the KSE 30 index futures market. Its integration into a l ive trading operation,

## Page 109

guided by disciplined risk management and rigorous signal monitoring, can deliver superior risk-
adjusted returns while preserving statistical integrity and operational resilience.
6.15 Discussion and Implications

6.15.1 Interpretation of Arbitrage Profitability

The profitability demonstrated by the statistical arbitrage strategies in this study carries important
implications for our understanding of market dynamics, model capabilities, and the real -world
feasibility of advanced hedging approaches. The consistent outperformance of deep learning
models, specifically the FT -Net Hybrid and LSTM –CNN Hybrid, provides strong empirical
evidence that sophisticated, data -driven techniques are better equipped to capture transient
inefficiencies and complex nonlinear dependencies in the equity futures market than conventional
econometric methods. This result reflects the inherent adaptability of machine learning models
to shifts in market regimes, microstructure effects, and the often-subtle mean-reversion signals
embedded in noisy financial data.
From an economic perspective, the ability to extract persistent, statistically significant profits
from a theoretically efficient market such as the KSE 30 futures points to the existence of
exploitable anomalies and time-varying frictions. All models tested created stationary and mean-
reverting spreads, suggesting that even in liquid, well-arbitraged environments, there are pricing
deviations and temporary misalignments stemming from market constraints like information
asymmetry, tran saction lags, behavioral biases, and heterogeneous agents. The better Sharpe
ratios and stable equity curves of FT -Net Hybrid and LSTM –CNN Hybrid models further
indicate that these anomalies are not mere random noise, but systematic patterns that advanced
algorithms can learn and exploit.
These results provide support for the statistical arbitrage strategy not simply as an academic
concept, but also applicable in the real world. The results show that dynamic hedge ratio
estimation along with robust statistical signal processing enables practitioners to routinely harvest
short-term arbitrage profits, while controlling for risk. Market making strategies tend to have
low drawdowns, high win rates, and fast mean -reversion in live simulation, suggesting an
implementable model supported by robust risk management and automated execution.

## Page 110

On a larger scale, the proven success of deep learning -based statistical arbitrage illustrates the
evolving nature of financial markets. As there is more data and computational res ources are
accessible, the frontier of alpha is moving toward models that generate value from increasingly
complicated, nonlinear, high -dimensional, and fast -changing data. This study’s successful
outcomes show a breakthrough in quantitative trading in eme rging markets as Pakistan, where
technology-based methods outsmart traditional ones in efficiency and profits.
6.15.2 Limitations and Caveats
Even though the statistical arbitrage strategies shown showed good performance there are several
limitations and caveats that need to be pointed out which give it a balanced outlook and set future
research. A major restriction is that the back tested calculations do not include transaction costs.
In practice, using high-frequency trading strategies like arbitrage comes with costs from broker
fees and taxes. Even small costs of trading can eat into your profits, particularly for strategies
that trade often or are thinly traded.
Market liquidity presents yet another significant challenge. Although the KSE 30 futures market
has a fair amount of liquidity during normal hours of trading, circumstances such as a stressed
market, contract rollover or shock events can cause a sudden drop in tradable volume, while the
bid-ask spread may be seen widening. The poor conditions can affect the quality of the execution
and may also result in unexpected losses if trades cannot be quickly unwound at expected levels.
Another limitation stems from the possibility of regime shifts and structural market breaks. Even
though analysis covers many yea rs and market conditions, it does not warranty that relations
noted in the past will apply in the future. Changes in regulations, the availability of new financial
products, shocks to the economy, or advancements in technology can all cause correlations
between different financial assets to behave in a fundamentally different way.
The risk of models becoming overfit is ever-present in data-driven approaches. Although out-of-
sample validation and parameter robustness checks were carried out, some measure of fitting to
historical quirks is possible. Regular recalibration and constant monitoring are required to prove
their worth.
The use of these models, especially major deep learning architectures, requires significant data
preparation, computing power and dom ain expertise. To use such systems in a live trading

## Page 111

environment, strong data governance, continuous monitoring and stringent operational control
are needed. These ensure reliability, conformity, risk management and other standards.
6.16 Summary

The study conducted a detailed and systematic analysis of the return to statistical arbitrage on a
dynamic hedge ratio for the KSE 30 equity futures market. The study showed that advanced data-
driven approaches, like the FT -Net Hybrid and LSTM-CNN Hybrid models, produced superior
risk-adjusted returns with a stable performance unlike traditional model by incorporating
sophisticated econometric models as well as deep learning architecture. The evidence suggests
that adaptive models are better able to exploit transient in efficiencies and mean -reverting
opportunities in financial markets, even those that are comparatively efficient.
The outcome also provides significant practical insights about such strategies since it is shown
that arbitrage on mean reversion being statist ically powerful and economically sizable can take
place through proper risk management and actual execution. Also at the same time, transaction
costs as well as liquidity constraints and structural market changes should be carefully assessed
as they can materially affect the realized performance.
To sum up, this work contributes to the theoretical development of statistical arbitrage and
dynamic hedging and also provides actionable insights for practitioners looking to implement
quantitative strategies in n ew and evolving markets. The bunch of generated code hints the
prominent possibility of machine learning in finance. We are likely to see further innovation and
application of their use in quantitative trading and risk management.

## Page 112

7 Implementing FT-Net Hybrid in a Macroeconomic
Context

7.1 Introduction and Rationale for Macroeconomic Integration
In the evolution of financial risk management models, it has become increasingly evident that
robust hedging frameworks must go beyond traditional price -based inputs. Modern financial
markets are deeply interconnected with the macroeconomic environment, especially in emerging
markets such as Pakistan. Recognizing this, our research advances the dynamic hedging literature
by incorporating macroeconomic variables specifically, GDP growth, inflation, and monetary
policy rates directly into the deep learning -based FT -Net Hybrid model. This extension is a
deliberate response to empirical and theoretical evidence indicating that these broader economic
factors exert significant influence on asset volatility, risk premia, and optimal hedge ratios.
Initially, our modeling strategy was centered around extracting the maximum predictive power
from financial price data using both econometric and advanced deep learning methods,
benchmarking them rigorously for hedge effectiveness, risk reduction, out-of-sample robustness
and exploitation of statistical arbitrage opportunities. Among the models tested, the FT -Net
Hybrid emerged as the statistically superior framework due to its d ual ability to learn from both
spectral (frequency-domain) and temporal (time-domain) features of market data. However, the
absence of macroeconomic signals posed a limitation: even the best -performing market-driven
model can fail when underlying economic regimes shift—such as during monetary tightening
cycles or inflationary shocks. Thus, integrating macroeconomic information was not merely a
novel experiment but a logical extension, motivated by both the realities of financial economics
and the practical needs of institutional hedgers.
7.2 Data Foundation: Weekly Market and Macroeconomic
Features
A core pillar of this research is the assembly and engineering of a rich, high -frequency dataset
that merges financial and macroeconomic signals. The weekly dataset comprises:

## Page 113

• KSE-30 Index Cash Price
• Indicative Fair Value of Futures
• Weekly GDP
• Weekly Inflation
• SBP Policy Rate (Weekly Rate)
The methodology for constructing high -frequency proxies for GDP and inflation, and aligning
all variables on a weekly grid, is ful ly detailed in Appendix B. This appendix explains the
collection, interpolation, and quality assurance processes for exogenous variables, ensuring
reproducibility and transparency.
Feature engineering is then performed to unlock the predictive relationship s among these
variables. Besides calculating weekly log returns for both the cash index and futures contract, we
introduce lagged versions of these returns to enable the model to learn temporal dependencies.
Macroeconomic variables are standardized (z -scored) and likewise lagged, based on the
hypothesis that their impact on financial markets is both immediate and delayed.
7.3 Feature Relationships: Empirical Insights from the Data
To understand the interaction structure within the engineered dataset, we computed and
visualized the correlation matrix (see Figure 32). This heatmap presents the pairwise correlations
Figure 32: Correlation Matrix

## Page 114

between financial returns, their lags, and all macroeconomic variables (both contemporaneous
and lagged).
Figure 30 s hows modest but non -trivial correlations between macroeconomic lags and market
returns, justifying their inclusion as exogenous predictors. While direct correlations between
returns and exogenous variables may appear modest, the temporal and non-linear interactions are
often undetectable by simple correlation, thus justifying the need for models capable of
uncovering deeper, latent dependencies.
7.4 Exploratory Visualization of Macroeconomic Variables

To contextualize the macroeconomic environment throughout the study period, we visualize the
time series of weekly GDP, inflation, and SBP policy rate (see figure 33). This dual -axis plot
highlights periods of economic contraction, inflationary surges, and changes in monetary policy
stance, all of which are known to affect market risk and hedging needs. It reveals economic
shocks and cycles that coincide with changes in equity market volatility and hedge effectiveness.
Such visualizations demonstrate the motivation for including these variables: market risk is often
a response not only to past prices but also to economic shocks and policy adjustments.

Figure 33: Weekly Series of GDP, Inflation, and SBP Policy Rate

## Page 115

7.5 FT-Net Hybrid Architecture: Integrating Macroeconomic
Information
The FT-Net Hybrid architecture is specifically designed to leverage both frequency-domain and
time-domain signals from the sequence of engineered features (see figure 3 4). The complete
pipeline, as visualized below, seamlessly integrate s macroeconomic exogenous variables at the
input layer, ensuring they are available for both spectral and temporal pattern extraction.
Figure 3 4 shows how macroeconomic exogenous features are embedded alongside financial
features, then jointly processed by spectral (FFT) and temporal (Dense) modules, with outputs
fused to generate the dynamic hedge ratio.
The inputs to the model are windows of 8 weeks, each containing 10 features: 4 market -based
(current and lagged log returns of the index and futures) and 6 macroeconomic-based (z-scored
values and their lags). This design ensures that the network can learn both instantaneous and
lagged economic effects on hedge ratios.

Figure 34: FT-Net Hybrid Model Architecture

## Page 116

7.6 Mathematical Formulation of the FT-Net Hybrid
Let 𝑥𝑡 denote the engineered feature vector at time 𝑡 shown in equation 59:
Equation 59
𝑥𝑡 = [
𝑅𝑖𝑛𝑑𝑒𝑥,𝑡, 𝑅𝑓𝑢𝑡𝑢𝑟𝑒,𝑡, 𝑅𝑖𝑛𝑑𝑒𝑥,𝑙𝑎𝑔1,𝑡, 𝑅𝑓𝑢𝑡𝑢𝑟𝑒,𝑙𝑎𝑔1,𝑡
𝐺𝐷𝑃𝑧,𝑡, 𝐼𝑛𝑓𝑙𝑎𝑡𝑖𝑜𝑛𝑧,𝑡, 𝑃𝑜𝑙𝑖𝑐𝑦 𝑅𝑎𝑡𝑒𝑧,𝑡 ,
𝐺𝐷𝑃𝑧,𝑡−1, 𝐼𝑛𝑓𝑙𝑎𝑡𝑖𝑜𝑛𝑧,𝑡−1, 𝑃𝑜𝑙𝑖𝑐𝑦 𝑅𝑎𝑡𝑒𝑧,𝑡−1 .
]
A rolling window of L=9 steps is stacked to form the input tensor for the neural network . (see
equation 60)
Equation 60
𝑋𝑡 = [𝑋𝑡−𝐿+1, 𝑋𝑡−𝐿+2, … , 𝑋𝑡]
• Spectral Path:
The sequence undergoes a real -valued fast Fourier transform (FFT), separating it into
frequency components:
Equation 61
𝐹𝑡 = ℱ(𝑋𝑡)
Both real and imaginary parts are concatenated and fed into dense layers to extract cyclic
and periodic information.
• Temporal Path:
The sequence is processed by two fully connected dense layers with dropout, learning
non-linear temporal interactions between features.
The outputs of both modules are concatenated and flattened, then passed through a final dense
layer with sigmoid activation to estimate the time-varying dynamic hedge ratio 𝛽𝑡:

## Page 117

Equation 62
𝛽𝑡 = 𝜎(𝑊𝑇[𝑆𝑝𝑒𝑐𝑡𝑟𝑎𝑙 𝐹𝑒𝑎𝑡𝑢𝑟𝑒𝑠, 𝑇𝑒𝑚𝑝𝑜𝑟𝑎𝑙 𝐹𝑒𝑎𝑡𝑢𝑟𝑒𝑠] + 𝑏)
The hedged return at each time t is computed using equation 34.
7.7 Model Training, Filtering, and Statistical Cleanliness
The model is trained using mean squared error (MSE) loss on the hedged return, with an auxiliary
output for 𝛽𝑡 itself. To address the residual structure, particularly autocorrelation and conditional
heteroskedasticity, we employ a LightGBM gradient boosting filter on the out-of-sample hedged
returns. This post-processing step, mathematically represented as:
Equation 63
𝑅̂ ℎ𝑒𝑑𝑔𝑒𝑑,𝑡 = 𝑅ℎ𝑒𝑑𝑔𝑒𝑑,𝑡 − 𝑔̂ (𝑅ℎ𝑒𝑑𝑔𝑒𝑑,𝑡−1, 𝑅ℎ𝑒𝑑𝑔𝑒𝑑,𝑡−2 … 𝑅ℎ𝑒𝑑𝑔𝑒𝑑,𝑡−5)
removes any lingering serial dependence, as confirmed by diagnostic tests such as the KPSS,
Ljung-Box, and ARCH LM.
7.8 Performance Assessment and Statistical Validation
7.8.1 Training and Validation Loss Curves
Figure 35: Convergence of Training and Validation Loss

## Page 118

Rapid convergence of training and validation loss demonstrates efficient learning and low
overfitting risk in the FT-Net Hybrid model. (see figure 35)
7.8.2 MVHR Time Series
In figure 36, dynamic hedge ratio ( 𝛽𝑡) is smoothly estimated in both training and test periods,
exhibiting regime sensitivity. The distinction between in-sample (blue) and out-of-sample (cyan)
periods is clear and continuous, indicating robust generalization.
7.8.3 Cumulative Returns: Hedged vs. Unhedged

Figure 36: Dynamic MVHRs Using FT-Net Hybrid Model with Macroeconomic Variables
Figure 37: Hedged Returns in Comparison with Unhedged Returns

## Page 119

As seen in figure 3 7, the hedged return series is significantly less volatile than the unhedged
series, particularly during economic shocks (e.g., early 2020 COVID crash), confirming the
model’s success in risk minimization.
7.8.3.1 Model Metrics and Diagnostics
The performance metrics and diagnostic results clearly illustrates the effectiveness and
robustness of the FT-Net Hybrid model with macroeconomic variables in hedging equity futures
risk. The statistical diagnostics, which include the KPSS, Ljung -Box, and ARCH LM tests, are
all successfully passed for both the training and testing datasets. This confirms that the filtered
hedged return series is stationary, devoid of significant autocorrelation, and free from conditional
heteroskedasticity, which are essential requirements for reliable risk measurement and model
validity.
Looking at the quantitative metrics, the model achieves a remarkably high variance reduction—
99.99% for the training set and 99.30% for the test set —demonstrating its capacity to almost
entirely neutralize the volatility of the unhedged returns. The mean absolute error (MAE) and
root mean squared error (RMSE) are both extremely low, indicating precise tracking of the
minimum variance hedge ratio. The model also shows strong performance in tail r isk reduction,
with EVT -based Value at Risk (VaR) and Conditional Value at Risk (CVaR) reductions
exceeding 93% in both training and test sets, further supporting its utility in mitigating extreme
loss events.
The directional accuracy, which reflects the model’s ability to correctly predict the sign of future
returns, is higher in the test set (71.01%) than in the training set (59.75%), indicating robust out-
of-sample predictive ability. While the Sharpe ratio is slightly negative in the test set, this is a
common outcome for minimum variance hedging models whose primary aim is risk
minimization rather than profit maximization. Overall, the Theil U statistic, which is close to 1,
reinforces the model’s strong forecasting performance. The computational effici ency is also
notable, with the complete training and evaluation process taking just over 41 seconds.
These results provide compelling evidence that the FT -Net Hybrid model, when enhanced with
macroeconomic exogenous variables, delivers exceptional hedging performance and statistical
reliability, making it highly suitable for real-world application in risk management.

## Page 120

7.8.4 Economic and Practical Interpretation
FT-Net Hybrid makes its hedging recommendations in response to regularly observed and
anticipated conditions in the monetary policy and economic performance by directly embedding
the weekly GDP, inflation and SBP policy rate (standardized and lagged) into the deep learning
pipeline. This means that the model will automatically adjust the hedge ratio in resp onse to a
detection of monetary tightening or inflation shock to undermine an increase in market risk. As
shown in figure 33, the peak inflation and changes in SBP policy rate increase volatility i.e.
macroeconomic vector on account of virtue of correlation.
The changes in the dynamic minimum variance hedge ratio (MVHR) time series, as depicted in
figure 36, suggest that the model can identify regime shifts. Inflation spikes or fast rate hikes go
up or down, which in turn stabilizes the hedged portfolio’s risk profile as seen with the rise and
fall of 𝛽𝑡.
The illustration of cumulative returns (Figure 37) highlights the practical impact on market stress
periods, where the first remains almost flat while the second line goes through some large
drawdowns. This evidence suggests that the model can be utilized as it is beneficial for
institutional investors who need stability and effective risk mitigation, with regime changes.
7.9 Summary
The application of FT -Net Hybrid to a macroeconomic context is shown in this chapter to be
methodologically sound and empirically effective. We create a hedging model that is robust to
economic shocks and policy regime changes by embedding macroeconomic information at a
weekly frequency with financial features in a unified modelling framework. The model performs
exceptionally well statistically, as it reduces variance by over 99%, has high directional accuracy
and the diagnostic tests are clean. This proves the approach to be robust. This paper, therefore,
contributes to the practical and academic frontiers of dynamic hedging, making a good case for
including macroeconomic signals in the next-generation financial risk hedging models.

## Page 121

8 Conclusion

8.1 Summary
This study offers a thorough analysis of dynamic hedging on equity futures and the identification
of statistical arbitrage opportunities for the traditional econometric models and deep learning
models. They focus on the KSE -30 Index and its futures contracts for Pakistani Stock market
aiming to construct dynamically adaptive hedging strategies, to estimate risk properly and
potentially to detect and exploit profitable market inefficiencies. The major contributions of this
research is the implementation, comparison and empirical evaluation of hedging models
subjected to realistic market constraints, followed by an extensive multi -criteria performance
evaluation and arbitrage-based trading analysis.
During data collection and pre -processing phase, one of the key obstacle in our study was the
incoherence of futures price series resulting from the rollover of these contracts. In the Pakistani
futures market, a futures contract has a month-end expiration date where a jump from the end of
one month to the beginning of the next month creates artificial jumps in the price series due to
the move from one expiring frequency to another. These discontinuities come about because the
expiring contract, by virtue of its imminent maturity, approximates more closely to the spot price,
while the new contract, having a longer time to maturity, embodies additional risk premia and
time values. Such structural breaks are another key problems for purposes of time series
modeling, especially for volatility estimation and hedge ratio determination, as they add noise
that is not related at all with market fundamentals. Therefore, mean-reverting technique was used
to achieve smooth and consistent futures returns. This method involved taking the average of the
expiring contract and roll-over contract price. The discontinuity magnitude at the rollover point
was then compared to this average value. The differential between this was calculated and that
value was added, or subtracted as app ropriate, equally to the prices of the new contracts
applicable to that month. This shift brought the entry point of new contracts in line with the old
series and eliminated artificial discontinuities. It was done successively for each rollover
processes in the whole time series, hence guaranteeing the accuracy of hedge ratio estimation.
Furthermore null values in interest rate -related features were interpolated by a Random Forest
Regression model. This was chosen due to its ability in modeling complex non-linear interactions

## Page 122

and to minimize overfitting. Interpolation validity was assessed with R² and RMSE values. This
showed that the model had a good predictive power which ensured the reliability of our database.
Moreover, prior to the implementation of th e mathematical models, a series of pre -estimation
tests were conducted on DCC-GARCH and Copula DCC-GARCH frameworks. These tests were
necessary because of the structural assumptions of GARCH -family models and for meeting the
statistical requirements for it s implementation. The Augmented Dickey -Fuller (ADF) test was
used to check the stationarity of time series data, which is a prerequisite for estimating GARCH.
ADF test values suggested that the log returns of KSE-30 index and its associated future series
were stationary, and, thus, proved suitable for GARCH modeling. Additionally, Engle’s ARCH-
LM test was conducted to check for heteroscedasticity in the return series. The findings
supported the presence of autoregressive conditional heteroscedasticity, therefore justifying the
use of DCC-GARCH and Copula DCC-GARCH to capture the volatility. This set of tests were
a means of confirming the statistical reliability of the econometric models prior to full
specification.
Therefore, in the first stage of the study , the traditional econometric models, i.e. the DCC -
GARCH model introduced by Engle (Engle R. , 2002) , was applied. This model is used to
characterize the volatility and dynamic correlations between index and futures returns. It is based
on ARMA-eGARCH specifications of the marginal distributions, thereby effectively capturing
volatility clustering and leverage effects. Although DCC-GARCH provides a dynamic modeling
approach for time -varying correlations, it is based on the nor mality assumption, which is
inadequate for capturing tail dependence. This drawback was overcome by extending the model
with the incorporation of a Student-t Copula, which led to the Copula DCC-GARCH model. The
copula-based probability integral transformat ions is applied to the standardized residuals from
the marginal models, which allows the hybrid econometric specification to model symmetric tail
dependence, hence providing a better representation of extreme co -movements. This
improvement was especially effective under stress, as it introduced better risk modelling than its
Gaussian-based equivalent.
In parallel to these traditional methods, this research adopted two deep learning models that can
capture nonlinear patterns and intricate relationships of th e hedge ratio dynamics. The LSTM –

## Page 123

CNN Hybrid Model appended convolutional neural networks as the front end to capture short -
term features of raw returns and futures spread data. It subsequently stacked with Long Short -
Term Memory (LSTM) layers to estimate l ong-range sequential behavior. This architecture
performed well in learning from noisy time series data and incorporated the ability to adjust to
varying market conditions particularly during extreme volatility.
The most novel model proposed in our work was the FT-Net Hybrid Model that utilized Fourier
Transform based spectral decomposition along with temporal convolutional and recurrent layers.
The Fourier modules were designed to capture the hidden cyclical pattern and long -range
memory effect in the time series which are usually ignored in the time domain -based models.
After the data was transformed into the frequency domain, convolutional blocks were used to
capture evolving dependencies, and then the sequence dynamics were preserved using LSTM or
Transformer layers. By utilizing this hybrid model, we let the model learn from the spectral
domain representations, which gradually improved its capacity to catch the data-driven structural
shifts, recurrent patterns, and improve the overall hedging accuracy.
Therefore, after implementing both mathematical and deep learning models, we verify that they
meet their respective requirements by performing and interpreting the post model statistical
diagnostic tests. The results of all tests including ARCH -LM, KPSS, Lj ung Box, and Theil U
Statistics, shows that the forecasted values are accurate and the residuals of all models are
stationary, free from autocorrelation, and conditional heteroscedasticity. These validations are
not only procedural, but also essential for supporting our research questions. Therefore, we can
now positively state that all of the models are fit for comparisons, and it is appropriate to apply
such models evaluating the hedging effectiveness and statistical arbitrage opportunities.
Moreover, a thorough model diagnostic was performed with statistical tools like ACF, indicating
statistically significant short -term memory in futures return, which are the main drivers of
memory-driven models like FT-Net and LSTM-CNN. The dynamic hedge ratios of each model
were visually and statistically demonstrated. The results indicated that the deep learning models,
especially FT-Net, contribute to more reactive and less erratic hedge ratios under the volatility
switching regimes.

## Page 124

In the second phase, a Fuzzy TOPSI S methodology was employed to uniformly compare
performances of all the models for evaluating their hedging effectiveness. This allowed the
models to be ranked based on a vector of nine financial and computational considerations
including variance reduction, RMSE, Sharpe ratio, average hedged return, directional accuracy,
Mean Absolute Deviation (MAD), time complexity, tail metrics i.e. Value -at-Risk (VaR) and
Conditional VaR (CVaR) using Extreme Value Theory (EVT). The criteria was converted to
fuzzy linguistic values, normalized and weighted to obtain the relative closeness of the models
to the ideal solution. This produced clear results for both in -sample and out-of-sample analysis,
ranking Ft-Net Hybrid model as the best performer among all models. This model showed highest
reduction in portfolio variance, smallest RMSE in hedge ratio forecast, and highest directionally
accuracy. The model also employed much better Sharpe ratios and diminished extreme tail risks
more efficiently than other models. Although the Copula DCC-GARCH model can perform well
in certain extreme market scenarios, it was not sufficiently flexible in the dynamic circumstances
when compared with the deep learning models. The LSTM -CNN combination also did obtain
good results, but underperformed FT-Net in terms of hedged returns.
In the third stage of the study, the concept of statistical arbitrage was investigated by determining
whether hedge ratios produced by the models could be used to identify and profit from temporary
mispricing between the spot index and the futures. We found that the spread signals from the FT-
Net hybrid model were significantly mean -reverting in nature. Then we applied them to build
the long-short strategies with deviation thresholds and rolling window measurements. The returns
of these strategies were evaluated and once more, the FT -Net Hybrid stood out as the most
effective model to yield efficient and economically significant arbitrage opportunities.
As an extension to our research, we also incorporated macroecon omic aspects into the FT -Net
framework. Weekly GDP, Inflation and SBP Policy Rate were incorporated as exogenous
features in a macro-informed FT-Net model. The exogenous variables were selected based on
how predictive they are using extensive feature engin eering which includes lagging,
normalization, and correlation analysis. This model was trained on weekly returns data which
showed significant improvements in the model and demonstrated hedging stability, especially
when there was macroeconomic uncertainty . This illustrated the model’s potential for

## Page 125

responding to exogenous variables and incorporating them into risk management and arbitrage
considerations.
Overall, this study has made novel contributions to the literature on risk management and
applications of deep learning in financial markets. The comparative results exhibit the
dominance of the hybrid deep learning models to the standard econometric models in
dynamically estimating the hedge ratios and discovering arbitrage opportunities. The FT -Net
Hybrid, among all tested models, had the best financial performance, scalability, and adaptability
to macroeconomic inputs.
8.2 Recommendations for Future Work
While this study offers a thorough assessment of dynamic hedge effectiveness and statistical
arbitrage with traditional and deep learning approaches, there remains several opportunities for
further research in this field. Therefore, based on the results and novel approaches that we provide
in this paper, future research could be done to address the shortcoming s, enhance the model’s
predictive capacity, and modify the modelling framework for broader financial applications.
The first recommended extension of this work would be to apply a multivariate asset framework
where hedge ratios would be calculated for a di versified portfolio of assets, rather than for an
index. This would enable portfolio managers and researchers to capture the cross -asset
relationships, sectoral dependencies, and correlations between multiple equity instruments. By
doing so, deeper understanding of portfolio-level hedging under dynamic and nonlinear settings
may be established using models such as multivariate DCC-GARCH, vine copulas, or attention-
based transformer networks.
Secondly, future work can include higher frequency data. As this study is limited by the nature
of the accessible data and market liquidity, due to which we examined only on daily and weekly
frequencies, however, modeling at finer intraday frequencies could provide more refined and
timelier measures of volatility dynamic s and arbitrage signals. Using tick or minute level data,
especially when using deep reinforcement learning may enhance model adaptability for high -
frequency trading (HFT) or intraday hedging. This, however, would depend on the existence of

## Page 126

efficient data processing pipelines, low latency set -up and careful tuning for microstructural
noise.
Furthermore, improving the macroeconomic integration features can provide better results in
future research. Although the present paper has introduced macro variables in cluding GDP,
inflation and policy rate, it is promising to consider a more comprehensive set of economic series
such as industrial production, foreign exchange reserves, fiscal balance, global market -related-
variables including crude oil price, US treasury yield and international volatility indices.
Therefore, to capture the behavior of market shifts and macro-financial regimes more effectively,
time-varying parameter models, such as Bayesian frameworks or interpretable deep learning
structures such as Temporal Fusion Transformers (TFTs) may be used.
Additionally, this research considers transaction costs to be relatively constant when testing
profitability of arbitrage opportunities. In future work we recommend exploring the possibility
of adopting dynamic transaction costs involving the liquidity state, bid-ask spread, slippage, and
execution delay. This would make it possible to evaluate arbitrage strategies in a better way,
especially where the market is less liquid or is developing. Moreover, adding regulatory
constraints like short -sale restrictions, leverage caps, or margin requirements would give more
support to the above arbitrage strategies.
Finally, in order to enable innovation in future work, we recommend efforts to develop modular
and reusable code libraries or APIs that encapsulate the hedging and arbitrage models we have
made available in this work. These could potentially be released for other researchers, financial
analysts, and developers who could then test them across other marke ts, asset classes or time
periods. A system combined with real -time data and cloud -based technology can learn about a
market's evolving structure over time and improve its hedging performance.

## Page 127

9 References

Adams, S. O., Asemota, O. J., & Ibrahim, A. A. (2024). Asymmetric GARCH Type Models and LSTM
Afor Volatility Characteristics Analysis of Nigeria Stock Exchange Returns. American Journal
of Mathematics and Statistic, 17-32.
Agram, N., Oksendal, B., & Rems , J. (2024). Deep learning for quadratic hedging in incomplete jump
market. Digital Finance, 6(3), 463-499.
Alshenawy, F. Y. (2024). Using Tail Dependence on Copula-based Regression Models in Mixed Data.
The Egyptian Statistical Journal , 68(2), 65-85.
Alvarez-Diez, S., & Gonzalez, M. (2006). Optimal hedge ratios for the Mexican stock market index
futures contract: A multivariate GARCH approach. Applied Financial Economics, 931–943.
Baba, Y., Engle , R. F., Kraft, D., & Kroner, K. F. (1990). Multivariate simultaneous generalized
ARCH. Department of Economics, Unuveristy of California, San Diego.
Baillie, R. T., & Myers, R. J. (1991). Bivariate GARCH estimation of the optimal commodity futures
hedge. Journal of Applied Econometrics, 109-24.
Banerjee, A., Dolado, J. J., Galbraith, J. W., & Hendry, D. (1993). Co-integration, error correction, and
the econometric analysis of non-stationary data. Oxford university press.
Beuhler, H., Gonon, L., & Wood, B. (2019). Deep hedging. Quantitative Finance, 19(8), 1271-1291.
Bodla, B., & Jindal, S. (2006). The hedging effectiveness of stock index futures: Evidence for the S&P
CNX Nifty index traded in India. IUP Journal of Derivatives Markets, 36–48.
Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. Journal of
Econometrics, 307–327.
Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. Journal of
Economics, 307-327.
Bollerslev, T. (1990). Modelling the coherence in short run nominal exchange rates: Amultivariate
generalized ARCH model. The Review of Economics and Statistics, 498-505.
Bollerslev, T. (1990). Modelling the Coherence in Short-Run Nominal Exchange Rates: A Multivariate
Generalized ARCH Model . The Review of Economics and Statistics, 498-505.
Brigola, R. (2025). Further Applications of the Fourier Transform. In Fourier Analysis and
Distributions: A First Course with Applications (pp. 383-436). Cham: Springer Nature
Switzerland.
Chen, W., Hussain, W., Cauteruccio, F., & Zhang, X. (2023). Deep learning for financial time series
prediction: A state-of-the-art review of standalone and hybrid models. CMES-Computer
Modeling in Engineering and Sciences.
Demarta, S., & McNeil, A. J. (2005). The t Copula and Related Copulas. International Statistical
Review, 73(1), 111-129.
Dicky , D. A., & Fuller , W. A. (1979). Distribution of the estimators for autoregressive time series with
a unit root. Journal of the American Statistical Association, 74(366), 427–431.
Ederington, L. H. (1979). The hedging performance of the new futures markets. The Journal of Finance,
157-170.
Engle, R. (2002). Dynamic Conditional Correlation: A Simple Class of Multivariate Generalized
Autoregressive Conditional Heteroskedasticity Models. Journal of Business & Economic
Statistics, 339-350.
Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of
United Kingdom inflation. Econometrica, 50(4), 987–1007.
Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of
United Kingdom Inflation. Econometrica, 987–1007.
Floros, C., & Vougas, D. (2007). Measuring minimum variance hedging effectiveness: Traditional vs.
sophisticated models. Managerial Finance, 686–697.
François, P., Gauthier, G., Godin, F., & Mendoza, C. O. (2025). Is the Difference between Deep

## Page 128

Hedging and Delta Hedging a Statistical Arbitrage? Finance Research Letters.
Friedman, M. (1989). The Quantity Theory of Money – A Restatement. In M. Friedman, Money (pp. 1-
40). London: Palgrave Macmillan UK.
Gholamy, A., Kreinovich, V., & Kosheleva, O. ( 2018). Why 70/30 or 80/20 relation between training
and testing sets: A pedagogical explanation. International Journal of Intelligent Technologies
and Applied Statistics, 105-111.
Ghosh, A. (1993). Hedging with Stock Index Futures: Estimation and Forecasting with Error Correction
Model. Journal of Futures Markets, 743-743.
Hafner, C. M., & Manner , H. (2012). Dynamic stochastic copula models: Estimation, inference and
applications. Journal of applied econometrics, 27(2), 269-295.
Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-
1780.
Horikawa, H., & Nakagawa, K. (2024). Relationship between deep hedging and delta hedging:
Leveraging a statistical arbitrage strategy. Finance Research Letters.
Hsu, C.-C., Tseng, C.-P., & Wang, Y.-H. (2008). Dynamic Hedging with Futures: A Copula-Based
GARCH Model. Journal of Futures Markets, Forthcoming, 34.
Hu , Y., & Ni, J. (2024). A deep learning‐based financial hedging approach for the effective
management of commodity risks. Journal of Futures Markets, 44(6), 879-900.
Hu, Y., & Ni, J. (2024). A deep learning-based financial hedging approach for the effective management
of commodity risks. Journal of Futures Markets, 877-1094.
Jondeau, E., & Rockinger, M. (2006). The copula-garch model of conditional dependencies: An
international stock market application. Journal of international money and finance, 25(5), 827-
853.
Khan , S. U. (2006). Role of the Futures Market on Volatility and Price Discovery of the Spot Market:
Evidence from Pakistan’s Stock Market. The Lahore Journal of Economics, 107-121.
Khan, S. U., & Abbas, Z. (2013). Does Equity Derivatives Trading Affect the Systematic Risk of the
Underlying Stocks in an Emerging Market Evidence from Pakistan’s Futures Market. Lahore
Journal of Economics.
Knoll, P. W. (2023). Deep mean-variance Hedging using LSTM RNNs . (Doctoral dissertation,
Technische Universität Wien).
Koutmos, G., & Tucker, M. (1996). Temporal relationships and dynamic interactions between spot and
futures stock markets. The Journal of Futures Markets, 55-69.
Kumar, S., Rao, A., & Dhochak, M. (2025). Hybrid ML models for volatility prediction in financial risk
management. International Review of Economics & Finance, 103915.
Lai, Y. H., Chen, C. W., & Gerlach, R. (2009). Optimal Hedging via Copula-Threshold-GARCH
Models. Mathematics and Computers in Simulation, 79.8, 2609-2624.
Lai, Y. S. (2018). Dynamic hedging with futures: a copula-based GARCH model with high-frequency
data. Review of Derivatives Research, 21, 307-329.
Li, S., Wang, Z., Wang, X., Yin, Z., & Yao, M. (2025). Frequency-enhanced and decomposed
transformer for multivariate time series anomaly detection. Applied Intelligence, 1-18.
Lin, F. L. (2002). A comparative analysis of hedging determination for three alternative international
equity index futures. International Review of Economics & Finance, 213–227.
Liu, P. (2023). Statistical Arbitrage with Hypothesis Testing. In In Quantitative Trading Strategies
Using Python: Technical Analysis, Statistical Testing, and Machine Learning (pp. 225-255).
Berkeley: CA: Apress.
Malik, I. R., & Shah, A. (2016). The Impact of Single Stock Futures on Market Efficiency and
Volatility: A Dynamic CAPM Approach. Emerging Markets Finance and Trade , 339-356.
Nelson, D. B. (1991). Conditional Heteroskedasticity in Asset Returns: A New Approach.
Econometrica, 347-370.
Patton , A. J. (2006). Modelling Assymetric Exchange Rate Dependence. International Economic
Review, 47.

## Page 129

Patton, A. J. (2006). Modelling asymmetric exchange rate dependence. International economic review,
47(2), 527-556.
Qu, H., Wang, T., Zhang, Y., & Sun, P. (2019). Dynamic hedging using the realized minimum variance
hedge ratio approach - Examination of the CSI 300 index futures. Pacific-Basin Finance
Journal.
Reyad, M., Sarhan, A. M., & Arafa, M. (2023). A modified Adam algorithm for deep neural network
optimization. Neural Computing and Applications, 17095-17112.
Sood, K., Pathak, P., & Gupta, S. (2025). How do the determinants of investment decisions get
prioritized? Peeking into the minds of investors. Kybernetes, 2175-2203.
Tahir et. al. (2018). Robust quarterization of GDP and determination of business cycle dates for IGC
partner countries. International Growth Centre.
Ullah, K., Ahsan, M., Hasanat, S. M., Haris, M., Yousaf, H., & Raza, S. F. (n.d.). Short-Term Load
Forecasting: A Comprehensive Review and Simulation Study With CNN-LSTM Hybrids
Approach.
Volpatti, G. (2024). Multi-scale periodic analysis of financial indexes for quantitative financial
forecasts. 1-334.
Wang, J., & Zhang, W. (2019). Dynamic hedging using the realized minimum-variance hedge ratio
approach: Examination of the CSI 300 index futures. Emerging Markets Finance and Trade,
1140-1153.
Xu, R., & Li, X. (2017). Study About the Minimum Value at Risk of Stock Index Futures Hedging
Applying Exponentially Weighted Moving Average - Generalized Autoregressive Conditional
Heteroskedasticity . International Journal of Economics and Financial Issues, 104-110.

## Page 130

Appendix A: Statistical Diagnostic Tests on Residuals Post Model
Implementation

The statistical tests, i.e. ARCH-LM, Ljung Box, and KPSS, were performed on residuals of both
mathematical and machine learning models. The future and spot return series of training and
test data sets were used in the mathematical models. These models are represented as: DCC -
GARCH (MM1), and Copula DCC GARCH (MM2). The results of these tests were obtained
from R programming language. Whereas, hedged return series were used in the machine
learning models, with results of each dataset computed in python. The p -values of these tests
are below,

Table 5: Statistical Tests for Residuals on Training Data – Mathematical Model

Table 6: Statistical Tests for Residuals on Testing Data – Mathematical Model

Table 7: Statistical Tests for Residuals – LSTM-CNN

Table 8: Statistical Tests for Residual – Ft-Net Hybrid

Test Type MM1 (Ft) MM1 (St) MM2 (Ft) MM2 (St)
ARCH LM 0.06452 0.05857 0.06754786 0.81109125
Ljung-Box 0.8779 0.8253 0.9279456 0.9302992
KPSS 0.1 0.1 0.1 0.1
Test Type MM1 (Ft) MM1 (St) MM2 (Ft) MM2 (St)
ARCH LM 0.1427 0.1687 0.1427 0.1687
Ljung-Box 0.3071 0.2302 0.3071 0.2302
KPSS 0.1 0.1 0.1 0.1
Test Type Training p-value Testing p-value
KPSS Test 0.0692 0.1000
Ljung-Box (10) 0.5791 0.9986
ARCH LM 1.0000 1.0000
Test Type Training p-value Testing p-value
KPSS Test 0.1000 0.1000
Ljung-Box (10) 0.6321 0.9911
ARCH LM 1.0000 1.0000

## Page 131

Appendix B: Data Collection and Processing of Macro Economic
Variables
In order to enhance the predictive power and macroeconomic relevance of the top -performing
hedging model (as determined by the fuzzy TOPSIS methodology), supplementary
macroeconomic variables were incorporated alongside the primary data on the KSE -30 index
and its associated futures contracts. The macroeconomic variables utilized in this research
include real GDP growth, the State Bank of Pakistan (SBP) policy rate, and the weekly Sensitive
Price Index (SPI) as a proxy for inflation. The data for these variables were sourced from official
repositories, specifically the State Bank of Pakistan and the Pakistan Bureau of Statistics.
A significant methodological challenge was reconciling the difference in data frequency between
macroeconomic variables and financial market data. While the index and futures data were
available on a daily and weekly basis, macroeconomic indicators such as GDP and polic y rates
are generally published at lower frequencies i.e., quarterly for GDP and annually for policy rates.
Therefore, interpolation procedures were undertaken to harmonize the frequency of these
variables with that of the financial dataset.
Interpolation of weekly GDP using Quarterly GDP
To interpolate the weekly GDP from quarterly GDP we took inspiration from the methodology
used by (Tahir et. al., 2018) for the quarterization of annual GDP. In his methodology, he used
quarterly macroeconomic variables to extract a common orthogonal factor to include the business
cycle dynamics and use a time series equation to disaggregate the annual GDP into quarterly
frequency. Therefore, replicating this methodology and using the theoretical basis and equation
of the Quantity Theory of Money as discussed by (Friedman, 1989) we disaggregated the
quarterly GDP into weekly GDP. The following i s the foundation of the methodology we’ve
used.

## Page 132

The theoretical basis developed using the Quantity Theory of Money
The quantity theory of money is a classical economics theory explaining the relationship between
the money supply and price level in an ec onomy. We started by using the basic equation that
implies this relationship. (equation 66) Here for simplicity, we assume that the velocity of money
is constant.
Equation 64
𝑀. 𝑉 = 𝑃. 𝑌
Where M represents the money supply, V is the velocity of money, P is the price level and Y
represents the real output or real GDP.
However, here to focus on our objective we assume that at week 0 the equation can be written as
equation 67 below:
Equation 65
𝑀0𝑉 = 𝑃0𝑌0
Subsequently, at week 1, i.e., for any week in the future, the equation can be written as:
Equation 66
𝑀1𝑉 = 𝑃1𝑌1
Dividing both the equations we get:
Equation 67
𝑀1
𝑀0
= 𝑃1
𝑃0
. 𝑌1
𝑌0

Let
𝑀1
𝑀0
be defined as the money supply index denoted by MI and
𝑃1
𝑃0
be defined as the price index
denoted by PI. Therefore, we can conclude by rewriting the equation that the weekly GDP at
time t equals the product of the GDP at t = 0 and the money supply index divided by the price
index. (see equation 70)
Equation 68
𝑌1 = 𝑀𝐼
𝑃𝐼 . 𝑌0
Here, MI and PI are constructed using available high-frequency proxies, including weekly SPI

## Page 133

data to reflect inflationary movements (PI) and relevant money supply aggregates (MI), which
collectively anchor the weekly GDP estimate to real macroeconomic movements. The figure 38
presents the results of the weekly gdp interpolation.
Weekly SPI and Policy Rate Frequency Conversion
The weekly SPI, already available at the required frequency from the Pakistan Bureau of
Statistics, was directly incorporated as the inflation measure. The SBP policy rate, however, is
typically reported at an annual frequency. To align this variable with the weekly financial
dataset, the annual policy rate was systematically converted to a w eekly equivalent. This was
achieved using standard frequency conversion formulas, analogous to those employed when
adjusting interest rates from annual to quarterly or monthly bases in macro -financial analyses.
Specifically, the annual rate was transformed into an equivalent weekly rate using the following
mathematical transformation:
Equation 69
𝑟_𝑤𝑒𝑒𝑘𝑙𝑦 = (1 + 𝑟_𝑎𝑛𝑛𝑢𝑎𝑙)^(1/52) − 1
where r_annual denotes the annual policy rate, and r_weekly represents its weekly equivalent.
This approach preserves the compounding characteristics of the rate over a year, ensuring
theoretical and empirical consistency with the high -frequency financial data employed in
subsequent modeling.
Figure 38: Weekly Interpolated GDP and M2 Money Supply

## Page 134

_No extractable text on this page._

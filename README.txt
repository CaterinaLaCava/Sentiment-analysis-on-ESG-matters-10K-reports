Report.pdf: 
-detailed report of what we did in our project.

Folder ResultsPolarity: 
-sentiment_evolution19_22: excel file with the evolution of the ESG-related average compound score using polarity approach.
-sentiment_evolution2022: excel file with the ESG-related average compound score of 2022 using polarity approach.
-evolution_bank/car/energy/food: plot of the evolution of companies' average compound score industry by industry 
-words_/bank/car/energy/food: word clouds of positive and negative words that are both in the Loughran-McDonald list and in the ESG-related reports.
-change_per_company: percentage change from 2019 to 2022 of companies' average compound score
-change_per_industry: table with the average change within every industry of the percentage change from 2019 to 2022 of companies' average compound score

Folder ResultsVADER: 
-sentiment_evolution19_22: excel file with the evolution of the ESG-related average compound score using VADER approach.
-sentiment_evolution2022: excel file with the ESG-related average compound score of 2022 using VADER approach.
-evolution_bank/car/energy/food: plot of the evolution of companies' average compound score industry by industry 
-change_per_company: percentage change from 2019 to 2022 of companies' average compound score
-change_per_industry: table with the average change within every industry of the percentage change from 2019 to 2022 of companies' average compound score
-evolution_all_companies: plot of the evolution of all companies' average compound score 
-sentiment_per_document2022: companies' 2022 compound score resulting from applying VADER to the entire ESG-reports instead of doing in sentence by sentence
-sentiment_statistics2022: mean and standard deviation (both per industry and in general) of companies' 2022 average compound score

Folder ResultsBERT: 
-Sentimental_Bert_on_10K: csv file with the evolution of the ESG-related score using BERT approach.
-evolution_bank/car/energy/food: plot of the evolution of companies' average compound score industry by industry 
-change_per_company: percentage change from 2019 to 2022 of companies' average compound score
-change_per_industry: table with the average change within every industry of the percentage change from 2019 to 2022 of companies' average compound score
-evolution_all_companies: plot of the evolution of all companies' average compound score 
-sentiment_per_document2022: companies' 2022 compound score resulting from applying VADER to the entire ESG-reports instead of doing in sentence by sentence

Folder 10-K_time_2019-2022:
-collection of 10K-reports from 2019 to 2022 of all the companies selected except Rivian and Royale Energy Inc.

Folder 10-KLast:
-collection of 2022 10K-reports of all the companies selected. 

Folder Lists: 
-economy/solcial/environmental: lists provided by the outhors of the article "Sustainability and Corporate Social Responsibility in the Text of Annual Reports—The Case of the IT Services Industry"
-LM_MasterDictionary : Loughran-McDonald list 

code1.ipynb: 
-complete collection of data and analysis for Vader and polarity approaches
-complete analysis for Bert approach

Folder To_run_code1: 
-everything you need to run code1.ipynb

Sentiment_ecological_Bert.ipynb: 
-to replicate our 3rd approach (Bert to recognise Ecological sentiment) you need this code plus the file ecological_bert (support code);it is possible to change the document in which the analysis is run in ecological_bert/src/predict_historical_10K. Results are in RESULTS folder.

Sentiment_FinBert.ipynb: 
-to replicate our 4th approach (FinBert for sentiment analysis) you need this code. Results are in RESULTS folder.

Select_ESG_sentences_via_Bert.ipynb:
-to replicate the selection of ESG sentences via BERT you need this code. One may need adaptation regarding the path of the 10_k reports within the notebooks. Results are in RESULTS folder.

Folder Data_analysis_summary_code:
-code of Data_analysis
-merged by industry reports(pre-processed and not pre-processed) (8 .txt files to run the code)

Folder Regression:
-folder 1: code for regression and data necessary to run the code
-folder 2: second code for regression (necessary data to run this code are all in Folder To_run_code1)
- Msci scores: necessary to run the two above codes for regressions


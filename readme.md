cs4641_ml_project/adjust_jng_dataset.py: Script used to clean and fix our data for processing  
cs4641_ml_project/Data_processing.ipynb: Script to find what champions were played in what roles from the data and to produce team ranking from pre and post random forest data
cs4641_ml_project/logistic_regression_ranking.py: Code for logistic regression model implementation and player ranking  
cs4641_ml_project/random_forest_role_weights.py: Code for random forest model implementation to find role weights
cs4641_ml_project/Team_ranking_post_random_forest.py: Team ranking output from post random forest data
cs4641_ml_project/Team_ranking_pre_random_forest.py: Team ranking output from pre random foreset data
cs4641_ml_project/.gitignore: git ignore file

cs4641_ml_project/data/: Folder for all our data: cleaned data, split apart data for each role, etc.
cs4641_ml_project/data/champions_per_game.txt: Output textfile containing list of every champion played every game  
cs4641_ml_project/data/champion_roles.txt: Output textfile containing list of champions played and what roles they were played in  
cs4641_ml_project/data/unique_champions.txt: Output textfile of list of unique champions played
cs4641_ml_project/data/congregate_scores.csv: Scores used to compute Random Forest
cs4641_ml_project/data/merged_data.csv: Merged data
cs4641_ml_project/data/post_random_forest.csv: Post Random Forest Performance scores
cs4641_ml_project/data/pre_random_forest.csv: Post Logistic Regression Performance scores


cs4641_ml_project/data/old_data/: Old data that was used before certain dataprocessing methods were implemented  
cs4641_ml_project/data/old_data/cleaned_data.csv: Unsplit cleaned data 


cs4641_ml_project/data/reduced_features_data/: Data currently in use for models  
cs4641_ml_project/data/reduced_features_data/bot_data_new_few_stats.csv: Current Bot data in use  
cs4641_ml_project/data/reduced_features_data/jg_data_new_few_stats.csv: Current Jungle data in use  
cs4641_ml_project/data/reduced_features_data/mid_data_new_few_stats.csv: Current Mid data in use  
cs4641_ml_project/data/reduced_features_data/sup_data_new_few_stats.csv: Current Support data in use  
cs4641_ml_project/data/reduced_features_data/top_data_new_few_stats.csv: Current Top data in use  

cs4641_ml_project/environment/: Coding environment files  
cs4641_ml_project/environment/league_ML_env.yml: Environment file  


cs4641_ml_project/logistic_results/: Logistic regression results and rankings
cs4641_ml_project/logistic_results/logistic_graphs/: Logistic Regression graph results
cs4641_ml_project/logistic_results/logistic_graphs/feature_weights_Bot.png : Bot weighted features
cs4641_ml_project/logistic_results/logistic_graphs/feature_weights_Jungle.png : Jungle weighted features
cs4641_ml_project/logistic_results/logistic_graphs/feature_weights_Mid.png : Mid weighted features
cs4641_ml_project/logistic_results/logistic_graphs/feature_weights_Support.png : Support weighted features
cs4641_ml_project/logistic_results/logistic_graphs/feature_weights_Top.png : Top weighted features

cs4641_ml_project/logistic_results/logistic_graphs/rankings: Player rankings per role
cs4641_ml_project/logistic_results/logistic_graphs/rankings/player_rankings_Bot.csv: Bot Player rankings
cs4641_ml_project/logistic_results/logistic_graphs/rankings/player_rankings_Jungle.csv: Jungle Player rankings
cs4641_ml_project/logistic_results/logistic_graphs/rankings/player_rankings_Mid.csv: Mid Player rankings
cs4641_ml_project/logistic_results/logistic_graphs/rankings/player_rankings_Support.csv: Support Player rankings
cs4641_ml_project/logistic_results/logistic_graphs/rankings/player_rankings_Top.csv: Top Player rankings

cs4641_ml_project/logistic_results/sorted_features_weights.txt: Sorted feature weights after Logistic Regression

cs4641_ml_project/LSTM: LSTM files
cs4641_ml_project/LSTM/checkpoints_positive/ :Pytorch files 
cs4641_ml_project/LSTM/checkpoints_positive/100_Thieveslstm_model.pth

cs4641_ml_project/LSTM/checkpoints_positive/Bilibili_Gaminglstm_model.pth: path file
cs4641_ml_project/LSTM/checkpoints_positive/Dplus_KIAlstm_model.pth: path file
cs4641_ml_project/LSTM/checkpoints_positive/FlyQuestlstm_model.pth: path file
cs4641_ml_project/LSTM/checkpoints_positive/Fnaticlstm_model.pth: path file
cs4641_ml_project/LSTM/checkpoints_positive/Fukuoka_SoftBank_HAWKS_gaminglstm_model.pth: path file
cs4641_ml_project/LSTM/checkpoints_positive/G2_Esportslstm_model.pth: path file
cs4641_ml_project/LSTM/checkpoints_positive/GAM_Esportslstm_model.pth: path file
cs4641_ml_project/LSTM/checkpoints_positive/Gen.Glstm_model.pth: path file
cs4641_ml_project/LSTM/checkpoints_positive/Hanwha_Life_Esportslstm_model.pth: path file
cs4641_ml_project/LSTM/checkpoints_positive/LNG_Esportslstm_model.pth: path file
cs4641_ml_project/LSTM/checkpoints_positive/MAD_Lions_KOIlstm_model.pth: path file
cs4641_ml_project/LSTM/checkpoints_positive/Movistar_R7lstm_model.pth: path file
cs4641_ml_project/LSTM/checkpoints_positive/PSG_Talonlstm_model.pth: path file
cs4641_ml_project/LSTM/checkpoints_positive/T1lstm_model.pth: path file
cs4641_ml_project/LSTM/checkpoints_positive/Team_Liquidlstm_model.pth: path file
cs4641_ml_project/LSTM/checkpoints_positive/Top_Esportslstm_model.pth: path file
cs4641_ml_project/LSTM/checkpoints_positive/Vikings_Esportslstm_model.pth: path file
cs4641_ml_project/LSTM/checkpoints_positive/Weibo_Gaminglstm_model.pth: path file
cs4641_ml_project/LSTM/checkpoints_positive/paiN_Gaminglstm_model.pth: path file

cs4641_ml_project/LSTM/data/: game data for each team with performance scores
cs4641_ml_project/LSTM/data/100_Thieves.csv: game data for team (team name is file name)
cs4641_ml_project/LSTM/data/Bilibili_Gaming.csv: game data for team (team name is file name)
cs4641_ml_project/LSTM/data/Dplus_KIA.csv: game data for team (team name is file name)
cs4641_ml_project/LSTM/data/FlyQuest.csv: game data for team (team name is file name)
cs4641_ml_project/LSTM/data/Fnatic.csv: game data for team (team name is file name)
cs4641_ml_project/LSTM/data/Fukuoka_SoftBank_HAWKS_gaming.csv: game data for team (team name is file name)
cs4641_ml_project/LSTM/data/G2_Esports.csv: game data for team (team name is file name)
cs4641_ml_project/LSTM/data/GAM_Esports.csv: game data for team (team name is file name)
cs4641_ml_project/LSTM/data/Gen.G.csv: game data for team (team name is file name)
cs4641_ml_project/LSTM/data/Hanwha_Life_Esports.csv: game data for team (team name is file name)
cs4641_ml_project/LSTM/data/LNG_Esports.csv: game data for team (team name is file name)
cs4641_ml_project/LSTM/data/MAD_Lions_KOI.csv: game data for team (team name is file name)
cs4641_ml_project/LSTM/data/Movistar_R7.csv: game data for team (team name is file name)
cs4641_ml_project/LSTM/data/PSG_Talon.csv: game data for team (team name is file name)
cs4641_ml_project/LSTM/data/T1.csv: game data for team (team name is file name)
cs4641_ml_project/LSTM/data/Team_Liquid.csv: game data for team (team name is file name)
cs4641_ml_project/LSTM/data/Top_Esports.csv: game data for team (team name is file name)
cs4641_ml_project/LSTM/data/Vikings_Esports.csv: game data for team (team name is file name)
cs4641_ml_project/LSTM/data/Weibo_Gaming.csv: game data for team (team name is file name)
cs4641_ml_project/LSTM/data/paiN_Gaming.csv: game data for team (team name is file name)

cs4641_ml_project/LSTM/positive_data/: game data for each team with performance scores scaled to be positive<br/> 
cs4641_ml_project/LSTM/positive_data/100_Thieves.csv: game data for team (team name is file name) scaled to be positive<br/> 
cs4641_ml_project/LSTM/positive_data/Bilibili_Gaming.csv: game data for team (team name is file name) scaled to be positive<br/> 
cs4641_ml_project/LSTM/positive_data/Dplus_KIA.csv: game data for team (team name is file name) scaled to be positive<br/> 
cs4641_ml_project/LSTM/positive_data/FlyQuest.csv: game data for team (team name is file name) scaled to be positive<br/> 
cs4641_ml_project/LSTM/positive_data/Fnatic.csv: game data for team (team name is file name) scaled to be positive<br/> 
cs4641_ml_project/LSTM/positive_data/Fukuoka_SoftBank_HAWKS_gaming.csv: game data for team (team name is file name) scaled to be positive<br/> 
cs4641_ml_project/LSTM/positive_data/G2_Esports.csv: game data for team (team name is file name) scaled to be positive<br/> 
cs4641_ml_project/LSTM/positive_data/GAM_Esports.csv: game data for team (team name is file name) scaled to be positive<br/> 
cs4641_ml_project/LSTM/positive_data/Gen.G.csv: game data for team (team name is file name) scaled to be positive<br/> 
cs4641_ml_project/LSTM/positive_data/Hanwha_Life_Esports.csv: game data for team (team name is file name) scaled to be positive<br/> 
cs4641_ml_project/LSTM/positive_data/LNG_Esports.csv: game data for team (team name is file name) scaled to be positive<br/> 
cs4641_ml_project/LSTM/positive_data/MAD_Lions_KOI.csv: game data for team (team name is file name) scaled to be positive<br/> 
cs4641_ml_project/LSTM/positive_data/Movistar_R7.csv: game data for team (team name is file name) scaled to be positive<br/> 
cs4641_ml_project/LSTM/positive_data/PSG_Talon.csv: game data for team (team name is file name) scaled to be positive<br/> 
cs4641_ml_project/LSTM/positive_data/T1.csv: game data for team (team name is file name) scaled to be positive<br/> 
cs4641_ml_project/LSTM/positive_data/Team_Liquid.csv: game data for team (team name is file name) scaled to be positive<br/> 
cs4641_ml_project/LSTM/positive_data/Top_Esports.csv: game data for team (team name is file name) scaled to be positive<br/> 
cs4641_ml_project/LSTM/positive_data/Vikings_Esports.csv: game data for team (team name is file name) scaled to be positive<br/> 
cs4641_ml_project/LSTM/positive_data/Weibo_Gaming.csv: game data for team (team name is file name) scaled to be positive<br/> 
cs4641_ml_project/LSTM/positive_data/paiN_Gaming.csv: game data for team (team name is file name) scaled to be positive<br/> 


cs4641_ml_project/LSTM/results_positive/: Results of the Performance Scores for every team
cs4641_ml_project/LSTM/results_positive/100_Thieves_results.png : Performance Score graph
cs4641_ml_project/LSTM/results_positive/Bilibili_Gaming_results.png: Performance Score graph
cs4641_ml_project/LSTM/results_positive/Dplus_KIA_results.png: Performance Score graph
cs4641_ml_project/LSTM/results_positive/FlyQuest_results.png: Performance Score graph
cs4641_ml_project/LSTM/results_positive/Fnatic_results.png: Performance Score graph
cs4641_ml_project/LSTM/results_positive/Fukuoka_SoftBank_HAWKS_gaming_results.png: Performance Score graph
cs4641_ml_project/LSTM/results_positive/G2_Esports_results.png: Performance Score graph
cs4641_ml_project/LSTM/results_positive/GAM_Esports_results.png: Performance Score graph
cs4641_ml_project/LSTM/results_positive/Gen.G_results.png: Performance Score graph
cs4641_ml_project/LSTM/results_positive/Hanwha_Life_Esports_results.png: Performance Score graph
cs4641_ml_project/LSTM/results_positive/LNG_Esports_results.png: Performance Score graph
cs4641_ml_project/LSTM/results_positive/MAD_Lions_KOI_results.png: Performance Score graph
cs4641_ml_project/LSTM/results_positive/Movistar_R7_results.png: Performance Score graph
cs4641_ml_project/LSTM/results_positive/PSG_Talon_results.png: Performance Score graph
cs4641_ml_project/LSTM/results_positive/T1_results.png: Performance Score graph
cs4641_ml_project/LSTM/results_positive/Team_Liquid_results.png: Performance Score graph
cs4641_ml_project/LSTM/results_positive/Top_Esports_results.png: Performance Score graph
cs4641_ml_project/LSTM/results_positive/Vikings_Esports_results.png: Performance Score graph
cs4641_ml_project/LSTM/results_positive/Weibo_Gaming_results.png: Performance Score graph
cs4641_ml_project/LSTM/results_positive/paiN_Gaming_results.png: Performance Score graph


cs4641_ml_project/LSTM/2024_worlds_prediction_positive.txt : Team scores
cs4641_ml_project/LSTM/LSTM.py : LSTM base model
cs4641_ml_project/LSTM/evaluation_metrics_positive.txt : Error metrics
cs4641_ml_project/LSTM/rank_teams.py : Ranking script
cs4641_ml_project/LSTM/train.py : Model application on teams

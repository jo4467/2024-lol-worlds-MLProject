import os
import pandas as pd

with open(os.path.join('LSTM', '2024_worlds_prediction_positive.txt'), 'r') as f:
    lines = f.readlines()

team_scores = {}
for line in lines:
    team, score = line.strip().split(', ')
    team_scores[team] = float(score)

ranked_teams = sorted(team_scores, key=team_scores.get, reverse=True)

ranked_teams_df = pd.DataFrame({'Team': ranked_teams, 'Predicted Score': [team_scores[team] for team in ranked_teams]})
ranked_teams_df.index.name = 'Rank'
ranked_teams_df.index += 1
print(ranked_teams_df)

evaluation_metrics_file = os.path.join('LSTM', 'evaluation_metrics_positive.txt')
evaluation_metrics = pd.read_csv(evaluation_metrics_file)
print(evaluation_metrics.to_string(index=False))
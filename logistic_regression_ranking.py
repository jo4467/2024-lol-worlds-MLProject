import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def logistic_regrssion(df):
    features_df = df.iloc[:, 9:]
    feature_names = features_df.columns
    labels_df = df['result']
    
    X_train, X_test, y_train, y_test = train_test_split(features_df, labels_df, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # model.fit(X_train, y_train)

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_[0],
        'abs_importance': abs(model.coef_[0])
    })
    feature_importance = feature_importance.sort_values('abs_importance', ascending=False)

    y_pred = model.predict(X_test_scaled)

    std_dev_impact = {}
    for feature, coef in zip(feature_names, model.coef_[0]):
        std_dev = df[feature].std()
        prob_change = (np.exp(coef * std_dev) - 1) * 100
        std_dev_impact[feature] = prob_change
    
    impact_df = pd.DataFrame({
        'feature': std_dev_impact.keys(),
        'win_probability_change': std_dev_impact.values()
    }).sort_values('win_probability_change', ascending=False)
    
    return model, feature_importance, impact_df, classification_report(y_test, y_pred)

def plot_feature_importance(feature_importance):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='coefficient', y='feature', data=feature_importance)
    plt.title('Impact of Each Feature on Win Probability')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    return plt

def rank_players(df, model, scaler, feature_cols):

    df_clean = df.fillna(0)
    
    features_scaled = scaler.transform(df_clean[feature_cols])

    scores = np.dot(features_scaled, model.coef_[0])
    
    regional_strength_score = {
        'Fukuoka SoftBank HAWKS gaming': 863,
        'PSG Talon': 863,
        'Gen.G': 1873,
        'Bilibili Gaming': 1826,
        'Hanwha Life Esports': 1873,
        'GAM Esports': 839,
        'Movistar R7': 583,
        'T1': 1873,
        'G2 Esports': 1542,
        'Team Liquid': 1486,
        'FlyQuest': 1486,
        'Vikings Esports': 839,
        'Fnatic': 1542,
        'paiN Gaming': 634,
        'Top Esports': 1826,
        '100 Thieves': 1486,
        'Dplus KIA': 1873,
        'LNG Esports': 1826,
        'Weibo Gaming': 1826,
        'MAD Lions KOI': 1542,
        'Rainbow Warriors': 830,
        'Royal Never Give Up': 1826,
        'Team Whales': 839
    }

    rankings = pd.DataFrame({
        'player': df_clean['playername'],
        'champion': df_clean['champion'],
        'teamname': df_clean['teamname'],
        'league': df_clean['league'],
        'score': scores,
        'games_played': 1
    })

    rankings['score'] = rankings.apply(
        lambda row: row['score'] * regional_strength_score[row['teamname']],
        axis=1)
    
    player_rankings = rankings.groupby(['player', 'teamname']).agg({
        'score': 'mean',
        'games_played': 'sum',
        'champion': lambda x: list(x)
    }).reset_index()

    player_rankings = player_rankings.sort_values('score', ascending=False)
    player_rankings['rank'] = range(1, len(player_rankings) + 1)  
    player_rankings['champion_pool'] = player_rankings['champion'].apply(lambda x: len(set(x)))
    player_rankings['percentile'] = player_rankings['score'].rank(pct=True) * 100
    
    player_rankings['score'] = player_rankings['score'].round(2)
    player_rankings['percentile'] = player_rankings['percentile'].round(1)
    
    cols = ['rank', 'player', 'teamname', 'score', 'percentile', 'games_played', 
            'champion_pool', 'champion']
    return player_rankings[cols]

def get_player_details(df, player_name, feature_cols, scaler):
    player_data = df[df['playername'] == player_name].copy()
    
    avg_stats = player_data[feature_cols].mean()

    scaled_stats = scaler.transform(avg_stats.values.reshape(1, -1))[0]

    stats_df = pd.DataFrame({
        'metric': feature_cols,
        'raw_value': avg_stats,
        'scaled_value': scaled_stats
    })
    return stats_df

def feature_graphs(model, feature_cols, role):
    feature_weights = model.coef_[0]

    sorted_indices = np.argsort(np.abs(feature_weights))
    sorted_features = feature_names[sorted_indices]
    sorted_weights = feature_weights[sorted_indices]

    output_path = "./logistic_results/sorted_features_weights.txt"

    with open(output_path, "a") as file:
        file.write(f"Role: {role}\n")
        for i in range(len(sorted_features)):
            file.write(f"{sorted_features[i]}: {sorted_weights[i]}\n")
        file.write("-" * 40 + "\n")

    plt.figure(figsize=(10, 6))
    colors = ['red' if w < 0 else 'blue' for w in sorted_weights]
    plt.barh(sorted_features, np.abs(sorted_weights), color=colors)
    plt.gca().invert_yaxis()
    plt.xlabel('Weight')
    plt.title(f'Most Impactful Features for {role} Using Logistic Regression')
    
    output_path = f"./logistic_results/logistic_graphs/feature_weights_{role}.png"
    plt.savefig(output_path, bbox_inches='tight')

if __name__ == "__main__":

    data_List = []

    file_path = 'data/reduced_features_data/top_data_new_few_stats.csv'
    data_List.append(("Top", file_path))
    
    file_path = 'data/reduced_features_data/sup_data_new_few_stats.csv'
    data_List.append(("Support", file_path))
    
    file_path = 'data/reduced_features_data/jg_data_new_few_stats.csv'
    data_List.append(("Jungle", file_path))
    
    file_path = 'data/reduced_features_data/mid_data_new_few_stats.csv'
    data_List.append(("Mid", file_path))
    
    file_path = 'data/reduced_features_data/bot_data_new_few_stats.csv'
    data_List.append(("Bot", file_path))

    for (role, file_path) in data_List:
        data = pd.read_csv(file_path)
        features_df = data.iloc[:, 9:]
        feature_names = features_df.columns
        data = data.fillna(0)

        model, feature_importance, impact_df, classification_reporting = logistic_regrssion(data)

        feature_graphs(model, feature_names, role)

        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(data[feature_names])

        rankings = rank_players(data, model, scaler, feature_names)

        rankings.to_csv(f'./logistic_results/rankings/player_rankings_{role}.csv', index=False)

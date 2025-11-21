import pandas as pd

datapath = 'data/jng_data_only.csv'
df = pd.read_csv(datapath)

def shift_data(df, columns):
    positions = df['position'].tolist()
    
    for column in columns:
        values = df[column].tolist()
        new_values = values.copy()
        
        for i in range(0, len(positions), 2):
            if i + 1 < len(positions):
                if positions[i] == 'jng' and positions[i + 1] == 'team':
                    new_values[i] = values[i + 1]
                elif positions[i] == 'jng' and positions[i + 1] == 'jng' and i + 3 < len(positions):
                    if positions[i + 2] == 'team' and positions[i + 3] == 'team':
                        new_values[i] = values[i + 2]
                        new_values[i + 1] = values[i + 3]
        
        df[column] = new_values
    
    return df

columns_to_shift = [
    'firstdragon', 'dragons', 'opp_dragons', 'firstherald', 
    'heralds', 'opp_heralds', 'void_grubs', 'opp_void_grubs', 'firstbaron'
]

df = shift_data(df, columns_to_shift)


df = df[df['position'] != 'team']

df.to_csv('data/adjusted_jng_data_only.csv', index=False)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pandas as pd
df = pd.read_csv('train.csv')

y = df['price'].values
y

df.columns

df.head(10)

scalar = StandardScaler()

# Setting up the neural net classifier
clf = MLPClassifier(alpha=1e-05, hidden_layer_sizes=(10,),
                    random_state=1, max_iter=10)

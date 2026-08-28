# word input -> stemmers [porter, lancaster, snowball] -> comparative analysis

# step 1 and 2
import nltk
import pandas as pd
from tabulate import tabulate

# step 3
porter = nltk.stem.PorterStemmer()
lancaster = nltk.stem.LancasterStemmer()
snowball = nltk.SnowballStemmer('english')

# step 4
inp_list = ["Playing", "played", "player", "studies", "studying", "studies", "happiness", "flying", "flies",
            "connected", "organisation",
            "connection", "connecting"]
porter_op = []
lancaster_op = []
snowball_op = []

for i in inp_list:
    porter_op.append(porter.stem(i))  # step 5
    lancaster_op.append(lancaster.stem(i))  # step 6
    snowball_op.append(snowball.stem(i))  # step 7

out_data = {"Input": inp_list,
            "Porter": porter_op,
            "Lancaster": lancaster_op,
            "Snowball": snowball_op}

out_table_df = pd.DataFrame(out_data)
print(tabulate(out_table_df, headers='keys', tablefmt='psql', showindex=False))  # step 8

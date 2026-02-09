import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("data/pokemon.csv")

print(df.shape)     # (809, 4)
print(df.columns)   # (Name, Type1, Type2, Evolution)
print(df.head())    # Some pokemons don't have an evolution or secondary type


type_counts = {}    # key - type, value - count

for _, row in df.iterrows():    # df.itterows returns (index, row)
    for col in ["Type1", "Type2"]:   
        t = row[col]
        if pd.notna(t):
            t = t.lower().strip()
            type_counts[t] = type_counts.get(t, 0) + 1

types = list(type_counts.keys())
counts = list(type_counts.values())


plt.figure()
plt.bar(types, counts)  # bar chart
plt.xticks(rotation=90) # rotate x-axis CCW labels so type names don't overlap
plt.title("Pokemon Type Distribution")
plt.ylabel("Count")
plt.tight_layout()  # auto adjust spacing
plt.show()

# Water most common type, ice least common type
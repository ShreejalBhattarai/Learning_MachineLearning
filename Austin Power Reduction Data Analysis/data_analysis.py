import pandas as pd
import matplotlib.pyplot as plt

filepath = "Peak_Demand_Reduction.csv"

df = pd.read_csv(filepath)

years = [str(y) for y in range(2007, 2018)]
df.fillna(0,inplace=True)
df.replace(["", " "], 0, inplace=True)


totals = []
for year in years:
    df[year]=df[year].astype(float)
    totals.append(df[year].iloc[1:].sum())

#print(totals)

plt.plot(years, totals, marker='o', color='black', label='Total Reduction')
category_trend = df.groupby("Program Type")[years].sum().T
for col in category_trend.columns:
    plt.plot(category_trend.index, category_trend[col], marker='o', label=col)

# labels and title
plt.title("Austin's Cumulative Reduction of Power Over the Years")
plt.xlabel("Year")
plt.ylabel("Reduction (MW)")
plt.legend()
plt.grid(True)
plt.savefig("Analysis.jpg")



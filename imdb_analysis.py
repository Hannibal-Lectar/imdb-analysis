import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

print("IMDB Analysis")

# loading the data
df = pd.read_csv("imdb.csv")

print(df.head())


# Basic info
print("\nTotal movies:", len(df))
print("Average rating:", round(df["IMDB_Rating"].mean(), 2))


# DATA PREP / ANALYSIS

# Top movies
top = df.sort_values("IMDB_Rating", ascending=True).tail(10)

# Genre
g = df["Genre"].value_counts().head(10)

# Clean for regression
data = df.dropna(subset=["IMDB_Rating", "No_of_Votes"])
X = data[["No_of_Votes"]]
y = data["IMDB_Rating"]

model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)

# Directors
d = df["Director"].value_counts().head(10)

# Year trend
year = df.groupby("Released_Year")["IMDB_Rating"].mean()


# VISUALIZATION

sns.set_theme(style="whitegrid")

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "grid.alpha": 0.3
})

# Top Movies
plt.figure()
sns.barplot(
    x="IMDB_Rating",
    y="Series_Title",
    hue="Series_Title",
    data=top,
    palette="viridis",
    legend=False
)
plt.title("Top 10 Movies by IMDB Rating")
plt.xlabel("Rating")
plt.ylabel("Movie")
plt.tight_layout()
plt.show()

# Genre
plt.figure()
sns.barplot(
    x=g.values,
    y=g.index,
    hue=g.index,
    palette="magma",
    legend=False
)
plt.title("Top Genres")
plt.xlabel("Count")
plt.ylabel("Genre")
plt.tight_layout()
plt.show()

# Rating Distribution
plt.figure()
sns.histplot(df["IMDB_Rating"], bins=15, kde=True, color="skyblue")
plt.title("Distribution of IMDB Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Votes vs Rating
plt.figure()
sns.scatterplot(x=data["No_of_Votes"], y=data["IMDB_Rating"], alpha=0.5)
plt.plot(data["No_of_Votes"], pred, color="red")
plt.title("Votes vs Rating (Regression)")
plt.xlabel("Votes")
plt.ylabel("Rating")
plt.tight_layout()
plt.show()

# Directors
plt.figure()
sns.barplot(
    x=d.values,
    y=d.index,
    hue=d.index,
    palette="coolwarm",
    legend=False
)
plt.title("Top Directors")
plt.xlabel("Number of Movies")
plt.ylabel("Director")
plt.tight_layout()
plt.show()

# Year Trend
plt.figure()
sns.lineplot(x=year.index, y=year.values, marker="o")
plt.title("Average IMDB Rating Over Years")
plt.xlabel("Year")
plt.ylabel("Rating")
plt.tight_layout()
plt.show()


print("\nDone")

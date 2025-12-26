import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('Level-2/Flower_Data.csv')

X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

sns.set(style="whitegrid")
pairplot = sns.pairplot(df, hue='species', markers=["o", "s", "D"])
pairplot.fig.suptitle("Pairwise Relationships of Flower Features", y=1.02)
plt.savefig('Level-2/plots/t2/Flower_pairplot.png', bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='RdYlGn', fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.savefig('Level-2/plots/t2/correlation_heatmap.png', bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix Heatmap')
plt.savefig('Level-2/plots/t2/confusion_matrix.png', bbox_inches='tight')
plt.close()

species_mean = df.groupby('species').mean()
species_mean.plot(kind='bar', figsize=(10, 6))
plt.title('Average Measurements per Species')
plt.ylabel('Measurement (cm)')
plt.xticks(rotation=0)
plt.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('Level-2/plots/t2/species_measurements_bar.png', bbox_inches='tight')
plt.close()

print("All plots have been saved successfully.")
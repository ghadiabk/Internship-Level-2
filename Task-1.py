import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset and seperate columns
df = pd.read_csv('Level-2/House_Data.csv', sep=r'\s+', header=None)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = {}

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
results['Linear Regression'] = {
    'MSE': mean_squared_error(y_test, y_pred_lr),
    'R2': r2_score(y_test, y_pred_lr)
}

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
results['Decision Tree'] = {
    'MSE': mean_squared_error(y_test, y_pred_dt),
    'R2': r2_score(y_test, y_pred_dt)
}

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results['Random Forest'] = {
    'MSE': mean_squared_error(y_test, y_pred_rf),
    'R2': r2_score(y_test, y_pred_rf)
}

results_df = pd.DataFrame(results).T
print(results_df)

results_df.to_csv('Level-2/model_comparison.csv')

results_df = pd.read_csv('Level-2/model_comparison.csv', index_col=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

results_df['MSE'].plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_title('Mean Squared Error (Lower is Better)')
ax1.set_ylabel('MSE')

results_df['R2'].plot(kind='bar', ax=ax2, color='salmon')
ax2.set_title('R-squared (Higher is Better)')
ax2.set_ylabel('R2 Score')

plt.tight_layout()
plt.savefig('Level-2/plots/model_comparison.png')

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest: Actual vs Predicted')
plt.savefig('Level-2/plots/rf_actual_vs_predicted.png')


results_df = pd.read_csv('Level-2/model_comparison.csv', index_col=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

results_df['MSE'].plot(kind='bar', ax=ax1, color=['blue', 'green', 'orange'])
ax1.set_title('Mean Squared Error (Lower is Better)')
ax1.set_ylabel('MSE')
ax1.tick_params(axis='x', rotation=45)

results_df['R2'].plot(kind='bar', ax=ax2, color=['blue', 'green', 'orange'])
ax2.set_title('R-squared (Higher is Better)')
ax2.set_ylabel('R2 Score')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('Level-2/plots/model_comparison.png')

df = pd.read_csv('Level-2/House_Data.csv', sep=r'\s+', header=None)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5, color='darkgreen')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Random Forest: Actual vs Predicted Prices')
plt.grid(True)
plt.savefig('Level-2/plots/rf_actual_vs_predicted.png')
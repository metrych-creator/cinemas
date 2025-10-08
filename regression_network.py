import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import copy
from sklearn.metrics import r2_score

from models import RegressionModel
from tools import plot_loss_function


def train_evaluate(model, epochs, batch_size, optimizer, loss_fn):
    history = []
    batch_start = torch.arange(0, len(X_train), batch_size)

    for epoch in range(epochs):
        train_loss = 0.0

        # --------- TRAIN ---------
        model.train()
        for start in batch_start:
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        avg_train_loss = train_loss / len(X_train) 

        # --------- EVAL ---------
        with torch.no_grad():
            model.eval()
            y_pred = model(X_test)

            test_loss = loss_fn(y_pred, y_test).item()
            test_loss = float(test_loss)

            r2 = r2_score(y_test.cpu().numpy(), y_pred.cpu().numpy())

            history.append({
                "train_loss": avg_train_loss,
                "test_loss": test_loss,
                "r2": r2,
            })

    return history


df = pd.read_pickle('cinemas_cleaned.pkl')
df = df[df.select_dtypes(include=['number']).columns.tolist()]

X_reg = df.loc[:, ['cast_total_facebook_likes','movie_facebook_likes','num_voted_users','duration','actor_1_facebook_likes',
'actor_2_facebook_likes','num_critic_for_reviews','color', 'genres_Drama', 'genres_Horror', 'gross', 'title_year']]
y_reg = df['imdb_score']

X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)



model_reg = RegressionModel(X_train.shape[1])

optimizer = optim.Adam(model_reg.parameters(), lr=0.001)
history = train_evaluate(model_reg, epochs=10, batch_size=10, optimizer=optimizer, loss_fn=nn.MSELoss())


train_loss = [h["train_loss"] for h in history]
test_loss = [h["test_loss"] for h in history]
r2 = [h["r2"] for h in history]

print("MSE: %.2f" % test_loss[-1])
print("RMSE: %.2f" % np.sqrt(test_loss[-1]))

plot_loss_function(train_loss, test_loss, 'regression')
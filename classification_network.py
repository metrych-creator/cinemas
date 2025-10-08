import copy
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, hamming_loss
from sklearn.metrics import multilabel_confusion_matrix
import random
from models import ClassificationModel
from tools import plot_loss_function

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def train_evaluate(model, epochs, batch_size, optimizer, loss_fn):
    history = []
    batch_start = torch.arange(0, len(X_train), batch_size)

    # --------- TRAIN ---------
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for start in batch_start:
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            # backward pass
            optimizer.zero_grad()
            loss.backward()

            optimizer.step() # update weights
            train_loss += loss.item() * X_batch.size(0)

        avg_train_loss = train_loss / len(X_train)

        # --------- EVAL ---------
        model.eval()
        with torch.no_grad():
            y_pred_probs_test = model(X_test)
            test_loss = loss_fn(y_pred_probs_test, y_test).item()

            y_pred = (y_pred_probs_test >= 0.5).int()
            y_true = y_test.int()

            y_pred_np = y_pred.numpy()
            y_true_np = y_true.numpy()

            f1 = f1_score(y_true_np, y_pred_np, average='macro')
            hamming = hamming_loss(y_true_np, y_pred_np)

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "test_loss": test_loss,
            "f1_macro": f1,
            "hamming_loss": hamming
        })

        print(f"Epoch {epoch}: Loss = {avg_train_loss:.4f}, F1-macro = {f1:.4f}, Hamming = {hamming:.4f}")

    return history


df = pd.read_pickle('cinemas_cleaned.pkl')
df = df[df.select_dtypes(include=['number']).columns.tolist()]

genres = [col for col in df.columns if col.startswith('genres_')]
y_class = df[genres].copy()
X_class = df.loc[:, ['content_rating_PG','content_rating_R','gross','facenumber_in_poster','keywords_other','duration','keywords_vegetarian','num_critic_for_reviews',
                 'keywords_alien','keywords_battle','imdb_score','budget','keywords_party','keywords_dog','keywords_singer','keywords_wedding']]

X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.3, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)


model_class = ClassificationModel(X_train.shape[1], len(genres), dropout_prob=0.3)

optimizer = optim.Adam(model_class.parameters(), lr=0.001, weight_decay=1e-4)
history = train_evaluate(model_class, epochs=10, batch_size=10, optimizer=optimizer, loss_fn=nn.BCELoss())

train_loss = [h["train_loss"] for h in history]
test_loss = [h["test_loss"] for h in history]
f1_macro = [h["f1_macro"] for h in history]
hamming_history = [h["hamming_loss"] for h in history]

print(f"F1-macro: {max(f1_macro):.4f}")
print(f"Hamming loss: {hamming_history[-1]:.4f}")

plot_loss_function(train_loss, test_loss, 'classification')
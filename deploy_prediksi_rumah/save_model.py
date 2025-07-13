import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([
    [100, 3, 1],  # luas, kamar, lokasi
    [70, 2, 2],
    [50, 2, 3],
    [120, 4, 1],
    [90, 3, 2],
    [60, 2, 3],
    [150, 5, 1]
])
y = np.array([1000, 700, 400, 1200, 850, 450, 1400])

model = LinearRegression()
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

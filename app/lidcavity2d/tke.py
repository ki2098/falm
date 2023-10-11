import pandas as pd
import matplotlib.pyplot as plt

tke = pd.read_csv("data/probe.csv")

plt.plot(tke["t"], tke["v"])
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(6,10))

sheet_exp = pd.read_excel('data/profile_experimental.xlsx', sheet_name='x=5')
plt.plot(sheet_exp['u'], sheet_exp['z'], label='experiment')

sheep_sim = pd.read_excel('data/profile_dx=0.04.xlsx', sheet_name='x=5')
plt.plot(sheep_sim['u'], sheep_sim['z'], label='simulation (dz=0.04D)')

plt.legend()
plt.xlabel('U/Uin')
plt.ylabel('z [mm]')
plt.title('U profile at x=5D')
plt.grid(True)
plt.show()
# Importa as bibliotecas

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from bcb import sgs

# Busca a série da SELIC no SGS
selic = sgs.get({"selic": 432}, start="2010-01-01")

# Plota
sns.set_theme()
selic.plot(figsize=(15, 10))

import probscale as probscale
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import norm

entrada732 = pd.Series([16, 17, 15,
                        14, 14, 16,
                        12, 16, 12,
                        13, 11, 14,
                        10, 15, 14,
                        13, 15, 16,
                        17, 13, 15,
                        14, 18, 14,
                        11, 12, 13,
                        13, 15, 12])

entrada733 = pd.Series([18, 16, 11,
                        17, 12, 13,
                        17, 17, 17,
                        16, 18, 17,
                        16, 17, 17,
                        16, 18, 15,
                        18, 18, 16,
                        16, 16, 17,
                        14, 18, 15,
                        11, 18, 10])

plt.hist(entrada732, bins=9)
plt.xlabel('Hours')
plt.ylabel('Frequency')
plt.show()

probscale.probplot(entrada732, plot=plt, dist=norm, problabel='Percentile', datalabel='Hours', probax='y')
plt.grid()
plt.show()

media = np.mean(entrada732)
std = np.std(entrada732, ddof=1)
ks_stat, ks_p_valor = stats.kstest(entrada732, cdf='norm', args=(media, std), N=len(entrada732), mode='auto')


# Checking the critical value of the Kolmogorov-Smirnov test
def kolmogorov_smirnov_critico(n):
    # table of critical values for the kolmogorov-smirnov test - 95% confidence
    # Source: https://www.soest.hawaii.edu/GG/FACULTY/ITO/GG413/K_S_Table_one_Sample.pdf
    # Source: http://www.real-statistics.com/statistics-tables/kolmogorov-smirnov-table/
    # alpha = 0.05 (95% confidential level)

    if n <= 40:
        # valores entre 1 e 40
        kolmogorov_critico = [0.97500, 0.84189, 0.70760, 0.62394, 0.56328, 0.51926, 0.48342, 0.45427, 0.43001, 0.40925,
                              0.39122, 0.37543, 0.36143, 0.34890, 0.33760, 0.32733, 0.31796, 0.30936, 0.30143, 0.29408,
                              0.28724, 0.28087, 0.27490, 0.26931, 0.26404, 0.25907, 0.25438, 0.24993, 0.24571, 0.24170,
                              0.23788, 0.23424, 0.23076, 0.22743, 0.22425, 0.22119, 0.21826, 0.21544, 0.21273, 0.21012]
        ks_critico = kolmogorov_critico[n - 1]
    elif n > 40:
        # valores acima de 40:
        kolmogorov_critico = 1.36 / (np.sqrt(n))
        ks_critico = kolmogorov_critico
    else:
        pass

    return ks_critico


ks_critico = kolmogorov_smirnov_critico(len(entrada732))

print("Média dos dados: ", media)
print("Desvio padrão: ", std)
print("N: ", len(entrada732))
print("P-Valor: ", ks_p_valor)
print("Com 95% de confiança, o valor crítico do teste de Kolmogorov-Smirnov é de: ", ks_critico)
print("Valor calculado do teste de Kolmogorov-Smirnov: ", ks_stat)

print('\nConclusão:')
if ks_critico >= ks_stat:
    print(
        "Com 95% de confiança, não temos evidências para rejeitar a hipótese de normalidade dos dados, segundo o teste de Kolmogorov-Smirnov")
else:
    print(
        "Com 95% de confiança, temos evidências para rejeitar a hipótese de normalidade dos dados, segundo o teste de Kolmogorov-Smirnov")

from random import randint, randrange, random
from math import exp
import matplotlib.pyplot as plt

N = 350
J = -1
H = 0
Kb = 1
trigger = 300000
sampleDelay = 500
res_arr = []
n_samples = 300
S = [[randrange(-1, 2, 2) for i in range(N)] for j in range(N)]

def runCycles(T, H = 0):
    m_arr_loc = []
    e_arr_loc = []
    for i in range(trigger + n_samples * sampleDelay):
        m_loc = 0
        e_loc = 0
        x = randint(0, N - 1)
        y = randint(0, N - 1)
        deltaE = -2 * J * S[x][y] * (S[(x + 1) % N][(y + 1) % N] +
                                  S[(x + 1) % N][(y - 1) % N] +
                                  S[(x - 1) % N][(y + 1) % N] +
                                  S[(x - 1) % N][(y - 1) % N]) - 2 * H * S[x][y]

        if deltaE < 0 or random() < exp(-deltaE/(Kb * T)):
            S[x][y] *= -1

        if i % sampleDelay == 0 and i > trigger:
            for j in S:
                for el in j:
                    m_loc += el
            m_arr_loc.append(m_loc / N**2)

            for a in range(N):
                for b in range(N):
                    e_loc += J * S[a][b] * (S[(a + 1) % N][b] + S[a][(b + 1) % N]) - 2 * H * S[a][b]
            e_arr_loc.append(e_loc)

    susc = 0
    sq_av = 0
    av_sq = 0
    magn = 0

    for m in m_arr_loc:
        sq_av += m / len(m_arr_loc)
        av_sq += m**2 / len(m_arr_loc)

    magn = abs(sq_av)

    sq_av = sq_av ** 2
    susc = (av_sq - sq_av) / T

    cal = 0
    sq_av = 0
    av_sq = 0
    for e in e_arr_loc:
        sq_av += e / len(m_arr_loc)
        av_sq += e**2 / len(m_arr_loc)

    sq_av = sq_av ** 2
    cal = (av_sq - sq_av) / T

    return [T, susc, cal, magn]

for t in range(400, 1, -4):
    res_arr.append(runCycles(t / 100))

susc = []
temp = []
heat = []
arr_magn = []
for i in range(len(res_arr)):
    temp.append(res_arr[i][0])
    susc.append(res_arr[i][1])
    heat.append(res_arr[i][2])
    arr_magn.append(res_arr[i][3])

plt.xlabel("Temperature")
plt.ylabel("Susceptibility")
plt.plot(temp, susc)
#plt.show()

plt.ylabel("Specific heat")
plt.xlabel("Temperature")
plt.plot(temp, heat)
#plt.show()

plt.ylabel("Magnetization")
plt.xlabel("Temperature")
plt.plot(temp, arr_magn)
#plt.show()

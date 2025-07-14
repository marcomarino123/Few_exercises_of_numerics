import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


class Ising2d:

    def __init__(self, Nx, Ny, beta, kb):
        self.Nx = Nx
        self.Ny = Ny
        self.kb = kb
        self.beta = beta
        self.spin_lattice = np.zeros((self.Nx, self.Ny))

        # Initial configuration
        #initial_indices = np.random.rand(self.Nx, self.Ny)
        #self.spin_lattice[initial_indices > 0.5] = 1
        #self.spin_lattice[initial_indices <= 0.5] = -1
        self.spin_lattice = np.ones((self.Nx,self.Ny))

    def _nn_energy_difference(self, i, j):
        # Apply periodic boundary conditions
        top = self.spin_lattice[(i - 1) % self.Nx, j]
        bottom = self.spin_lattice[(i + 1) % self.Nx, j]
        left = self.spin_lattice[i, (j - 1) % self.Ny]
        right = self.spin_lattice[i, (j + 1) % self.Ny]

        summing_nn = top + bottom + left + right
        return - summing_nn * self.spin_lattice[i, j]

    def _spin_flipping_Metropolis(self):
        i = np.random.randint(0, self.Nx)
        j = np.random.randint(0, self.Ny)
        de = - 2 * self._nn_energy_difference(i, j)
        if de < 0:
            self.spin_lattice[i, j] = -self.spin_lattice[i, j]
        else:
            if np.random.rand() < np.exp(-self.beta * de):
                self.spin_lattice[i, j] = -self.spin_lattice[i, j]

    def warmup(self, total_steps):
        for _ in range(total_steps):
            self._spin_flipping_Metropolis()

    def _data_initializing(self, number_measures):
        self.data = {
            'e': np.zeros(number_measures),
            'm': np.zeros(number_measures),
            'e2': np.zeros(number_measures)
        }

    def measuring(self, total_steps, measure_steps):
        number_measures = total_steps // measure_steps
        self._data_initializing(number_measures)
        measure_index = 0

        for i in range(total_steps):
            self._spin_flipping_Metropolis()
            if (i + 1) % measure_steps == 0:
                self.data['e'][measure_index] = self._average_energy_perspin()
                self.data['m'][measure_index] = self._average_magnetization_perspin()
                self.data['e2'][measure_index] = self.data['e'][measure_index] ** 2
                measure_index += 1

        e_avg = np.mean(self.data['e'])
        e2_avg = np.mean(self.data['e2'])

        self.specific_heat = (e2_avg - e_avg ** 2) * self.beta ** 2 * self.kb

    def _average_energy_perspin(self):
        total_energy = 0
        for i in range(self.Nx):
            for j in range(self.Ny):
                total_energy += 0.5 * self._nn_energy_difference(i, j)  # 0.5 to avoid double-counting
        return total_energy / (self.Nx * self.Ny)

    def _average_magnetization_perspin(self):
        return np.sum(self.spin_lattice) / (self.Nx * self.Ny)

    def visualization(self):
        plt.figure(figsize=(6, 6))
        plt.imshow(self.spin_lattice, cmap='coolwarm', interpolation='nearest', origin='lower')
        plt.colorbar(label='Spin')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.clim(-1,1)
        plt.grid(False)
        plt.title('Ising Model Spin Configuration')
        plt.show()

class Ising0d:
    def __init__(self, B, beta):
        self.beta = beta
        self.B = B
        self.spin = 1 if np.random.rand() < 0.5 else -1

    def _spin_flipping_Metropolis(self):
        de = 2 * self.spin * self.B
        if de < 0:
            self.spin = -self.spin
        else:
            if np.random.rand() < np.exp(-self.beta * de):
                self.spin = -self.spin

    def warmup(self, total_steps):
        for _ in range(total_steps):
            self._spin_flipping_Metropolis()

    def _data_initializing(self, number_measures):
        self.data = {
        #  'e': np.zeros(number_measures),
            'm': np.zeros(number_measures),
        # 'e2': np.zeros(number_measures)
        }

    def measuring(self, total_steps, measure_steps):
        number_measures = total_steps // measure_steps
        self._data_initializing(number_measures)
        measure_index = 0
        for i in range(total_steps):
            self._spin_flipping_Metropolis()
            if (i + 1) % measure_steps == 0:
                #self.data['e'][measure_index] = -self.B * self.spin
                self.data['m'][measure_index] = self.spin
                #self.data['e2'][measure_index] = self.data['e'][measure_index] ** 2
                measure_index += 1
        #e_avg = np.mean(self.data['e'])
        #e2_avg = np.mean(self.data['e2'])
        #self.specific_heat = (e2_avg - e_avg ** 2) * self.beta ** 2 * self.kb

    def extracing_average_m(self):
        return np.average(self.data['m'])


def main(part):
    
    if part==1:
        Nx = 100
        Ny = 100
        kb = 1
        beta = 0.2  

        ising2d = Ising2d(Nx, Ny, beta, kb)

        total_steps = 10000
        ising2d.warmup(total_steps)
        #ising.visualization()

        total_steps = 1000
        measure_steps = 10
        ising2d.measuring(total_steps, measure_steps)

        print(f"Specific Heat: {ising2d.specific_heat}")
        ising2d.visualization()
    else:
        beta = 1  
        Bs = np.linspace(-5,5,1000)
        total_steps_warmup  = 100000
        total_steps_measure = 100000
        measure_steps = 10
        m=[]
        for B in Bs:
            #print(B)
            ising0d = Ising0d(B, beta)
            ising0d.warmup(total_steps_warmup)
            ising0d.measuring(total_steps_measure, measure_steps)
            m.append(ising0d.extracing_average_m())
        m=np.array(m)
        plt.figure(figsize=(6, 6))
        plt.plot(Bs, m, 'o', label='Numerical solution')
        plt.plot(Bs, np.tanh(Bs * beta), label='Analytical solution')
        plt.title('Numerical vs Analytical Solution')
        plt.xlabel('B')
        plt.ylabel('m')
        plt.legend()
        plt.show()
        
if __name__ == "__main__":
    main(0)

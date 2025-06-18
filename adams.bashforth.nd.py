import numpy as np
import time
import matplotlib.pyplot as plt

### function giving the approximation of the time derivative in an ODE system 
def rungekutta_method(function,y,t,dt):
    k1=dt*function(t,y)
    k2=dt*function(t+dt/2,y+k1/2)
    k3=dt*function(t+dt/2,y+k2/2)
    k4=dt*function(t+dt,y+k3)
    return y+(1/6)*(k1+2*k2+2*k3+k4)

### function giving the approximation of the time derivative in an ODE system
### approximating the time-derivative at step n+1, trough polynomail combination of the time-derivatives at the preceding steps
def adams_bashforth_4degree_method(function,y,t,dt):
    k1=dt*function(t-3*dt,y[0])
    k2=dt*function(t-2*dt,y[1])
    k3=dt*function(t-dt,y[2])
    k4=dt*function(t,y[3])
    return y[3]+(1/24)*(55*k4-59*k3+37*k2-9*k1)

# time step adjustment
def time_step_adjustment(function,y,t,dt,threshold,evolution_method,threshold_minimum_dt):
    y = np.atleast_1d(y)
    number_dimensions=int(len(y)/2)
    while True: 
        solution1h=y
        solution2h=evolution_method(function,y,t,2*dt)
        for i in range(2):
            solution1h=evolution_method(function,solution1h,t+i*dt,dt)
        #print(np.linalg.norm(solution1h-solution2h), threshold*np.linalg.norm(y))
        if np.linalg.norm(solution1h-solution2h) > threshold*np.linalg.norm(y):
            dt_new=dt*threshold*(np.linalg.norm(y)/np.linalg.norm(solution1h-solution2h))**(1/(1+number_dimensions))
            #print(dt_new)
            if dt_new<threshold_minimum_dt:
                return dt
            else:
                dt=dt_new
        else:
            break
    return dt

### evaluation of the energy
def function_energy(y,parameters):
    y = np.atleast_1d(y)
    number_dimensions=int(len(y)/2)
    kintetic_energy=0.5*parameters[0]*np.dot(y[:number_dimensions],y[:number_dimensions])
    potential_energy=0.5*parameters[1]*np.dot(y[number_dimensions:],y[number_dimensions:])
    return kintetic_energy,potential_energy 

### function expressing the time-increment in the ODE system
def function_force(t,y,parameters):
    y = np.atleast_1d(y)
    number_dimensions=int(len(y)/2)
    return np.concatenate((y[number_dimensions:],-parameters[1]/parameters[0]*y[:number_dimensions]-parameters[2]/parameters[0]*y[number_dimensions:]))

### function explicitly calculating the time evolution of the ODE system given certain initial conditions  
def evolution(initial_conditions,time_interval,dt,evolution_method,function,parameters,threshold,time_step_adjustment_flag,threshold_minimum_dt):
    y=[]
    energy=[]
    t=[]
    y.append(initial_conditions)
    energy.append(function_energy(y[-1],parameters))
    t.append(time_interval[0])
    function_fixed_parameters=lambda z,x: function(z,x,parameters)
    if evolution_method==rungekutta_method:
        count=0
        start_time = time.time()
        while t[-1]<=time_interval[1]:
            # time step adjustment
            if time_step_adjustment_flag==True:
                dt=time_step_adjustment(function_fixed_parameters,y[-1],t[-1],dt,threshold,evolution_method,threshold_minimum_dt)
            #print(dt)
            y.append(evolution_method(function_fixed_parameters,y[-1],t[-1],dt))
            energy.append(function_energy(y[-1],parameters))
            t.append(dt+t[-1])
            count+=1
        end_time = time.time()  # End timer
        elapsed_time = end_time - start_time
        print(f"Execution time for {evolution_method.__name__}: {elapsed_time:.6f} seconds")
    else:
        t.append(time_interval[0])
        start_time = time.time()
        for i in range(4):
            y.append(rungekutta_method(function_fixed_parameters,y[-1],t[-1],dt))
            energy.append(function_energy(y[-1],parameters))
            t.append(dt+t[-1])
        while t[-1]<=time_interval[1]:
            y.append(evolution_method(function_fixed_parameters,y[-4:],t[-1],dt))
            energy.append(function_energy(y[-1],parameters))
            t.append(dt+t[-1])
        end_time = time.time()  # End timer
        elapsed_time = end_time - start_time
        print(f"Execution time for {evolution_method.__name__}: {elapsed_time:.6f} seconds")
    return np.array(y).reshape(-1,len(initial_conditions)),np.array(energy).reshape(-1,2),t


import numpy as np
import matplotlib.pyplot as plt

def plotting_1(y_list, energy_list, t_list):

    # Check if single solution passed as array, not list
    if isinstance(y_list, np.ndarray):
        y_list = [y_list]
        energy_list = [energy_list]
        t_list = [t_list]

    n_solutions = len(y_list)
    number_dimensions = y_list[0].shape[1] // 2
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'black']

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    for i in range(n_solutions):
        y = y_list[i]
        energy = energy_list[i]
        t = t_list[i]

        positions = y[:, :number_dimensions]
        color = colors[i % len(colors)]

        # 3D plot
        ax1.plot3D(positions[:, 0], positions[:, 1], positions[:, 2],
                   color=color, alpha=0.7, label=f'Traj {i+1}')
        ax1.scatter3D(positions[:, 0], positions[:, 1], positions[:, 2],
                      color=color, s=10)

        # Energy plot
        ax2.plot(t, energy[:, 0], color=color, linestyle='--', label=f'U {i+1}')
        ax2.plot(t, energy[:, 1], color=color, linestyle='-.', label=f'K {i+1}')
        ax2.plot(t, energy[:, 0] + energy[:, 1], color=color, linestyle='-', label=f'Total {i+1}')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('3D Trajectories')
    ax1.legend()

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Energy')
    ax2.set_title('Energy Evolution')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


########### MAIN FUNCTION
def main(part):
    
    time_interval=[0,24]
    # time step adjustment is gonna provide for this too large time step
    dt=0.1
    initial_conditions=np.array([1,0,0,0,0,1])
    mass=1
    k=1
    threshold=1.0e-5
    threshold_minimum_dt=1.0e-16
    time_step_adjustment_flag=True

    if part==0:
        method=rungekutta_method
        alphas=np.arange(0,1,0.2)
        print(alphas)
        total_y=[]
        total_energy=[]
        total_t=[]
        for alpha in alphas:
            parameters=[mass,k,alpha]
            y,energy,t=evolution(initial_conditions,time_interval,dt,method,function_force,parameters,threshold,time_step_adjustment_flag,threshold_minimum_dt)
            total_y.append(np.array(y))
            total_energy.append(np.array(energy))
            total_t.append(np.array(t))
        plotting_1(total_y,total_energy,total_t) 
    else:
        methods=[rungekutta_method,adams_bashforth_4degree_method]
        alpha=0.1
        parameters=[mass,k,alpha]
        for method in methods:
            y,energy,_=evolution(initial_conditions,time_interval,dt,method,function_force,parameters,threshold,time_step_adjustment_flag,threshold_minimum_dt)

if __name__ == "__main__":
    main(0)


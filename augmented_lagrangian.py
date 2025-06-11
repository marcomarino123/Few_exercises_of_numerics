import numpy as np
import matplotlib.pyplot as plt

############################# MAJOR FUNCTIONS

#### Gradient of the input function evaluated in x 
#### type gives the type of derivative in the different dimensions
#### dx gives the infinitesimal interval in the different dimensions
def gradient_function(function,x,dx):

    x = np.atleast_1d(x)  
    dx = np.atleast_1d(dx)
    number_dimension=len(dx)
    
    if number_dimension>1:

        ### create a base matrix where each row is a copy of x
        x_matrix = np.tile(x, (number_dimension, 1))  
        ### create shifted matrices using broadcasting and vectorized diagonal addition
        plus_x = x_matrix + np.eye(number_dimension) * dx
        minus_x = x_matrix - np.eye(number_dimension) * dx

        ### plus_function =[f(x+h0),f(x+h1),f(x+h2),...]
        ### minus_function=[f(x-h0),f(x-h1),f(x-h2),...]
        plus_function = np.array([function(row) for row in plus_x])
        minus_function = np.array([function(row) for row in minus_x])

        ### calculating the gradient vector
        dx=dx.reshape(-1, 1)
        gradient=(plus_function-minus_function)/(2*dx.flatten())
    else:
        plus_function=function(x+dx)
        minus_function=function(x-dx)
        gradient=(plus_function-minus_function)/(2*dx)
    
    return gradient

### Bisection method for line search
def bisection_method(function,domain,first_intermediate_point,max_iterations,threshold,flag):
    #print("bs --> iter: ",max_iterations," points: ",domain)
    if max_iterations==0 and flag==0:
        raise('max number of iterations reached!')
    if max_iterations==0 and flag==1:
        return first_intermediate_point
    if np.abs(domain[1]-domain[0])<threshold:
        return domain[0]
    if first_intermediate_point is None:
        while True:
            first_intermediate_point=np.random.uniform(domain[0],domain[1])
            if function(first_intermediate_point)<function(domain[0]) and function(first_intermediate_point)<function(domain[1]):
                break
    if np.abs(domain[0]-first_intermediate_point)<np.abs(domain[1]-first_intermediate_point):
        second_intermediate_point=np.random.uniform(first_intermediate_point,domain[1])
        if function(second_intermediate_point)<function(first_intermediate_point):
            return bisection_method(function,[first_intermediate_point,domain[1]],second_intermediate_point,max_iterations-1,threshold,flag)
        else:
            return bisection_method(function,[domain[0],second_intermediate_point],first_intermediate_point,max_iterations-1,threshold,flag)
    else:
        second_intermediate_point=np.random.uniform(domain[0],first_intermediate_point)
        if function(second_intermediate_point)<function(first_intermediate_point):
            return bisection_method(function,[domain[0],first_intermediate_point],second_intermediate_point,max_iterations-1,threshold,flag)
        else:
            return bisection_method(function,[second_intermediate_point,domain[1]],first_intermediate_point,max_iterations-1,threshold,flag)

### Gradient method
def gradient_method(function,initial_point,dx,max_iterations,threshold,flag,linear_search):
    
    ### saving past positions
    history=[]
    ### saving differences between past positions
    errors=[]

    ### minimization procedure
    point=initial_point
    vector=-gradient_function(function,point,dx)
    iteration=0
    while iteration<max_iterations:
        history.append(np.array(point))
        if iteration > 1:
            errors.append(np.linalg.norm(history[-1] - history[-2]))
        linear_search_function = lambda alpha: function(point+alpha*vector)
        if linear_search==1:
            alpha = bisection_method(linear_search_function,[-5,5],0.0,max_iterations,threshold,1)
        else:
            alpha = 1.0     
        if np.linalg.norm(vector)<threshold:
            history=np.array(history).reshape(-1,len(point))
            return iteration,history,errors,point
        point+=alpha*vector
        vector=-gradient_function(function,point,dx)
        iteration+=1

    if iteration==max_iterations and flag==0:
        raise('max number of iterations reached!')
    else:
        return iteration,history,errors,point

### Conjugate-Gradient method
def conjugate_gradient_method(function,initial_point,dx,max_iterations,threshold,flag,linear_search):
    
    ### saving past positions
    history=[]
    ### saving differences between past positions
    errors=[]

    old_vector=-gradient_function(function,initial_point,dx)
    point=initial_point
    iteration=0
    while iteration<max_iterations:
        #print(iteration)
        history.append(np.array(point))
        if iteration > 1:
            errors.append(np.linalg.norm(history[-1] - history[-2]))
        linear_search_function = lambda alpha: function(point + alpha*old_vector)
        if linear_search==1:
            alpha = bisection_method(linear_search_function,[-5,5],0.0,max_iterations,threshold,1)
        else:
            alpha = 1.0     
        if np.linalg.norm(old_vector)<threshold:
            history=np.array(history).reshape(-1,len(point))
            return iteration,history,errors,point
        point=point+alpha*old_vector

        new_vector=-gradient_function(function,point,dx)
        old_vector=new_vector+old_vector*np.linalg.norm(new_vector)**2/np.linalg.norm(old_vector)**2
        iteration+=1

    if iteration==max_iterations and flag==0:
        raise('max number of iterations reached!')
    else:
        return iteration,history,errors,point

### Augmented Lagrangian method
def al_method(number_dimensions,initial_point,function_minimization,dx,max_iterations_minimization,threshold_minimization,function_constraint,value_constraint,max_iterations_constraint,threshold_constraint_ratio,threshold_constraint_gradient):
    
    ### iterative minimization
    linear_search=1
    iteration=0
    point=initial_point

    ### starting values of the lagrange multipliers
    difference=(function_constraint(point,number_dimensions)-value_constraint)
    mu_value=1/2
    lambda_value=1+mu_value*difference

    while iteration<max_iterations_constraint:
        
        ### defining the constrained functional
        def constrained_function(positions):
            area=function_constraint(positions,number_dimensions)
            return function_minimization(positions,number_dimensions)-lambda_value*(area-value_constraint)+(mu_value/2)*(area-value_constraint)**2
        
        ### checking the area value and the gradient of the constrained function
        ratio=(function_constraint(point,number_dimensions)-value_constraint)/value_constraint
        gradient_modulus=gradient_function(constrained_function,point,dx)
        if np.abs(ratio)<threshold_constraint_ratio:
            if np.linalg.norm(gradient_modulus)<threshold_constraint_gradient:
                return iteration,point
            else:
                max_iterations_minimization*=10
        else:        
            ### increasing the lagrange multipliers
            lambda_value=lambda_value-mu_value*ratio*value_constraint
            mu_value*=2

        ### minimizing the constrained functional
        #print(lambda_value,mu_value,constrained_function(point),function_constraint(point,number_dimensions),value_constraint,ratio,(function_constraint(point,number_dimensions)-value_constraint)**2,np.linalg.norm(gradient_modulus))
        #####_1,_2,_3,point=conjugate_gradient_method(constrained_function,point,dx,max_iterations_minimization,threshold_minimization,1,linear_search)
        _1,_2,_3,point=gradient_method(constrained_function,point,dx,max_iterations_minimization,threshold_minimization,1,linear_search)
        
        iteration+=1

    if iteration==max_iterations_constraint:
        print('max number of iterations reached!')
        return iteration,point
        
def plotting(r_init,r):
    plt.plot(r_init[:, 0], r_init[:, 1], 'ob--')
    plt.plot([r_init[-1, 0], r_init[0, 0]], [r_init[-1, 1], r_init[0, 1]], 'ob--')

    plt.plot(r[:, 0], r[:, 1], 'or--')
    plt.plot([r[-1, 0], r[0, 0]], [r[-1, 1], r[0, 1]], 'or--')

    plt.grid()
    plt.axis('equal')
    plt.show()

########### MAIN FUNCTION
def main():
   
    number_points=20
    number_dimensions=2
    ### positions=[x0,y0,z0,x1,y1,z1,....]
    ### reordered vector inside the functional expressions to facilitate its manupulation
    def energy_functional(positions,number_dimensions):
        positions=np.array(positions).reshape(-1,number_dimensions)
        return np.sum([np.linalg.norm(positions[i+1,:]-positions[i,:]) for i in range(positions.shape[0]-1)])\
            +np.linalg.norm(positions[0,:]-positions[positions.shape[0]-1,:])
    def area_functional(positions,number_dimensions):
        positions=np.array(positions).reshape(-1,number_dimensions)
        center_of_mass=np.sum(positions,axis=0)/positions.shape[0]
        return np.sum([(1/2)*np.abs(((positions[i+1,1]-center_of_mass[1])*(positions[i,0]-center_of_mass[0])+(center_of_mass[0]-positions[i+1,0])*(positions[i,1]-center_of_mass[1]))) for i in range(positions.shape[0]-1)])\
            +(1/2)*np.abs(((positions[0,1]-center_of_mass[1])*(positions[positions.shape[0]-1,0]-center_of_mass[0])+(center_of_mass[0]-positions[0,0])*(positions[positions.shape[0]-1,1]-center_of_mass[1]))) 
    
    max_iterations_minimization=100
    max_iterations_constraint=1000
    threshold_minimization=1.0e-6
    threshold_constraint_ratio=1.0e-6
    threshold_constraint_gradient=1.0e-5
    value_area=np.pi
    dx=np.reshape([1.0e-6]*number_dimensions*number_points,(number_dimensions*number_points))
    dx=dx.flatten()

    ### building the square of points
    side=1
    coordinate=np.linspace(-side/2,side/2,int(number_points/4),endpoint=False)
    points=np.zeros((number_points,number_dimensions))
    shift=np.ones(int(number_points/4))*(side/2)
    points[:int(number_points/4),:]=np.column_stack((coordinate,shift))
    points[int(number_points/4):int(number_points/2),:]=np.column_stack((shift,-coordinate))
    shift=np.ones(int(number_points/4))*(-side/2)
    points[int(number_points/2):int(number_points*3/4),:]=np.column_stack((-coordinate,shift))
    points[int(number_points*3/4):,:]=np.column_stack((shift,coordinate))
    initial_point=points.flatten()
    
    ### minimizing
    iteration,new_points=al_method(number_dimensions,initial_point,energy_functional,dx,max_iterations_minimization,threshold_minimization,area_functional,value_area,max_iterations_constraint,threshold_constraint_ratio,threshold_constraint_gradient)
    new_points=np.reshape(new_points,(number_points,number_dimensions))
    ### plotting
    plotting(points,new_points)

if __name__ == "__main__":
    main()
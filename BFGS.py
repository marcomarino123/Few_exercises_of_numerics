import numpy as np
import cmath
import matplotlib.pyplot as plt
import time

########### MINOR FUNCTIONS
def update_inv_hessian(inv_hessian,step_vector,step_gradient):
    step_vector = step_vector.reshape(-1, 1) 
    step_gradient = step_gradient.reshape(-1, 1)
    sTy = step_vector.T @ step_gradient
    Hy = inv_hessian @ step_gradient
    yHy = step_gradient.T @ Hy
    if sTy != 0:
        term1 = ((sTy + yHy) / (sTy ** 2)) * (step_vector @ step_vector.T)
        term2 = (Hy @ step_vector.T + step_vector @ Hy.T) / sTy
    else:
        term1 = 0.0
        term2 = 0.0
    return inv_hessian + term1 - term2
###########

########### MAJOR FUNCTIONS
#### Gradient of the input function evaluated in x 
#### type gives the type of derivative in the different dimensions
#### dx gives the infinitesimal interval in the different dimensions
def gradient_function(function,x,dx,types):

    x = np.atleast_1d(x)  
    dx = np.atleast_1d(dx)
    number_dimension=len(dx)
    ###print(types)

    if types is None:
        ### if types is not indicated all the components of the gradient are evaluated as central derivatives
        numeric_types=np.zeros(number_dimension+3)
    else:
        ### if types are given as letters a conversion into integers is considered
        if types and isinstance(types[0], str):
            default_types=["c","l","r"]
            _,numeric_types=np.unique((default_types+types),return_inverse=True)

    #print(numeric_types)
    ### calculation of the gradient
    ### square matrices of order N
    ### x+dx_i (each row corresponds to increase only one coordinate and leaves the others untouched)
    plus_x=np.reshape([x]*number_dimension,(number_dimension,number_dimension))+np.diag(dx)
    minus_x=np.reshape([x]*number_dimension,(number_dimension,number_dimension))-np.diag(dx)

    ### arrays of order N
    ### for each shift x+dx_i the function is evaluated: f(x+dx_i) and f(x-dx_i)
    ### each column correspond to a shift only in the correspondent coordinate
    ### plus_x_function =[f(x+h0),f(x+h1),f(x+h2),...]
    ### minus_x_function=[f(x-h0),f(x-h1),f(x-h2),...]
    plus_x_function=np.array([function(plus_x[i,:]) for i in range(number_dimension)])
    minus_x_function=np.array([function(minus_x[i,:]) for i in range(number_dimension)])
    
    ###print(plus_x_function)
    ###print(minus_x_function)
    
    ### the function is evaluated with a null shift in each component
    if number_dimension == 1:
        x_function=function(x)
    else:
        x_function=np.array([function(x) for i in range(number_dimension)])
    #x_function=np.array([function(x) for i in range(number_dimension)])
    ##print(x_function)
    ### grouping togheter the different single coordinates shifts
    matrix_function=np.vstack([plus_x_function,minus_x_function,x_function]).T
    ### matrix_function[:,i]=[f(x+h_i),f(x-h_i),f(x)]
    ### matrix_function=[[f(x+h0),f(x-h0),f(x)],
    ###                  [f(x+h1),f(x-h1),f(x)],
    ###                  [f(x+h2),f(x-h2),f(x)]...]
    #print("matrix_function\n",matrix_function)
    ### each component of the gradient, i.e. each directional derivative is calculated taking into account the pointed type
    ### the central derivative is correspondent to (f(x+h_i)-f(x-h_i))/2|h_i|=(f_i)'(x)
    ### the left derivative is correspondent to (f(x+h_i)-f(x))/|h_i|=(f_i)'(x)
    ### the right derivative is correspondent to (f(x)-f(x-h_i))/|h_i|=(f_i)'(x)
    ### here for each coordinate the components of the matrix function are selected
    numeric_types_conversion = np.zeros((3, 3))
    numeric_types_conversion[:,0] = np.array([1, -1, 0]) 
    numeric_types_conversion[:,1] = np.array([1, 0, -1])  
    numeric_types_conversion[:,2] = np.array([0, -1, 1])  

    factor2 = lambda x: 2.0 if x == 0 else 1.0
    if number_dimension>1:
        gradient=np.zeros(number_dimension)
        for i in range(number_dimension):
            gradient[i]=matrix_function[i,:]@(numeric_types_conversion[:,int(numeric_types[3+i])])/(factor2(int(numeric_types[3+i]))*dx[i])
    else:
        gradient=matrix_function@(numeric_types_conversion[:,int(numeric_types[3])])/(factor2(int(numeric_types[3]))*dx[0])
        #print(numeric_types_conversion[:,int(numeric_types[3])],(factor2(int(numeric_types[3]))*dx[0]))

    #print(gradient)
    return gradient

## Hessian of the input function evaluated in x
def hessian_function(function,x,dx,types):
    if types is not None:
        ### the implementation is quite easy...
        raise('types for hessian matrix not implemented')
    
    ### due to the fact that the Hessian needs to be symmetric, the derivative type is the same for all the coordinates
    x = np.atleast_1d(x)  
    dx = np.atleast_1d(dx)
    number_dimension=len(dx)
    
    if number_dimension>1:
        ### evaluating different single shifts
        plus_x=np.reshape([x]*number_dimension,(number_dimension,number_dimension))+np.diag(dx)
        minus_x=np.reshape([x]*number_dimension,(number_dimension,number_dimension))-np.diag(dx)
        single_x=np.reshape([x]*number_dimension,(number_dimension,number_dimension))
        ### plus_function =[f(x+h0),f(x+h1),f(x+h2),...]
        ### minus_function=[f(x-h0),f(x-h1),f(x-h2),...]
        plus_function=np.array([function(plus_x[i,:]) for i in range(number_dimension)])
        minus_function=np.array([function(minus_x[i,:]) for i in range(number_dimension)])
        single_function=np.array([function(single_x[i,:]) for i in range(number_dimension)])

        ### evaluating different double shifts
        ### permutation 0 --> plusplus_x[0] = [[x0+h0+h0 x1 x2]],
        ###                                    [x0 x1+h1+h0 x2],
        ###                                    [x0 x1 x2+h2+h0]]
        ### permutation 1 --> plusplus_x[1] = [[x0+h0+h1 x1 x2]],
        ###                                    [x0 x1+h1+h1 x2],
        ###                                    [x0 x1 x2+h2+h1]]
        ### permutation 2 --> plusplus_x[2] = [[x0+h0+h2 x1 x2]],
        ###                                    [x0 x1+h1+h2 x2],
        ###                                    [x0 x1 x2+h2+h2]]
        ones=[np.eye(1, number_dimension, k, dtype=int).flatten() for k in range(number_dimension)]
        plusplus_x=np.empty(number_dimension,dtype=object)
        plusminus_x=np.empty(number_dimension,dtype=object)
        minusminus_x=np.empty(number_dimension,dtype=object)
        for i in range(number_dimension):
            plusplus_x[i]=np.reshape([x]*number_dimension,(number_dimension,number_dimension))+np.diag(dx)+np.array([ones[i]*dx]*number_dimension)
            plusminus_x[i]=np.reshape([x]*number_dimension,(number_dimension,number_dimension))+np.diag(dx)-np.array([ones[i]*dx]*number_dimension)
            minusminus_x[i]=np.reshape([x]*number_dimension,(number_dimension,number_dimension))-np.diag(dx)-np.array([ones[i]*dx]*number_dimension)
        ### evaluating the function
        plusplus_function=np.zeros((number_dimension,number_dimension))
        plusminus_function=np.zeros((number_dimension,number_dimension))
        minusminus_function=np.zeros((number_dimension,number_dimension))
        ### permutation 0 --> plusplus_function[0,:] = [f(x+h0+h0),f(x+h0+h1),f(x+h0+h2)]
        ### permutation 1 --> plusplus_function[1,:] = [f(x+h1+h0),f(x+h1+h1),f(x+h1+h2)]
        ### permutation 2 --> plusplus_function[2,:] = [f(x+h2+h0),f(x+h2+h1),f(x+h2+h2)]
        ### permutation 0 --> minusminus_function[0,:] = [f(x-h0-h0),f(x-h0-h1),f(x-h0-h2)]
        ### permutation 1 --> minusminus_function[1,:] = [f(x-h1-h0),f(x-h1-h1),f(x-h1-h2)]
        ### permutation 2 --> minusminus_function[2,:] = [f(x-h2-h0),f(x-h2-h1),f(x-h2-h2)]
        ### permutation 0 --> plusminus_function[0,:] = [f(x+h0-h0),f(x+h0-h1),f(x+h0-h2)]
        ### permutation 1 --> plusminus_function[1,:] = [f(x+h1-h0),f(x+h1-h1),f(x+h1-h2)]
        ### permutation 2 --> plusminus_function[2,:] = [f(x+h2-h0),f(x+h2-h1),f(x+h2-h2)]              
        for i in range(number_dimension):
            plusplus_function[:,i]=[function(plusplus_x[i][j,:]) for j in range(number_dimension)]
            plusminus_function[:,i]=[function(plusminus_x[i][j,:]) for j in range(number_dimension)]
            minusminus_function[:,i]=[function(minusminus_x[i][j,:]) for j in range(number_dimension)]
        
        ### building the Hessian matrix
        dx=dx.reshape(-1, 1)
        dx_matrix=dx@dx.T
        ### mixed terms
        ### d_id_jf=(plusplus_f_ij-plusminus_f_ij-minusplus_f_ij+minusminus_f_ij)/(4*h_i*h_j)
        hessian=(plusplus_function-plusminus_function-plusminus_function.T+minusminus_function)/(4*dx_matrix)
        ### diagonal terms
        ### d_id_if=(plus_f_i-2*single_f+minus_f_i)/h_i^2
        np.fill_diagonal(hessian,(plus_function-2*single_function+minus_function)/(dx.flatten()**2))
        ### calculating the gradient vector
        gradient=(plus_function-minus_function)/(2*dx.flatten())
    else:
        plus_function=function(x+dx)
        minus_function=function(x-dx)
        single_function=function(x)
        hessian=(plus_function-2*single_function+minus_function)/(dx**2)
        gradient=(plus_function-minus_function)/(2*dx)
    
    return gradient,hessian

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
        
#### Newton's method for line search
def newton_method(function,domain,initial_point,dx,max_iterations,threshold,flag):
    ### saving past positions
    history=[]
    
    iteration=0
    next_point=initial_point
    while iteration<max_iterations:
        #print("nm --> iter: ",iteration," point: ",next_point)
        history.append(np.array(next_point))
        if np.abs(next_point-domain[0])<threshold:
            type=["l"]
        elif np.abs(next_point-domain[1])<threshold:
            type=["r"]
        else:
            type=["c"]
        # extracting gradient function and hessian in next_point 
        first_derivative,second_derivative=hessian_function(function,next_point,dx,None)
        #print("derivatives: ",first_derivative,second_derivative)
        if second_derivative!=0:
            step=-first_derivative/second_derivative
            #print("step",step)
            if np.abs(step)>threshold:
                if function(next_point+step)<function(next_point):
                    next_point+=step
                else:
                    next_point=bisection_method(function,domain,next_point,1,threshold,1)
            else:
                return history,next_point
        else:
            next_point=bisection_method(function,domain,next_point,1,threshold,1)
        iteration+=1

    if iteration==max_iterations and flag==0:
        raise('max number of iterations reached!')
    else:
        return history,next_point

### Gradient method
def gradient_method(function,initial_point,dx,max_iterations,threshold):
    
    ### saving past positions
    history=[]
    ### saving differences between past positions
    errors=[]

    ### minimization procedure
    point=initial_point
    vector=-gradient_function(function,point,dx,None)
    iteration=0
    while iteration<max_iterations:
        history.append(np.array(point))
        if iteration > 1:
            errors.append(np.linalg.norm(history[-1] - history[-2]))
        linear_search_function = lambda alpha: function(point+alpha*vector)
        #_,alpha = newton_method(linear_search_function,[-100,100],0.0,dx[0],max_iterations,threshold,1)
        alpha = bisection_method(linear_search_function,[-100,100],0.0,max_iterations,threshold,1)
        #print("step",iteration,"point",point,"alpha",alpha,"old_vector",vector)  
        if np.linalg.norm(alpha*vector)<threshold:
            history=np.array(history).reshape(-1,len(point))
            return iteration,history,errors,point
        point+=alpha*vector
        vector=-gradient_function(function,point,dx,None)
        iteration+=1

    if iteration==max_iterations:
        raise('max number of iterations reached!')

### Conjugate-Gradient method
def conjugate_gradient_method(function,initial_point,dx,max_iterations,threshold):
    
    ### saving past positions
    history=[]
    ### saving differences between past positions
    errors=[]

    old_vector=-gradient_function(function,initial_point,dx,None)
    point=initial_point
    iteration=0
    while iteration<max_iterations:
        history.append(np.array(point))
        if iteration > 1:
            errors.append(np.linalg.norm(history[-1] - history[-2]))
        linear_search_function = lambda alpha: function(point + alpha*old_vector)
        #_,alpha = newton_method(linear_search_function,[-100,100],0.0,dx[0],max_iterations,threshold,1)
        alpha = bisection_method(linear_search_function,[-100,100],0.0,max_iterations,threshold,1)
        #print("step",iteration,"point",point,"alpha",alpha,"old_vector",old_vector)  
        if np.linalg.norm(alpha*old_vector)<threshold:
            history=np.array(history).reshape(-1,len(point))
            return iteration,history,errors,point
        point=point+alpha*old_vector
        new_vector=-gradient_function(function,point,dx,None)
        old_vector=new_vector+old_vector*np.linalg.norm(new_vector)**2/np.linalg.norm(old_vector)**2
        iteration+=1

    if iteration==max_iterations:
        raise('max number of iterations reached!')

### BFGS method
def bfgs_method(function,initial_point,dx,max_iterations,threshold,start,epsilon):
    
    number_dimensions=len(initial_point)
    
    ### initial hessian matrix
    next_gradient,hessian=hessian_function(function,initial_point,dx,None)
    ### initial procedure of inversion
    if  start == 'inv':
        inv_hessian=np.linalg.inv(hessian+np.eye(number_dimensions)*epsilon)
    elif start == 'diag':
        inv_hessian=np.linalg.inv(np.diag(np.diag(hessian))+np.eye(number_dimensions)*epsilon)
    elif start == 'eye':
        inv_hessian=function(initial_point)*np.eye(number_dimensions)
    else:
        raise('starting not implemented')
    
    ### saving history positions
    history=[]
    ### saving differences between past positions
    errors=[]

    ### iterative minimization
    iteration=0
    point=initial_point
    
    while iteration<max_iterations:  
        history.append(np.array(point))
        if iteration > 1:
            errors.append(np.linalg.norm(history[-1] - history[-2]))
        initial_vector=-inv_hessian@next_gradient
        linear_search_function = lambda alpha: function(point + alpha*initial_vector)
        alpha = bisection_method(linear_search_function,[-1000,1000],0.0,max_iterations,threshold,1)
        ##alpha = newton_method(linear_search_function,[-100,100],0.0,dx[0],max_iterations,threshold,1)
        step_vector = alpha*initial_vector
        #print(step_vector)
        if np.linalg.norm(step_vector)<threshold:
            history=np.array(history).reshape(-1,number_dimensions)
            return iteration,history,errors,point
        point+=step_vector
        step_gradient = gradient_function(function,point,dx,None)-next_gradient
        inv_hessian = update_inv_hessian(inv_hessian,step_vector,step_gradient)
        next_gradient+=step_gradient
        iteration+=1

    if iteration==max_iterations:
        raise('max number of iterations reached!')

###### Plot
##def plotting(function,initial_point,dx,max_iterations,threshold,name,epsilon):
##
##    iteration_grad,x_grad,errors_grad,fin_grad = gradient_method(function,initial_point,dx,max_iterations,threshold)
##    iteration_cg,x_cg,errors_cg,fin_cg = conjugate_gradient_method(function,initial_point,dx,max_iterations,threshold)
##    
##    starts=['inv','diag','eye']
##    iteration_bfgs=np.empty(3,dtype=object)
##    x_bfgs=np.empty(3,dtype=object)
##    errors_bfgs=np.empty(3,dtype=object)
##    fin_bfgs=np.empty(3,dtype=object)
##    for j,start in enumerate(starts):
##        iteration_bfgs[j],x_bfgs[j],errors_bfgs[j],fin_bfgs[j] = bfgs_method(function,initial_point,dx,max_iterations,threshold,start,epsilon)
##
##    # Create a plot
##    x_values_grad = [pair[0] for pair in x_grad]
##    y_values_grad = [pair[1] for pair in x_grad]
##    x_values_cg = [pair[0] for pair in x_cg]
##    y_values_cg = [pair[1] for pair in x_cg]
##    x_values_bfgs=np.empty(3,dtype=object)
##    y_values_bfgs=np.empty(3,dtype=object)
##    for j,start in enumerate(starts):
##        x_values_bfgs[j] = [pair[0] for pair in x_bfgs[j]]
##        y_values_bfgs[j] = [pair[1] for pair in x_bfgs[j]]
##
##    x = np.linspace(-2.5, 2.5, 1000)
##    y = np.linspace(-2.5, 2.5, 1000)
##    X, Y = np.meshgrid(x, y)
##
##    # Create a plot for minimum
##    plt.imshow(function((X,Y)), extent=[x.min(), x.max(), y.min(), y.max()], cmap="inferno", origin="lower",)
##    plt.colorbar()
##    plt.plot(x_values_grad, y_values_grad, marker='o', label='Gradient method')
##    plt.plot(x_values_cg,   y_values_cg, marker='o', label='Conjugate-Gradient method')
##    for j,start in enumerate(starts):
##        plt.plot(x_values_bfgs[j], y_values_bfgs[j], marker='o', label='BFGS method {}'.format(start))
##    
##    plt.plot(x_grad[0][0],x_grad[0][1], marker='x', label='Initial Point')
##    
##    if name == 'Ros':
##        plt.plot(1,1, marker='x', label='Analytical minimum')
##    else:
##        plt.plot(-1.76,-1.76, marker='x', label='Analytical minimum')
##        plt.plot(1.76,1.76, marker='x', label='Analytical minimum')
##
##    print("Iterations G method:",iteration_grad," Minimum:",fin_grad)
##    print("Iterations CG method:",iteration_cg," Minimum:",fin_cg)
##    for j,start in enumerate(starts):
##        print("Iterations BFGS method {}:".format(start),iteration_bfgs[j]," Minimum:",fin_bfgs[j])
##    
##    plt.title('')
##    plt.xlabel('x')
##    plt.ylabel('y')
##    plt.legend()
##    plt.grid(True)
##    plt.show()
##    
##    # Create plot for L2 norm
##    step_grad = np.arange(len(errors_grad))
##    step_cg   = np.arange(len(errors_cg))
##    step_bfgs =np.empty(3,dtype=object)
##    for j,start in enumerate(starts):
##        step_bfgs[j]=np.arange(len(errors_bfgs[j]))
##
##    plt.plot(step_grad, errors_grad, marker='o', label='Gradient method')
##    plt.plot(step_cg,   errors_cg, marker='o', label='Conjugate-Gradient method')
##    for j,start in enumerate(starts):
##        plt.plot(step_bfgs[j], errors_bfgs[j], marker='o', label='BFGS method {}'.format(start))
##    
##    plt.title('')
##    plt.xlabel('steps')
##    plt.ylabel('error')
##    plt.xscale('log')
##    plt.yscale('log')
##    plt.legend()
##    plt.grid(True)
##    plt.show()
##    


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plotting(function, initial_point, dx, max_iterations, threshold, name, epsilon):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Optimization methods
    iteration_grad, x_grad, errors_grad, fin_grad = gradient_method(function, initial_point, dx, max_iterations, threshold)
    iteration_cg, x_cg, errors_cg, fin_cg = conjugate_gradient_method(function, initial_point, dx, max_iterations, threshold)

    starts = ['inv', 'diag', 'eye']
    iteration_bfgs = np.empty(3, dtype=object)
    x_bfgs = np.empty(3, dtype=object)
    errors_bfgs = np.empty(3, dtype=object)
    fin_bfgs = np.empty(3, dtype=object)
    for j, start in enumerate(starts):
        iteration_bfgs[j], x_bfgs[j], errors_bfgs[j], fin_bfgs[j] = bfgs_method(function, initial_point, dx, max_iterations, threshold, start, epsilon)

    # Create a grid
    x = np.linspace(-2.5, 2.5, 100)
    y = np.linspace(-2.5, 2.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = function((X, Y))

    # 3D Surface
    surf = ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='k', alpha=0.6)
    ax.contour(X, Y, Z, zdir='z', offset=np.min(Z)-1, cmap='gray', linewidths=0.5)

    # Plot paths
    def plot_path(points, label, color):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [function(p) for p in points]
        ax.plot(xs, ys, zs, marker='o', label=label, color=color)

    plot_path(x_grad, "Gradient Descent", 'black')
    plot_path(x_cg, "Conjugate Gradient", 'green')
    colors = ['red', 'orange', 'purple']
    for j, start in enumerate(starts):
        plot_path(x_bfgs[j], f'BFGS ({start})', colors[j])

    # Initial and analytical points
    ax.scatter(*initial_point, function(initial_point), color='blue', s=100, marker='x', label='Initial Point')
    if name == 'Ros':
        ax.scatter(1, 1, function((1,1)), color='magenta', s=100, marker='x', label='Analytical Minimum')
    else:
        for val in [-1.76, 1.76]:
            ax.scatter(val, val, function((val, val)), color='magenta', s=100, marker='x', label='Analytical Minimum')

    ax.set_title('Optimization Paths on 3D Surface')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(x, y)')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Print results
    print("Iterations G method:", iteration_grad, " Minimum:", fin_grad)
    print("Iterations CG method:", iteration_cg, " Minimum:", fin_cg)
    for j, start in enumerate(starts):
        print("Iterations BFGS method {}:".format(start), iteration_bfgs[j], " Minimum:", fin_bfgs[j])


########### MAIN FUNCTION
def main(part):
    
    max_iterations=1.0e+17
    threshold=1.0e-8
    dx=[0.0001,0.0001]
    epsilon=0.000001

    if part == 0:
        def func_rosenbrock(x):
            return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        initial_point=[-1,-1]
        name='Ros'
        plotting(func_rosenbrock,initial_point,dx,max_iterations,threshold,name,epsilon)
    else:
        def func_complicated(x):
            return 1/(1+np.exp(-10*(x[0]*x[1]-3)**2)/(x[0]**2+x[1]**2)) 
        initial_points=[[1.5,2.3],[-1.7,-1.9],[0.5,0.6]]
        name='Compl'
        for _,initial_point in enumerate(initial_points):
            plotting(func_complicated,initial_point,dx,max_iterations,threshold,name,epsilon)
    
if __name__ == "__main__":
    main(1)


#### TESTING 1-DIMENSIONAL MINIMIZATION
##def function_test(x):
##    return x**2-2
##domain=[-10,10]
##initial_point=-5
##max_iterations=10000
##threshold=1.0e-4
##dx=1.0e-6
##print(hessian_function(function_test,initial_point,dx,None))
##print(bisection_method(function_test,domain,initial_point,max_iterations,threshold,0))
##print(newton_method(function_test,domain,initial_point,dx,max_iterations,threshold,0))

#### TESTING ANOTHER IMPLEMENTATION OF HESSIAN
## Hessian of the input function evaluated in x, passing through the gradient function
## critical definition, not preserving symmetry of double partial derivates
###def hessian_function_from_gradient(function,x,dx,types):
###    x = np.atleast_1d(x)  
###    dx = np.atleast_1d(dx)
###    def gradient(y):
###        return gradient_function(function,y,dx,types)
###    number_dimension=len(dx)
###    if types is None:
###        ### if types is not indicated all the components of the gradient are evaluated as central derivatives
###        numeric_types=np.zeros(number_dimension+3)
###    else:
###        ### if types are given as letters a conversion into integers is considered
###        if types and isinstance(types[0], str): 
###            default_types=["c","l","r"]
###            _,numeric_types=np.unique((default_types+types),return_inverse=True)
###    ### calculation of the gradient
###    ### square matrices of order N
###    ### x+dx_i (each row corresponds to increase only one coordinate and leaves the others untouched)
###    plus_x=np.reshape([x]*number_dimension,(number_dimension,number_dimension))+np.diag(dx)
###    minus_x=np.reshape([x]*number_dimension,(number_dimension,number_dimension))-np.diag(dx)
###    ### arrays of order N
###    ### for each shift x+dx_i the function is evaluated: f(x+dx_i) and f(x-dx_i)
###    ### each column correspond to a shift only in the correspondent coordinate
###    ### plus_x_function =[f'(x+h0),f'(x+h1),f'(x+h2),...]=[[d_0f(x+h0),d_1f(x+h0),d_2f(x+h0),...],
###    ###                                                    [d_0f(x+h1),d_1f(x+h1),d_2f(x+h1),...],
###    ###                                                    [d_0f(x+h2),d_1f(x+h2),d_2f(x+h2),...]...]
###    ### minus_x_function=[f'(x-h0),f'(x-h1),f'(x-h2),...]=[[d_0f(x-h0),d_1f(x-h0),d_2f(x-h0),...]...]
###    plus_x_function=np.array([gradient(plus_x[i,:]) for i in range(number_dimension)])
###    minus_x_function=np.array([gradient(minus_x[i,:]) for i in range(number_dimension)])
###    ##print("plus_x_function\n",plus_x_function)
###    ##print("minus_x_function\n",minus_x_function)
###    ### the function is evaluated with a null shift in each component
###    ### x_function=[d_0f(x),d_1f(x),d_2f(x),...]
###    x_function=np.array([gradient(x) for i in range(number_dimension)])
###    ##print("x_function\n",x_function)
###    ### grouping togheter the different single coordinates shifts
###    matrix_function = np.hstack([plus_x_function, minus_x_function,x_function]) 
###    ### print(matrix_function)
###    ### matrix_function[:,i]=[f(x+h_i),f(x-h_i),f(x)]
###    ### matrix_function=[[f'(x+h0),f'(x-h0),f'(x)],
###    ###                  [f'(x+h1),f'(x-h1),f'(x)],
###    ###                  [f'(x+h2),f'(x-h2),f'(x)]...]
###    ### matrix_function=[[d_0f(x+h0),d_1f(x+h0),d_2f(x+h0)...,d_0f(x-h0),d_1f(x-h0)...,d_0f(x),d_1f(x)...],
###    ###                  [d_0f(x+h1),d_1f(x+h1),d_2f(x+h1)...,d_0f(x-h1),d_1f(x-h1)...,d_0f(x),d_1f(x)...],
###    ###                  [d_0f(x+h2),d_1f(x+h2),d_2f(x+h2)...,d_0f(x-h2),d_1f(x-h2)...,d_0f(x),d_1f(x)...],...]
###    ### matrix_function.T=[[d_0f(x+h0),d_0f(x+h1),d_0f(x+h2)...],
###    ###                    [d_1f(x+h0),d_1f(x+h1),d_1f(x+h2)...],
###    ###                    [d_2f(x+h0),d_2f(x+h1),d_2f(x+h2)...],
###    ###                     ....
###    ###                    [d_0f(x-h0),d_0f(x-h1),d_0f(x-h2)...],
###    ###                    [d_1f(x-h0),d_1f(x-h1),d_1f(x-h2)...],
###    ###                    [d_2f(x-h0),d_2f(x-h1),d_2f(x-h2)...],
###    ###                     ....
###    ###                    [d_0f(x),d_0f(x),d_0f(x)............],
###    ###                    [d_1f(x),d_1f(x),d_1f(x)............],
###    ###                    ....]
###    ###print("matrix_function\n",matrix_function)
###    ### each component of the Hessian, i.e. each directional derivative is calculated taking into account the pointed type
###    ### the central derivative is correspondent to (f(x+h_i)-f(x-h_i))/2|h_i|=(f_i)'(x)
###    ### the left derivative is correspondent to (f(x+h_i)-f(x))/|h_i|=(f_i)'(x)
###    ### the right derivative is correspondent to (f(x)-f(x-h_i))/|h_i|=(f_i)'(x)
###    ### here for each coordinate the components of the matrix function are selected
###    ### to each dimension is associated a weighting matrix
###    ### for i in range(number_dimension):
###    ###         M[i,:]*w[i]=H[i,:] 1x3N * 3NxN --> 1xN giving the row of the Hessian
###    numeric_types_conversion=np.zeros((3,3*number_dimension,number_dimension))
###    ### central derivative
###    numeric_types_conversion[0]=np.vstack((np.eye(number_dimension),-np.eye(number_dimension),np.zeros((number_dimension,number_dimension))))
###    ### left derivative
###    numeric_types_conversion[1]=np.vstack((np.eye(number_dimension),np.zeros((number_dimension,number_dimension)),-np.eye(number_dimension)))
###    ### right derivative
###    numeric_types_conversion[2]=np.vstack((np.zeros((number_dimension,number_dimension)),-np.eye(number_dimension),np.eye(number_dimension)))
###    factor2 = lambda x: 2.0 if x == 0 else 1.0
###    ### hessian
###    ### d_0d_0f d_0d_1f d_0d_2f d_0d_3f....d_0d_Nf ---> first row i have to combine h0
###    ### d_1d_0f d_1d_1f.... ---> second columnd i have to combine h1
###    if number_dimension>1:
###        hessian=np.zeros((number_dimension,number_dimension))
###        for i in range(number_dimension):
###            hessian[i,:]=matrix_function[i,:]@(numeric_types_conversion[int(numeric_types[3+i])]/(factor2(int(numeric_types[3+i]))*dx[i]))
###        ### enforcing symmetry
###        #hessian = 0.5 * (hessian + hessian.T)
###        return x_function[0,:], hessian
###    else:
###        hessian=matrix_function@numeric_types_conversion[int(numeric_types[3])]/(factor2(int(numeric_types[3]))*dx[0])
###        return x_function[0],hessian[0]
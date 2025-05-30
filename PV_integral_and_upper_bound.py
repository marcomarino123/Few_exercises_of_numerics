import numpy as np
import sys
import cmath

# define functions to be intergrated
def function_1(x):
    return 2*np.sinh(x)/x
def function_2(x):
   return 2*np.exp(-x**2)
def function_3(x,epsilon):
   return 2*np.sin(x)*np.exp(-epsilon*x)/x
def function_4(x,epsilon):
   return 2*np.sin(x/(1-x+epsilon))/(x*(1-x)+epsilon)

# define error function
def err_function_2(x_max):
    return np.exp(-x_max**2)/(x_max)
def err_function_3(x_max,epsilon):
    return 2*np.exp(-x_max*epsilon)/(x_max*epsilon)

# define integration procedure (ONC, to avoid borders) 
def midpoint_rule(f, a, b, h):
    n=int((b-a)/h)
    x=np.linspace(a+h/2,b-h/2,n)
    y=f(x)
    return h*np.sum(y)

# Simpson's rule (ANC)
def simpsons_rule(f, a, b, h):
    n = int((b - a)/h)
    if n % 2 == 1:  
        n += 1
    x = np.linspace(a,b,2*n+1)
    y = f(x)
    return (h / 6) * (
        y[0]+y[2*n]+4*np.sum(y[1:2*n+1:2])
        +2*np.sum(y[2:2*n:2])
    )

# reducing the step-size of integration to reach a convergence of the integral value
def convergence(function,a,b,threshold,max_iterations,h):
    if h is None:
        h=b-a
    count=0
    old=0
    while count<max_iterations:
        new=midpoint_rule(function,a,b,h)
        #new=simpsons_rule(function,a,b,h)
        if count>=1:
            if np.abs(new-old)<threshold:
                break
        count+=1
        old=new
        h=h/2
    if count==max_iterations:
        sys.exit("No convergence after"+str(max_iterations)+"interations")
    return h,new
    

# minimizing the ratio between the error and the integral value in order to fix appropriate upper bound of the integral
def minimizing_upper_bound_error(function,a,initial_b,threshold_integration,max_iterations_integration,threshold_minimization,max_iterations_minimization,err_function,step_minimization): 
    b_new=initial_b
    count=0
    while count<max_iterations_minimization:
        h,integral=convergence(function,a,b_new,threshold_integration,max_iterations_integration,None)
        #print("Integral value",integral,"Upper bound value",b_new,"Step size value",h)
        ratio=np.abs(err_function(b_new)/integral)
        #print("Ratio between error function and integral value is",ratio)
        if count>=1:
            if ratio<threshold_minimization:
                break
        count+=1
        if step_minimization is None:
            step_minimization=np.abs(a-b_new)/10
        b_new+=step_minimization

    return h,integral,b_new,ratio

def main(prob):
    if prob==0:
        ### EXERCISE 2 INTEGRAL A
        a=0
        b=1
        threshold=1.0e-10
        max_iterations=32
        h,integral=convergence(function_1,a,b,threshold,max_iterations,None)
        print("The value of the integral is ",integral," with a step size of ",h) 
    elif prob==1:
        ### EXERCISE 2 INTEGRAL B
        a=0
        initial_b=1
        threshold_integration=1.0e-6
        max_iterations_integration=32
        # this is a threshold over the ratio between the error function and the value of the integral
        threshold_minimization=1.0e-10
        max_iterations_minimization=40
        # this is considered step to increase the upper bound
        #step_minimization=np.abs(initial_b-a)
        step_minimization=None
        h,integral,final_b,percentage_error=minimizing_upper_bound_error(function_2,a,initial_b,threshold_integration,max_iterations_integration,threshold_minimization,max_iterations_minimization,err_function_2,step_minimization)
        print("The value of the integral is",integral," with a step size of",h," and a ratio between the error function and the integral value of ",percentage_error," and a upper bound of",final_b) 
    else:
        ### EXERCISE 2 INTEGRAL C
        epsilon=0.0
        a=0
        b=1.0e+4
        threshold=1.0e-12
        max_iterations=32
        function_3_epsilon=lambda x: function_3(x,epsilon)
        h,integral=convergence(function_3_epsilon,a,b,threshold,max_iterations,0.001)
        print("The value of the integral is ",integral," with a step size of ",h) 
        print("Difference with pi",np.abs(integral-np.pi))
        #print("Error committed less than",2*np.exp(-epsilon*b)/(epsilon*b))

if __name__ == "__main__":
    main(3)
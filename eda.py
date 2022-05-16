


def stats_result(p,null_h,**kwargs):
    """
    Compares p value to α and outputs whether or not the null hypothesis
    is rejected or if it failed to be rejected.
    DOES NOT HANDLE 1-TAILED T TESTS
    
    Required inputs:  p, null_h (str)
    Optional inputs: alpha (default = .05), chi2, r, t, u
    
    """
    #get r value if passed, else none
    u=kwargs.get('u',None)
    t=kwargs.get('t',None)
    r=kwargs.get('r',None)
    chi2=kwargs.get('chi2',None)
    alpha=kwargs.get('alpha',.05) #default value of alpha is .05
    
    #Determine whether or not we reject the null hypothesis
    if p < alpha: print(f"\n\033[1mWe reject the null hypothesis\033[0m, p = {p} | α = {alpha}")
    else: print(f"\n\033[1mWe failed to reject the null hypothesis\033[0m, p = {p} | α = {alpha}")
    #Print any other relevant statistical variables
    if 'u' in kwargs: print(f'  u: {u}')
    if 't' in kwargs: print(f'  t: {t}')
    if 'r' in kwargs: print(f'  r: {r}')
    if 'chi2' in kwargs: print(f'  chi2: {chi2}')
    #remind the user of the null hypothesis
    print(f'The null hypothesis was: {null_h}\n')
    return None
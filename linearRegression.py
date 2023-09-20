def linearRegression(x,y):
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    x = np.array(x).reshape((-1,1))
    
    model = LinearRegression().fit(x,y)
    r_sq = model.score(x,y)
    slope = model.coef_
    intercept = model.intercept_
    y_predict = model.predict(x)
    
    return r_sq, slope, intercept, y_predict
    
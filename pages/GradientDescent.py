import streamlit as st
st.markdown("# Gradient Descent ")
st.sidebar.markdown("# Gradient Descent ")

def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

app_mode = st.sidebar.selectbox('Select Page',['GradientDescent01','GradientDescent02']) 

if app_mode=='GradientDescent01':
    st.title("GradientDescent01") 
    from pages.GradientDescent1.Gradientdescent01 import *
    (x1, it1) = myGD1(-5, .1)
    st.write(x1[-1], cost(x1[-1]) ,it1)
    (x1, it1) = myGD1(5, .1)
    st.write(x1[-1], cost(x1[-1]) ,it1)

    x = np.linspace(-6, 6, 100)
    y = x**2 + 5*np.sin(x)
    fig, ax = plt.subplots()
    k = 0
    plt.subplot(2,4,1)
    plt.plot(x, y, 'b')
    plt.plot(x1[k], cost(x1[k]), 'ro')
    s = 'iter %d/%d,grad = %.4f' % (k, it1, grad(x1[k]))
    plt.xlabel(s, fontsize = 8)

    k = 1
    plt.subplot(2,4,2)
    plt.plot(x, y, 'b')
    plt.plot(x1[k], cost(x1[k]), 'ro')
    s = 'iter %d/%d,grad = %.4f' % (k, it1, grad(x1[k]))
    plt.xlabel(s, fontsize = 8)

    plt.tight_layout()
    st.pyplot(fig)

elif app_mode == 'GradientDescent02':
    st.title('GradientDescent02')
    from pages.GradientDescent1.Gradientdescent02 import *
    X = np.random.rand(1000)
    y = 4 + 3 * X + .5*np.random.randn(1000)
    fig, ax = plt.subplots()
    plt.plot(X,y,'bo', markersize = 2)
    # chuyển mảng một chiều thành ma trận
    X = np.array([X])
    y = np.array([y])
    # chuyển vị ma trận
    X=X.T
    y=y.T
    model = LinearRegression()
    model.fit(X, y)
    w0 = model.intercept_
    w1 = model.coef_[0]
    x0 = 0
    y0 = w1*x0 +w0
    x1 = 1
    y1 = w1*x1 +w0
    plt.plot([x0,x1],[y0,y1], 'r')
    st.pyplot(fig)
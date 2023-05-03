import streamlit as st
st.markdown("# SVM ")
st.sidebar.markdown("# SVM ")

def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

app_mode = st.sidebar.selectbox('Select Page',['SVM01','SVM02']) 

if app_mode=='SVM01':
    st.title("SVM01") 
    from pages.SVM1.SVM01 import *
    data, labels = make_blobs(n_samples=N, 
                          centers=np.array(centers),
                          random_state=1)

    res = train_test_split(data, labels, 
                        train_size=0.8,
                        test_size=0.2,
                        random_state=1)

    train_data, test_data, train_labels, test_labels = res 
    svc = LinearSVC(max_iter = 10000)

    svc.fit(train_data, train_labels)
    predicted = svc.predict(test_data)
    accuracy = accuracy_score(predicted, test_labels)
    st.write('Độ chính xác: %.0f%%' % (accuracy*100))
    fig, ax = plt.subplots()
    nhom0 = []
    nhom1 = []
    for i in range(N):
        if labels[i] == 0:
            nhom0.append([data[i,0], data[i,1]])
        elif labels[i] == 1:
            nhom1.append([data[i,0], data[i,1]])
    
    nhom0 = np.array(nhom0)
    nhom1 = np.array(nhom1)

    plt.plot(nhom0[:,0],nhom0[:,1],'go', markersize = 5)
    plt.plot(nhom1[:,0],nhom1[:,1], 'rs', markersize = 5)
    plt.legend([0,1])

    w = svc.coef_[0]
    intercept = svc.intercept_[0]
    a = -w[0] / w[1]

    xx = np.linspace(2, 6)
    yy = a * xx - (intercept) / w[1]
    plt.plot(xx, yy, 'b')

    xx = np.linspace(2, 6)
    yy = a * xx - (intercept) / w[1] + 0.5/w[1]
    plt.plot(xx, yy, 'b--')

    xx = np.linspace(2, 6)
    yy = a * xx - (intercept) / w[1] - 0.5/w[1]
    plt.plot(xx, yy, 'b--')

    st.pyplot(fig)

elif app_mode == 'SVM02':
    st.title('SVM02')
    from pages.SVM1.SVM01a import *
    data, labels = make_blobs(n_samples=N, 
                          centers=np.array(centers),
                          random_state=1)

    res = train_test_split(data, labels, 
                        train_size=0.8,
                        test_size=0.2,
                        random_state=1)

    train_data, test_data, train_labels, test_labels = res 
    svc = SVC(kernel = 'linear')

    svc.fit(train_data, train_labels)
    predicted = svc.predict(test_data)
    accuracy = accuracy_score(predicted, test_labels)
    st.write('Độ chính xác: %.0f%%' % (accuracy*100))
    fig, ax = plt.subplots()
    nhom0 = []
    nhom1 = []
    for i in range(N):
        if labels[i] == 0:
            nhom0.append([data[i,0], data[i,1]])
        elif labels[i] == 1:
            nhom1.append([data[i,0], data[i,1]])
    
    nhom0 = np.array(nhom0)
    nhom1 = np.array(nhom1)

    plt.plot(nhom0[:,0],nhom0[:,1],'go', markersize = 5)
    plt.plot(nhom1[:,0],nhom1[:,1], 'rs', markersize = 5)
    plt.legend([0,1])

    w = svc.coef_[0]
    intercept = svc.intercept_[0]
    a = -w[0] / w[1]

    xx = np.linspace(2, 7)
    yy = a * xx - (intercept) / w[1]

    margin = 1 / np.sqrt(np.sum(svc.coef_**2))
    yy_down = yy - np.sqrt(1 + a**2) * margin
    yy_up = yy + np.sqrt(1 + a**2) * margin

    plt.plot(xx, yy, 'b')
    plt.plot(xx, yy_down, 'b--')
    plt.plot(xx, yy_up, 'b--')

    plt.plot(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1], 'bs')
    st.pyplot(fig)


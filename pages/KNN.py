import streamlit as st
st.markdown("# KNN ")
st.sidebar.markdown("# KNN ")

def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

app_mode = st.sidebar.selectbox('Select Page',['KNN','KNNa','KNNb']) 

if app_mode=='KNN':
    st.title("KNN") 
    from pages.KNN1.KNN import *
    fig, ax = plt.subplots()
    plt.plot(nhom0[:,0],nhom0[:,1],'go')
    plt.plot(nhom1[:,0],nhom1[:,1], 'ro')
    plt.plot(nhom2[:,0],nhom2[:,1], 'bo')
    plt.legend([0,1,2])
    st.pyplot(fig)

    res = train_test_split(data, labels, 
                        train_size=0.8,
                        test_size=0.2,
                        random_state=1)

    train_data, test_data, train_labels, test_labels = res 
    knn = KNeighborsClassifier()
    knn.fit(train_data, train_labels)
    predicted = knn.predict(test_data)
    accuracy = accuracy_score(predicted, test_labels)
    st.write('Độ chính xác: %.0f%%' % (accuracy*100))

elif app_mode == 'KNNa':
    st.title('KNNa')
    from pages.KNN1.KNNa import *
    fig, ax = plt.subplots()
    plt.plot(nhom0[:,0],nhom0[:,1],'go')
    plt.plot(nhom1[:,0],nhom1[:,1], 'ro')
    plt.plot(nhom2[:,0],nhom2[:,1], 'bo')
    plt.legend([0,1,2])
    st.pyplot(fig)
    res = train_test_split(data, labels, 
                        train_size=0.8,
                        test_size=0.2,
                        random_state=1)

    train_data, test_data, train_labels, test_labels = res 
    knn = KNeighborsClassifier()
    knn.fit(train_data, train_labels)
    predicted = knn.predict(test_data)
    accuracy = accuracy_score(predicted, test_labels)
    st.write('Độ chính xác: %.0f%%' % (accuracy*100))
    joblib.dump(knn, "pages/KNN1/knn.pkl")

elif app_mode == 'KNNb':
        st.title('KNNb')
        from pages.KNN1.KNNb import *
        
        app = App()
        app.mainloop()
        app.destroy()   
        from pages.KNN1.KNN import *
        knn = joblib.load("pages/KNN1/knn_digit.pkl")

        app_mode = 'KNN'
        
        
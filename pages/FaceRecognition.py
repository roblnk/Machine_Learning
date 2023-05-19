import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import joblib

st.markdown("# Face Recognition")
st.sidebar.markdown("# Face Recognition")

def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

app_mode = st.sidebar.selectbox('Select Page',['Real Time Face Regconition','Upload Image Face Regconition']) 

if app_mode == 'Real Time Face Regconition':
    
    def visualize(input, faces, fps, thickness=2):
        if faces[1] is not None:
            for idx, face in enumerate(faces[1]):
                #print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

                coords = face[:-1].astype(np.int32)
                cv2.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
                cv2.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
                cv2.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
                cv2.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
                cv2.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
                cv2.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
        cv2.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    
    st.subheader('Nhận dạng khuôn mặt')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    if 'stop' not in st.session_state:
        st.session_state.stop = False
        stop = False

    press = st.button('Stop')
    if press:
        if st.session_state.stop == False:
            st.session_state.stop = True
            cap.release()
        else:
            st.session_state.stop = False

    print('Trang thai nhan Stop', st.session_state.stop)

    if 'frame_stop' not in st.session_state:
        frame_stop = cv2.imread('stop.jpg')
        st.session_state.frame_stop = frame_stop
        print('Đã load stop.jpg')

    if st.session_state.stop == True:
        pass
        #FRAME_WINDOW.image(st.session_state.frame_stop, channels='BGR')


    svc = joblib.load('pages/svc.pkl')
    mydict = ['BanKiet', 'BanNghia', 'BanNguyen', 'BanThanh', 'SangSang', 'ThayDuc']
    detector = cv2.FaceDetectorYN.create(
        'pages/face_detection_yunet_2022mar.onnx',
        "",
        (320, 320),
        0.9,
        0.3,
        5000)
    
    recognizer = cv2.FaceRecognizerSF.create(
    'pages/face_recognition_sface_2021dec.onnx',"")

    tm = cv2.TickMeter()

    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])

    dem = 0

    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        # Inference
        tm.start()
        faces = detector.detect(frame) # faces is a tuple
        tm.stop()
        
        if faces[1] is not None:
            face_align = recognizer.alignCrop(frame, faces[1][0])
            face_feature = recognizer.feature(face_align)
            test_predict = svc.predict(face_feature)
            result = mydict[test_predict[0]]
            cv2.putText(frame,result,(1,50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw results on the input image
        visualize(frame, faces, tm.getFPS())

        # Visualize results
        FRAME_WINDOW.image(frame, channels='BGR') 
    
   
else:

    st.title("Face Detection")
    st.write("Face detection is a central algorithm in computer vision. The algorithm implemented below is a Haar-Cascade Classifier. It detects several faces using OpenCV.")

    choice = st.radio("", ("Show Demo", "Browse an Image"))
    st.write()

    face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

    def detect_faces(our_image):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        new_img = np.array(our_image.convert('RGB'))
        img = cv2.cvtColor(new_img,1)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        col1, col2 = st.columns(2)

        col1.markdown("#### Original Image")
        plt.figure(figsize = (12,8))
        plt.imshow(our_image)
        col1.pyplot(use_column_width=True)

        scaleFactor = st.sidebar.slider("Scale Factor", 1.02,1.15,1.1,0.01)
        minNeighbors = st.sidebar.slider("Number of neighbors", 1, 15, 5, 1)
        minSize = st.sidebar.slider("Minimum Size", 10,50,20,1)

        #Detect Faces
        faces = face_cascade.detectMultiScale(gray,scaleFactor=scaleFactor,minNeighbors=minNeighbors,flags = cv2.CASCADE_SCALE_IMAGE)

        #Draw Bounding Box
        for (x,y,w,h) in faces:
            if w > minSize:
                cv2.rectangle(gray, (x,y), (x+w,y+h), (255,255,255), 5)

        col2.markdown("#### Detected Faces")
        plt.figure(figsize = (12,8))
        plt.imshow(gray, cmap = 'gray')
        col2.pyplot(use_column_width=True)
        if len(faces)>1:
            st.success("Found {} faces".format(len(faces)))
        else:
            st.success("Found {} face".format(len(faces)))

    if choice == "Browse an Image":
        st.set_option('deprecation.showfileUploaderEncoding', False)
        image_file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)  
            detect_faces(our_image)

    elif choice == "Show Demo":
        our_image = Image.open("images/human1.jpg")
        detect_faces(our_image)
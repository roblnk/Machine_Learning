# Name of project: Build a Machine Learning Application Website 
## HCMC University of Technology and Education 
## Faculty of Information Technology
### Members of group:

- Vũ Nguyễn Trung Khang - 20110277
- Võ Ngọc Quý - 20110709
- Hồ Thành Danh - 20110207

### Instructor: 
T.S Trần Tiến Đức
### Content:
- Face Recognition
- Fruits Recognition
- Calihousing
- Gradient Descent
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

### App framework and language:
Streamlit, Python
### Setup:
    pip install -r requirements.txt
#### How to training in your Face Recognition?
First go to direct image in folder training and paste your foler that have your personal photos,then go to folder training:

    cd training
    py training.py
    cd ..    
Next, go to FaceRecognition.py and set your listmodels orderly same as list folder arrived in folder image above:
```python
listmodels = ['BanKiet', 'BanNghia', 'BanNguyen', 'BanThanh', 'SangSang', 'ThayDuc'] 
```
### Build:
Local URL: http://localhost:8501

### Run project:
    streamlit run manage.py


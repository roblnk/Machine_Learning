import streamlit as st
st.markdown("# Cali Housing")
st.sidebar.markdown("# Cali Housing")

def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

app_mode = st.sidebar.selectbox('Select Page',['Decision_Tree_Regression','Linear_Regression','Random_Forest_Regression_Grid_Search_CV',
                                                'Random_Forest_Regression_Random_Search_CV', 'Random_Forest_Regression',
                                                'PhanNhomMedianIncome']) 

if app_mode=='Decision_Tree_Regression':
    st.title("Decision Tree Regression") 
    from pages.CaliHousing1.Decision_Tree_Regression import *

    # Them column income_cat dung de chia data
    housing["income_cat"] = pd.cut(housing["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])
    housing.replace(0, np.nan, inplace=True)

                   
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # Chia xong thi delete column income_cat
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    housing_num = housing.drop("ocean_proximity", axis=1)

    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])

    housing_prepared = full_pipeline.fit_transform(housing)

    # Training
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing_prepared, housing_labels)

    # Prediction
    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data)

    # Prediction 5 samples 
    st.write("Predictions:", tree_reg.predict(some_data_prepared))
    st.write("Labels:", list(some_labels))
    st.write('\n')

    # Tính sai số bình phương trung bình trên tập dữ liệu huấn luyện
    housing_predictions = tree_reg.predict(housing_prepared)
    mse_train = mean_squared_error(housing_labels, housing_predictions)
    rmse_train = np.sqrt(mse_train)
    st.write('Sai số bình phương trung bình - train:')
    st.write('%.2f' % rmse_train)

    # Tính sai số bình phương trung bình trên tập dữ liệu kiểm định chéo (cross-validation) 
    scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

    st.write('Sai số bình phương trung bình - cross-validation:')
    rmse_cross_validation = np.sqrt(-scores)
    display_scores(rmse_cross_validation)

    # Tính sai số bình phương trung bình trên tập dữ liệu kiểm tra (test)
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    X_test_prepared = full_pipeline.transform(X_test)
    y_predictions = tree_reg.predict(X_test_prepared)

    mse_test = mean_squared_error(y_test, y_predictions)
    rmse_test = np.sqrt(mse_test)
    st.write('Sai số bình phương trung bình - test:')
    st.write('%.2f' % rmse_test)

elif app_mode == 'Linear_Regression':
    st.title('Linear Regression')
    from pages.CaliHousing1.Linear_Regression import *
    # Them column income_cat dung de chia data
    housing["income_cat"] = pd.cut(housing["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])
    housing.replace(0, np.nan, inplace=True)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # Chia xong thi delete column income_cat
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    housing_num = housing.drop("ocean_proximity", axis=1)

    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])

    housing_prepared = full_pipeline.fit_transform(housing)

    # Training
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    # Prediction
    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data)
    # Prediction 5 samples 
    st.write("Predictions:", lin_reg.predict(some_data_prepared))
    st.write("Labels:", list(some_labels))
    st.write('\n')

    # Tính sai số bình phương trung bình trên tập dữ liệu huấn luyện
    housing_predictions = lin_reg.predict(housing_prepared)
    mse_train = mean_squared_error(housing_labels, housing_predictions)
    rmse_train = np.sqrt(mse_train)
    st.write('Sai số bình phương trung bình - train:')
    st.write('%.2f' % rmse_train)

    # Tính sai số bình phương trung bình trên tập dữ liệu kiểm định chéo (cross-validation) 
    scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

    st.write('Sai số bình phương trung bình - cross-validation:')
    rmse_cross_validation = np.sqrt(-scores)
    display_scores(rmse_cross_validation)

    # Tính sai số bình phương trung bình trên tập dữ liệu kiểm tra (test)
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    X_test_prepared = full_pipeline.transform(X_test)
    y_predictions = lin_reg.predict(X_test_prepared)

    mse_test = mean_squared_error(y_test, y_predictions)
    rmse_test = np.sqrt(mse_test)
    st.write('Sai số bình phương trung bình - test:')
    st.write('%.2f' % rmse_test)

elif app_mode == 'Random_Forest_Regression_Grid_Search_CV':
    st.title('Random Forest Regression Grid Search CV')
    from pages.CaliHousing1.Random_Forest_Regression_Grid_Search_CV import *
    # Them column income_cat dung de chia data
    housing["income_cat"] = pd.cut(housing["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])
    housing.replace(0, np.nan, inplace=True)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # Chia xong thi delete column income_cat
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    housing_num = housing.drop("ocean_proximity", axis=1)

    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])

    housing_prepared = full_pipeline.fit_transform(housing)

    param_grid = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
                {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
                ]
    # Training
    forest_reg = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, 
                            scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(housing_prepared, housing_labels)

    final_model = grid_search.best_estimator_

    # Prediction
    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data)
    # Prediction 5 samples 
    st.write("Predictions:", final_model.predict(some_data_prepared))
    st.write("Labels:", list(some_labels))
    st.write('\n')

    # Tính sai số bình phương trung bình trên tập dữ liệu huấn luyện
    housing_predictions = final_model.predict(housing_prepared)
    mse_train = mean_squared_error(housing_labels, housing_predictions)
    rmse_train = np.sqrt(mse_train)
    st.write('Sai số bình phương trung bình - train:')
    st.write('%.2f' % rmse_train)

    # Tính sai số bình phương trung bình trên tập dữ liệu kiểm định chéo (cross-validation) 
    scores = cross_val_score(final_model, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

    st.write('Sai số bình phương trung bình - cross-validation:')
    rmse_cross_validation = np.sqrt(-scores)
    display_scores(rmse_cross_validation)

    # Tính sai số bình phương trung bình trên tập dữ liệu kiểm tra (test)
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    X_test_prepared = full_pipeline.transform(X_test)
    y_predictions = final_model.predict(X_test_prepared)

    mse_test = mean_squared_error(y_test, y_predictions)
    rmse_test = np.sqrt(mse_test)
    st.write('Sai số bình phương trung bình - test:')
    st.write('%.2f' % rmse_test)

elif app_mode == 'Random_Forest_Regression_Random_Search_CV':
    st.title('Random Forest Regression Random Search CV')
    from pages.CaliHousing1.Random_Forest_Regression_Random_Search_CV import *
    # Them column income_cat dung de chia data
    housing["income_cat"] = pd.cut(housing["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])
    housing.replace(0, np.nan, inplace=True)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # Chia xong thi delete column income_cat
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    housing_num = housing.drop("ocean_proximity", axis=1)

    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])

    housing_prepared = full_pipeline.fit_transform(housing)

    param_distribs = {
            'n_estimators': randint(low=1, high=200),
            'max_features': randint(low=1, high=8),
        }

    # Training
    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                    n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
    rnd_search.fit(housing_prepared, housing_labels)

    final_model = rnd_search.best_estimator_

    # Prediction
    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data)
    # Prediction 5 samples 
    st.write("Predictions:", final_model.predict(some_data_prepared))
    st.write("Labels:", list(some_labels))
    st.write('\n')

    # Tính sai số bình phương trung bình trên tập dữ liệu huấn luyện
    housing_predictions = final_model.predict(housing_prepared)
    mse_train = mean_squared_error(housing_labels, housing_predictions)
    rmse_train = np.sqrt(mse_train)
    st.write('Sai số bình phương trung bình - train:')
    st.write('%.2f' % rmse_train)

    # Tính sai số bình phương trung bình trên tập dữ liệu kiểm định chéo (cross-validation) 
    scores = cross_val_score(final_model, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

    st.write('Sai số bình phương trung bình - cross-validation:')
    rmse_cross_validation = np.sqrt(-scores)
    display_scores(rmse_cross_validation)

    # Tính sai số bình phương trung bình trên tập dữ liệu kiểm tra (test)
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    X_test_prepared = full_pipeline.transform(X_test)
    y_predictions = final_model.predict(X_test_prepared)

    mse_test = mean_squared_error(y_test, y_predictions)
    rmse_test = np.sqrt(mse_test)
    st.write('Sai số bình phương trung bình - test:')
    st.write('%.2f' % rmse_test)


elif app_mode == 'Random_Forest_Regression':
    st.title('Random Forest Regression')
    from pages.CaliHousing1.Random_Forest_Regression import *
    # Them column income_cat dung de chia data
    housing["income_cat"] = pd.cut(housing["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])
    housing.replace(0, np.nan, inplace=True)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # Chia xong thi delete column income_cat
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    housing_num = housing.drop("ocean_proximity", axis=1)

    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])

    housing_prepared = full_pipeline.fit_transform(housing)

    # Training
    forest_reg = RandomForestRegressor()
    forest_reg.fit(housing_prepared, housing_labels)


    # Prediction
    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data)
    # Prediction 5 samples 
    st.write("Predictions:", forest_reg.predict(some_data_prepared))
    st.write("Labels:", list(some_labels))
    st.write('\n')

    # Tính sai số bình phương trung bình trên tập dữ liệu huấn luyện
    housing_predictions = forest_reg.predict(housing_prepared)
    mse_train = mean_squared_error(housing_labels, housing_predictions)
    rmse_train = np.sqrt(mse_train)
    st.write('Sai số bình phương trung bình - train:')
    st.write('%.2f' % rmse_train)

    # Tính sai số bình phương trung bình trên tập dữ liệu kiểm định chéo (cross-validation) 
    scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

    st.write('Sai số bình phương trung bình - cross-validation:')
    rmse_cross_validation = np.sqrt(-scores)
    display_scores(rmse_cross_validation)

    # Tính sai số bình phương trung bình trên tập dữ liệu kiểm tra (test)
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    X_test_prepared = full_pipeline.transform(X_test)
    y_predictions = forest_reg.predict(X_test_prepared)

    mse_test = mean_squared_error(y_test, y_predictions)
    rmse_test = np.sqrt(mse_test)
    st.write('Sai số bình phương trung bình - test:')
    st.write('%.2f' % rmse_test)
elif app_mode == 'PhanNhomMedianIncome':    
    st.title('Phân nhóm Median Income')
    from pages.CaliHousing1.PhanNhomMedianIncome import *
    fig, ax = plt.subplots()
    housing["income_cat"].hist()
    ax.hist(housing["income_cat"])
    st.pyplot(fig)
    st.write(housing)
    housing.to_csv('pages/CaliHousing1/housing.csv')
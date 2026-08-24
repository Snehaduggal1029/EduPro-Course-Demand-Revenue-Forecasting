import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EduPro Intelligence Hub",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background-color: #101827;
}

[data-testid="stSidebar"] * {
    color: white;
}

.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(135deg, #111827, #243b64);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 18px;
    color: #dbe4f0;
}

.kpi {
    background: white;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
}

.kpi-title {
    color: #6b7280;
    font-size: 14px;
}

.kpi-value {
    font-size: 30px;
    font-weight: 700;
    color: #111827;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 15px;
}

.prediction-card {
    padding: 25px;
    border-radius: 18px;
    background: white;
    border: 1px solid #e5e7eb;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
}

.prediction-number {
    font-size: 32px;
    font-weight: 700;
    color: #2563eb;
}

.info-box {
    padding: 20px;
    border-radius: 15px;
    background: #eef4ff;
    border-left: 5px solid #2563eb;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    users = pd.read_csv(
        "EduPro Online Platform.xlsx - Users (2).csv"
    )

    teachers = pd.read_csv(
        "EduPro Online Platform.xlsx - Teachers.csv"
    )

    courses = pd.read_csv(
        "EduPro Online Platform.xlsx - Courses.csv"
    )

    transactions = pd.read_csv(
        "EduPro Online Platform.xlsx - Transactions.csv"
    )

    return users, teachers, courses, transactions


try:

    users, teachers, courses, transactions = load_data()

except Exception as e:

    st.error("❌ Dataset files could not be loaded.")

    st.info("""
    Make sure these four CSV files are inside the same folder as app.py:

    1. EduPro Online Platform.xlsx - Users (2).csv
    2. EduPro Online Platform.xlsx - Teachers.csv
    3. EduPro Online Platform.xlsx - Courses.csv
    4. EduPro Online Platform.xlsx - Transactions.csv
    """)

    st.stop()


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

users.columns = users.columns.str.strip()
teachers.columns = teachers.columns.str.strip()
courses.columns = courses.columns.str.strip()
transactions.columns = transactions.columns.str.strip()


# ============================================================
# DATA PREPARATION
# ============================================================

transactions["TransactionDate"] = pd.to_datetime(
    transactions["TransactionDate"],
    dayfirst=True,
    errors="coerce"
)

transactions["Amount"] = pd.to_numeric(
    transactions["Amount"],
    errors="coerce"
).fillna(0)

courses["CoursePrice"] = pd.to_numeric(
    courses["CoursePrice"],
    errors="coerce"
).fillna(0)

courses["CourseDuration"] = pd.to_numeric(
    courses["CourseDuration"],
    errors="coerce"
).fillna(0)

courses["CourseRating"] = pd.to_numeric(
    courses["CourseRating"],
    errors="coerce"
)

courses["CourseRating"] = courses["CourseRating"].fillna(
    courses["CourseRating"].median()
)


# ============================================================
# CREATE COURSE-LEVEL DATASET
# ============================================================

enrollment_data = (
    transactions
    .groupby("CourseID")
    .agg(
        Enrollment_Count=("TransactionID", "count"),
        Course_Revenue=("Amount", "sum")
    )
    .reset_index()
)


course_ml = courses.merge(
    enrollment_data,
    on="CourseID",
    how="left"
)

course_ml["Enrollment_Count"] = (
    course_ml["Enrollment_Count"]
    .fillna(0)
)

course_ml["Course_Revenue"] = (
    course_ml["Course_Revenue"]
    .fillna(0)
)


# ============================================================
# FIND TEACHER INFORMATION THROUGH TRANSACTIONS
# ============================================================

course_teacher = (
    transactions[
        ["CourseID", "TeacherID"]
    ]
    .dropna()
    .drop_duplicates("CourseID")
)


course_ml = course_ml.merge(
    course_teacher,
    on="CourseID",
    how="left"
)


# ============================================================
# MERGE TEACHER DETAILS
# ============================================================

teacher_columns = [
    "TeacherID",
    "Expertise",
    "YearsOfExperience",
    "TeacherRating"
]

available_teacher_columns = [
    col
    for col in teacher_columns
    if col in teachers.columns
]


course_ml = course_ml.merge(
    teachers[available_teacher_columns],
    on="TeacherID",
    how="left"
)


# ============================================================
# CLEAN TEACHER FEATURES
# ============================================================

if "YearsOfExperience" not in course_ml.columns:

    course_ml["YearsOfExperience"] = 0


if "TeacherRating" not in course_ml.columns:

    course_ml["TeacherRating"] = 0


course_ml["YearsOfExperience"] = pd.to_numeric(
    course_ml["YearsOfExperience"],
    errors="coerce"
).fillna(0)


course_ml["TeacherRating"] = pd.to_numeric(
    course_ml["TeacherRating"],
    errors="coerce"
)


teacher_rating_median = course_ml["TeacherRating"].median()

if pd.isna(teacher_rating_median):

    teacher_rating_median = 0


course_ml["TeacherRating"] = (
    course_ml["TeacherRating"]
    .fillna(teacher_rating_median)
)


# ============================================================
# ADD REVENUE METRICS
# ============================================================

course_ml["Average_Revenue"] = np.where(
    course_ml["Enrollment_Count"] > 0,
    course_ml["Course_Revenue"] /
    course_ml["Enrollment_Count"],
    0
)


course_ml["Revenue_Per_Enrollment"] = np.where(
    course_ml["Enrollment_Count"] > 0,
    course_ml["Course_Revenue"] /
    course_ml["Enrollment_Count"],
    0
)


# ============================================================
# MACHINE LEARNING DATA
# ============================================================

feature_columns = [
    "CourseCategory",
    "CourseType",
    "CourseLevel",
    "CoursePrice",
    "CourseDuration",
    "CourseRating",
    "YearsOfExperience",
    "TeacherRating"
]


categorical_features = [
    "CourseCategory",
    "CourseType",
    "CourseLevel"
]


numeric_features = [
    "CoursePrice",
    "CourseDuration",
    "CourseRating",
    "YearsOfExperience",
    "TeacherRating"
]


# ============================================================
# TRAIN MACHINE LEARNING MODELS
# ============================================================

@st.cache_resource
def train_models(data):

    X = data[feature_columns].copy()

    y_enrollment = data[
        "Enrollment_Count"
    ].copy()

    y_revenue = data[
        "Course_Revenue"
    ].copy()

    # --------------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enrollment,
        test_size=0.2,
        random_state=42
    )

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X,
        y_revenue,
        test_size=0.2,
        random_state=42
    )


    # --------------------------------------------------------
    # Preprocessor
    # --------------------------------------------------------

    def create_preprocessor():

        return ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                    categorical_features
                ),
                (
                    "numeric",
                    "passthrough",
                    numeric_features
                )
            ]
        )


    # ========================================================
    # ENROLLMENT MODELS
    # ========================================================

    enrollment_linear = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor()
            ),
            (
                "model",
                LinearRegression()
            )
        ]
    )


    enrollment_rf = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor()
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42
                )
            )
        ]
    )


    enrollment_linear.fit(
        X_train,
        y_train
    )

    enrollment_rf.fit(
        X_train,
        y_train
    )


    linear_enrollment_pred = (
        enrollment_linear.predict(X_test)
    )

    rf_enrollment_pred = (
        enrollment_rf.predict(X_test)
    )


    enrollment_metrics = pd.DataFrame({

        "Model": [
            "Linear Regression",
            "Random Forest"
        ],

        "MAE": [
            mean_absolute_error(
                y_test,
                linear_enrollment_pred
            ),

            mean_absolute_error(
                y_test,
                rf_enrollment_pred
            )
        ],

        "RMSE": [
            np.sqrt(
                mean_squared_error(
                    y_test,
                    linear_enrollment_pred
                )
            ),

            np.sqrt(
                mean_squared_error(
                    y_test,
                    rf_enrollment_pred
                )
            )
        ],

        "R² Score": [
            r2_score(
                y_test,
                linear_enrollment_pred
            ),

            r2_score(
                y_test,
                rf_enrollment_pred
            )
        ]
    })


    # ========================================================
    # REVENUE MODELS
    # ========================================================

    revenue_linear = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor()
            ),
            (
                "model",
                LinearRegression()
            )
        ]
    )


    revenue_rf = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor()
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42
                )
            )
        ]
    )


    revenue_linear.fit(
        X_train_r,
        y_train_r
    )

    revenue_rf.fit(
        X_train_r,
        y_train_r
    )


    linear_revenue_pred = (
        revenue_linear.predict(X_test_r)
    )

    rf_revenue_pred = (
        revenue_rf.predict(X_test_r)
    )


    revenue_metrics = pd.DataFrame({

        "Model": [
            "Linear Regression",
            "Random Forest"
        ],

        "MAE": [
            mean_absolute_error(
                y_test_r,
                linear_revenue_pred
            ),

            mean_absolute_error(
                y_test_r,
                rf_revenue_pred
            )
        ],

        "RMSE": [
            np.sqrt(
                mean_squared_error(
                    y_test_r,
                    linear_revenue_pred
                )
            ),

            np.sqrt(
                mean_squared_error(
                    y_test_r,
                    rf_revenue_pred
                )
            )
        ],

        "R² Score": [
            r2_score(
                y_test_r,
                linear_revenue_pred
            ),

            r2_score(
                y_test_r,
                rf_revenue_pred
            )
        ]
    })


    # --------------------------------------------------------
    # Select Best Models
    # --------------------------------------------------------

    best_enrollment_model_name = (
        enrollment_metrics
        .sort_values(
            "MAE"
        )
        .iloc[0]["Model"]
    )


    best_revenue_model_name = (
        revenue_metrics
        .sort_values(
            "MAE"
        )
        .iloc[0]["Model"]
    )


    return (
        enrollment_linear,
        enrollment_rf,
        revenue_linear,
        revenue_rf,
        enrollment_metrics,
        revenue_metrics,
        best_enrollment_model_name,
        best_revenue_model_name
    )


# Train models once

(
    enrollment_linear,
    enrollment_rf,
    revenue_linear,
    revenue_rf,
    enrollment_metrics,
    revenue_metrics,
    best_enrollment_model_name,
    best_revenue_model_name
) = train_models(course_ml)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown("""
<div style="text-align:center; padding:10px;">
    <h1>🎓 EduPro</h1>
    <p>Intelligence Hub</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")


page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Executive Overview",
        "📊 Demand Intelligence",
        "💰 Revenue Intelligence",
        "🤖 Predictive Simulator",
        "🧠 Model Intelligence",
        "🔍 Feature Intelligence",
        "📚 Course Explorer",
        "📈 Course Performance",
        "🔥 Category & Market Insights",
        "💡 Executive Recommendations"
    ]
)


st.sidebar.markdown("---")

st.sidebar.info(
    "Predictive Analytics Platform\n\n"
    "Course Demand & Revenue Forecasting"
)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <h1>🎓 EduPro Intelligence Hub</h1>
    <p>
    Course Demand • Revenue Analytics • Machine Learning • Business Intelligence
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

if page == "🏠 Executive Overview":

    st.markdown(
        '<div class="section-title">Executive Overview</div>',
        unsafe_allow_html=True
    )


    total_courses = len(course_ml)

    total_enrollments = int(
        course_ml["Enrollment_Count"].sum()
    )

    total_revenue = (
        course_ml["Course_Revenue"].sum()
    )

    avg_rating = (
        course_ml["CourseRating"].mean()
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-title">Total Courses</div>
                <div class="kpi-value">{total_courses:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with c2:

        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-title">Total Enrollments</div>
                <div class="kpi-value">{total_enrollments:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with c3:

        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-title">Total Revenue</div>
                <div class="kpi-value">₹{total_revenue:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with c4:

        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-title">Average Course Rating</div>
                <div class="kpi-value">⭐ {avg_rating:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        '<div class="section-title">Business Snapshot</div>',
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)


    with col1:

        category_data = (
            course_ml
            .groupby("CourseCategory")[
                "Enrollment_Count"
            ]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        st.subheader(
            "📈 Enrollment by Category"
        )

        st.bar_chart(
            category_data
        )


    with col2:

        revenue_category = (
            course_ml
            .groupby("CourseCategory")[
                "Course_Revenue"
            ]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        st.subheader(
            "💰 Revenue by Category"
        )

        st.bar_chart(
            revenue_category
        )


# ============================================================
# DEMAND INTELLIGENCE
# ============================================================

elif page == "📊 Demand Intelligence":

    st.markdown(
        '<div class="section-title">📊 Demand Intelligence</div>',
        unsafe_allow_html=True
    )


    category_demand = (
        course_ml
        .groupby("CourseCategory")
        .agg(
            Enrollments=(
                "Enrollment_Count",
                "sum"
            ),

            Courses=(
                "CourseID",
                "count"
            )
        )
        .sort_values(
            "Enrollments",
            ascending=False
        )
    )


    category_demand["Avg_Enrollment"] = (
        category_demand["Enrollments"]
        /
        category_demand["Courses"]
    )


    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "Enrollment by Course Category"
        )

        st.bar_chart(
            category_demand[
                "Enrollments"
            ]
        )


    with col2:

        st.subheader(
            "Average Enrollments per Course"
        )

        st.bar_chart(
            category_demand[
                "Avg_Enrollment"
            ]
        )


    st.subheader(
        "📋 Demand Ranking"
    )


    display_demand = category_demand.copy()


    # Dynamic demand classification

    q1 = display_demand[
        "Enrollments"
    ].quantile(0.33)

    q2 = display_demand[
        "Enrollments"
    ].quantile(0.66)


    def classify_demand(value):

        if value <= q1:
            return "Low"

        elif value <= q2:
            return "Moderate"

        else:
            return "High"


    display_demand["Demand Level"] = (
        display_demand[
            "Enrollments"
        ]
        .apply(classify_demand)
    )


    st.dataframe(
        display_demand,
        use_container_width=True
    )


# ============================================================
# REVENUE INTELLIGENCE
# ============================================================

elif page == "💰 Revenue Intelligence":

    st.markdown(
        '<div class="section-title">💰 Revenue Intelligence</div>',
        unsafe_allow_html=True
    )


    revenue_category = (
        course_ml
        .groupby("CourseCategory")
        .agg(
            Revenue=(
                "Course_Revenue",
                "sum"
            ),

            Enrollments=(
                "Enrollment_Count",
                "sum"
            )
        )
        .sort_values(
            "Revenue",
            ascending=False
        )
    )


    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "Revenue by Category"
        )

        st.bar_chart(
            revenue_category[
                "Revenue"
            ]
        )


    with col2:

        st.subheader(
            "Revenue vs Enrollments"
        )

        st.scatter_chart(
            revenue_category,
            x="Enrollments",
            y="Revenue"
        )


    st.subheader(
        "💎 Top Revenue Courses"
    )


    top_revenue = (
        course_ml[
            [
                "CourseName",
                "CourseCategory",
                "Enrollment_Count",
                "Course_Revenue",
                "CourseRating"
            ]
        ]
        .sort_values(
            "Course_Revenue",
            ascending=False
        )
        .head(10)
    )


    st.dataframe(
        top_revenue,
        use_container_width=True
    )


# ============================================================
# PREDICTIVE SIMULATOR
# ============================================================

elif page == "🤖 Predictive Simulator":

    st.markdown(
        '<div class="section-title">🤖 Predictive Simulator</div>',
        unsafe_allow_html=True
    )


    st.markdown("""
    <div class="info-box">
        <b>How this works:</b><br>
        Configure the characteristics of a new course and the trained
        Machine Learning models will estimate expected student enrollments
        and course revenue.
    </div>
    """, unsafe_allow_html=True)


    st.markdown(
        "### 🎛️ Course Configuration"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        category_options = sorted(
            course_ml[
                "CourseCategory"
            ]
            .dropna()
            .unique()
            .tolist()
        )


        selected_category = st.selectbox(
            "Course Category",
            category_options
        )


        type_options = sorted(
            course_ml[
                "CourseType"
            ]
            .dropna()
            .unique()
            .tolist()
        )


        selected_type = st.selectbox(
            "Course Type",
            type_options
        )


        level_options = sorted(
            course_ml[
                "CourseLevel"
            ]
            .dropna()
            .unique()
            .tolist()
        )


        selected_level = st.selectbox(
            "Course Level",
            level_options
        )


    with col2:

        selected_price = st.number_input(
            "Course Price",
            min_value=0.0,
            max_value=5000.0,
            value=float(
                course_ml[
                    "CoursePrice"
                ].median()
            ),
            step=10.0
        )


        selected_duration = st.number_input(
            "Course Duration",
            min_value=1.0,
            max_value=100.0,
            value=float(
                course_ml[
                    "CourseDuration"
                ].median()
            ),
            step=1.0
        )


        selected_course_rating = st.slider(
            "Course Rating",
            min_value=0.0,
            max_value=5.0,
            value=float(
                course_ml[
                    "CourseRating"
                ].median()
            ),
            step=0.1
        )


    with col3:

        selected_experience = st.slider(
            "Teacher Experience",
            min_value=0,
            max_value=30,
            value=int(
                course_ml[
                    "YearsOfExperience"
                ].median()
            )
        )


        selected_teacher_rating = st.slider(
            "Teacher Rating",
            min_value=0.0,
            max_value=5.0,
            value=float(
                course_ml[
                    "TeacherRating"
                ].median()
            ),
            step=0.1
        )


    st.markdown("---")


    predict_button = st.button(
        "🚀 Predict Course Demand & Revenue",
        use_container_width=True
    )


    if predict_button:

        prediction_input = pd.DataFrame([{

            "CourseCategory":
                selected_category,

            "CourseType":
                selected_type,

            "CourseLevel":
                selected_level,

            "CoursePrice":
                selected_price,

            "CourseDuration":
                selected_duration,

            "CourseRating":
                selected_course_rating,

            "YearsOfExperience":
                selected_experience,

            "TeacherRating":
                selected_teacher_rating

        }])


        # ====================================================
        # ENROLLMENT PREDICTION
        # ====================================================

        if best_enrollment_model_name == "Random Forest":

            enrollment_prediction = (
                enrollment_rf
                .predict(
                    prediction_input
                )[0]
            )

        else:

            enrollment_prediction = (
                enrollment_linear
                .predict(
                    prediction_input
                )[0]
            )


        predicted_enrollment = max(
            0,
            enrollment_prediction
        )


        # ====================================================
        # REVENUE PREDICTION
        # ====================================================

        if best_revenue_model_name == "Linear Regression":

            revenue_prediction = (
                revenue_linear
                .predict(
                    prediction_input
                )[0]
            )

        else:

            revenue_prediction = (
                revenue_rf
                .predict(
                    prediction_input
                )[0]
            )


        predicted_revenue = max(
            0,
            revenue_prediction
        )


        # ====================================================
        # DEMAND CATEGORY
        # ====================================================

        average_enrollment = (
            course_ml[
                "Enrollment_Count"
            ].mean()
        )


        if predicted_enrollment < (
            average_enrollment * 0.75
        ):

            demand_category = "🔴 Low Demand"


        elif predicted_enrollment < (
            average_enrollment * 1.25
        ):

            demand_category = "🟡 Moderate Demand"


        else:

            demand_category = "🟢 High Demand"


        # ====================================================
        # RESULTS
        # ====================================================

        st.markdown(
            "### 🎯 Prediction Results"
        )


        p1, p2, p3 = st.columns(3)


        with p1:

            st.markdown(
                f"""
                <div class="prediction-card">
                    <h4>👥 Predicted Enrollments</h4>
                    <div class="prediction-number">
                        {predicted_enrollment:.0f}
                    </div>
                    <p>Expected students</p>
                </div>
                """,
                unsafe_allow_html=True
            )


        with p2:

            st.markdown(
                f"""
                <div class="prediction-card">
                    <h4>💰 Predicted Revenue</h4>
                    <div class="prediction-number">
                        ₹{predicted_revenue:,.0f}
                    </div>
                    <p>Expected course revenue</p>
                </div>
                """,
                unsafe_allow_html=True
            )


        with p3:

            st.markdown(
                f"""
                <div class="prediction-card">
                    <h4>📊 Expected Demand</h4>
                    <div class="prediction-number">
                        {demand_category}
                    </div>
                    <p>Based on historical demand</p>
                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # BUSINESS RECOMMENDATION
        # ====================================================

        st.markdown(
            "### 💡 Business Recommendation"
        )


        if demand_category == "🟢 High Demand":

            st.success(
                "This course configuration shows strong expected demand. "
                "Consider increasing marketing and instructor visibility."
            )


        elif demand_category == "🟡 Moderate Demand":

            st.warning(
                "Demand is expected to be moderate. "
                "Consider improving course promotion, content quality, "
                "or pricing strategy."
            )


        else:

            st.error(
                "Expected demand is relatively low. "
                "Consider revising the course price, level, content, "
                "or instructor strategy."
            )


# ============================================================
# MODEL INTELLIGENCE
# ============================================================

elif page == "🧠 Model Intelligence":

    st.markdown(
        '<div class="section-title">🤖 Model Intelligence</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Performance comparison of the machine learning models used by EduPro."
    )


    # ========================================================
    # ENROLLMENT PREDICTION
    # ========================================================

    st.markdown(
        "### 👥 Enrollment Prediction"
    )


    best_enrollment_row = (
        enrollment_metrics
        .sort_values(
            "MAE"
        )
        .iloc[0]
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Best Enrollment Model",
            best_enrollment_model_name,
            "Based on MAE"
        )


    with col2:

        st.metric(
            "Best Enrollment MAE",
            f"{best_enrollment_row['MAE']:.2f}"
        )


    with col3:

        st.metric(
            "Best Enrollment R²",
            f"{best_enrollment_row['R² Score']:.2f}"
        )


    enrollment_display = (
        enrollment_metrics.copy()
    )


    enrollment_display["MAE"] = (
        enrollment_display["MAE"]
        .round(2)
    )

    enrollment_display["RMSE"] = (
        enrollment_display["RMSE"]
        .round(2)
    )

    enrollment_display["R² Score"] = (
        enrollment_display["R² Score"]
        .round(3)
    )


    st.dataframe(
        enrollment_display,
        use_container_width=True,
        hide_index=True
    )


    st.markdown(
        "#### 📊 Enrollment Model Comparison"
    )


    enrollment_chart = (
        enrollment_metrics
        .set_index("Model")[
            ["MAE", "RMSE"]
        ]
    )


    st.bar_chart(
        enrollment_chart
    )


    # ========================================================
    # REVENUE PREDICTION
    # ========================================================

    st.markdown(
        "### 💰 Course Revenue Prediction"
    )


    best_revenue_row = (
        revenue_metrics
        .sort_values(
            "MAE"
        )
        .iloc[0]
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Best Revenue Model",
            best_revenue_model_name,
            "Based on MAE"
        )


    with col2:

        st.metric(
            "Best Revenue MAE",
            f"₹{best_revenue_row['MAE']:,.0f}"
        )


    with col3:

        st.metric(
            "Best Revenue R²",
            f"{best_revenue_row['R² Score'] * 100:.2f}%"
        )


    revenue_display = (
        revenue_metrics.copy()
    )


    revenue_display["MAE"] = (
        revenue_display["MAE"]
        .round(2)
    )

    revenue_display["RMSE"] = (
        revenue_display["RMSE"]
        .round(2)
    )

    revenue_display["R² Score"] = (
        revenue_display["R² Score"]
        .round(3)
    )


    st.dataframe(
        revenue_display,
        use_container_width=True,
        hide_index=True
    )


    st.markdown(
        "#### 📈 Revenue Model Comparison"
    )


    revenue_chart = (
        revenue_metrics
        .set_index("Model")[
            ["MAE", "RMSE"]
        ]
    )


    st.bar_chart(
        revenue_chart
    )


    # ========================================================
    # MODEL RECOMMENDATION
    # ========================================================

    st.markdown(
        "### 🧠 Model Selection Insight"
    )


    st.info(
        f"Random Forest and Linear Regression were evaluated for "
        f"both prediction tasks. Based on the test-set evaluation, "
        f"{best_enrollment_model_name} is selected for enrollment "
        f"prediction and {best_revenue_model_name} is selected for "
        f"revenue prediction."
    )


# ============================================================
# FEATURE INTELLIGENCE
# ============================================================

elif page == "🔍 Feature Intelligence":

    st.markdown(
        '<div class="section-title">🧠 Feature Intelligence</div>',
        unsafe_allow_html=True
    )


    numeric_columns = [
        "CoursePrice",
        "CourseDuration",
        "CourseRating",
        "Enrollment_Count",
        "Course_Revenue",
        "YearsOfExperience",
        "TeacherRating"
    ]


    correlation = (
        course_ml[
            numeric_columns
        ]
        .corr()
    )


    st.subheader(
        "📊 Correlation Matrix"
    )


    st.dataframe(
        correlation.round(3),
        use_container_width=True
    )


    # ========================================================
    # ENROLLMENT DRIVERS
    # ========================================================

    st.subheader(
        "Enrollment Drivers"
    )


    enrollment_corr = (
        correlation[
            "Enrollment_Count"
        ]
        .drop(
            "Enrollment_Count"
        )
        .sort_values(
            ascending=False
        )
    )


    st.bar_chart(
        enrollment_corr
    )


    # ========================================================
    # REVENUE DRIVERS
    # ========================================================

    st.subheader(
        "Revenue Drivers"
    )


    revenue_corr = (
        correlation[
            "Course_Revenue"
        ]
        .drop(
            "Course_Revenue"
        )
        .sort_values(
            ascending=False
        )
    )


    st.bar_chart(
        revenue_corr
    )


# ============================================================
# COURSE EXPLORER
# ============================================================

elif page == "📚 Course Explorer":

    st.markdown(
        '<div class="section-title">📚 Course Explorer</div>',
        unsafe_allow_html=True
    )


    category_filter = st.selectbox(
        "Filter by Category",
        ["All"]
        +
        sorted(
            course_ml[
                "CourseCategory"
            ]
            .dropna()
            .unique()
            .tolist()
        )
    )


    filtered_data = course_ml.copy()


    if category_filter != "All":

        filtered_data = filtered_data[
            filtered_data[
                "CourseCategory"
            ]
            ==
            category_filter
        ]


    st.dataframe(
        filtered_data[
            [
                "CourseID",
                "CourseName",
                "CourseCategory",
                "CourseType",
                "CourseLevel",
                "CoursePrice",
                "CourseDuration",
                "CourseRating",
                "Enrollment_Count",
                "Course_Revenue",
                "YearsOfExperience",
                "TeacherRating"
            ]
        ]
        .sort_values(
            "Enrollment_Count",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# COURSE PERFORMANCE INTELLIGENCE
# ============================================================

elif page == "📈 Course Performance":

    st.markdown(
        '<div class="section-title">📊 Course Performance Intelligence</div>',
        unsafe_allow_html=True
    )


    st.caption(
        "Identify the strongest courses based on enrollment, "
        "revenue and ratings."
    )


    # ========================================================
    # TOP COURSES BY ENROLLMENT
    # ========================================================

    top_enrollment = (
        course_ml[
            [
                "CourseName",
                "Enrollment_Count",
                "Course_Revenue",
                "CourseRating"
            ]
        ]
        .sort_values(
            "Enrollment_Count",
            ascending=False
        )
        .head(10)
    )


    # ========================================================
    # TOP COURSES BY REVENUE
    # ========================================================

    top_revenue = (
        course_ml[
            [
                "CourseName",
                "Course_Revenue",
                "Enrollment_Count",
                "CourseRating"
            ]
        ]
        .sort_values(
            "Course_Revenue",
            ascending=False
        )
        .head(10)
    )


    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            "### 👥 Top Courses by Enrollment"
        )


        enrollment_display = (
            top_enrollment[
                [
                    "CourseName",
                    "Enrollment_Count"
                ]
            ]
            .copy()
        )


        enrollment_display.columns = [
            "Course",
            "Enrollments"
        ]


        st.dataframe(
            enrollment_display,
            use_container_width=True,
            hide_index=True
        )


    with col2:

        st.markdown(
            "### 💰 Top Courses by Revenue"
        )


        revenue_display = (
            top_revenue[
                [
                    "CourseName",
                    "Course_Revenue"
                ]
            ]
            .copy()
        )


        revenue_display.columns = [
            "Course",
            "Revenue"
        ]


        revenue_display["Revenue"] = (
            revenue_display[
                "Revenue"
            ]
            .apply(
                lambda x:
                f"₹{x:,.0f}"
            )
        )


        st.dataframe(
            revenue_display,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # ENROLLMENT CHART
    # ========================================================

    st.markdown(
        "### 📈 Highest Enrolled Courses"
    )


    enrollment_chart = (
        top_enrollment
        .set_index(
            "CourseName"
        )[
            "Enrollment_Count"
        ]
        .sort_values(
            ascending=True
        )
    )


    st.bar_chart(
        enrollment_chart
    )


    # ========================================================
    # REVENUE CHART
    # ========================================================

    st.markdown(
        "### 💵 Highest Revenue Courses"
    )


    revenue_chart = (
        top_revenue
        .set_index(
            "CourseName"
        )[
            "Course_Revenue"
        ]
        .sort_values(
            ascending=True
        )
    )


    st.bar_chart(
        revenue_chart
    )


# ============================================================
# COURSE CATEGORY & MARKET INSIGHTS
# ============================================================

elif page == "🔥 Category & Market Insights":

    st.markdown(
        '<div class="section-title">🔥 Course Category & Market Insights</div>',
        unsafe_allow_html=True
    )


    st.caption(
        "Understand category-level demand, revenue and market performance."
    )


    # ========================================================
    # CATEGORY AGGREGATION
    # ========================================================

    category_analysis = (
        course_ml
        .groupby(
            "CourseCategory"
        )
        .agg(
            Total_Enrollments=(
                "Enrollment_Count",
                "sum"
            ),

            Total_Revenue=(
                "Course_Revenue",
                "sum"
            ),

            Number_of_Courses=(
                "CourseID",
                "count"
            ),

            Average_Rating=(
                "CourseRating",
                "mean"
            )
        )
        .reset_index()
    )


    category_analysis[
        "Avg_Enrollment_Per_Course"
    ] = (
        category_analysis[
            "Total_Enrollments"
        ]
        /
        category_analysis[
            "Number_of_Courses"
        ]
    )


    category_analysis = (
        category_analysis
        .sort_values(
            "Total_Enrollments",
            ascending=False
        )
    )


    # ========================================================
    # CATEGORY KPIs
    # ========================================================

    best_demand_category = (
        category_analysis
        .sort_values(
            "Total_Enrollments",
            ascending=False
        )
        .iloc[0][
            "CourseCategory"
        ]
    )


    best_revenue_category = (
        category_analysis
        .sort_values(
            "Total_Revenue",
            ascending=False
        )
        .iloc[0][
            "CourseCategory"
        ]
    )


    highest_rated_category = (
        category_analysis
        .sort_values(
            "Average_Rating",
            ascending=False
        )
        .iloc[0][
            "CourseCategory"
        ]
    )


    k1, k2, k3 = st.columns(3)


    with k1:

        st.metric(
            "🔥 Highest Demand Category",
            best_demand_category
        )


    with k2:

        st.metric(
            "💰 Highest Revenue Category",
            best_revenue_category
        )


    with k3:

        st.metric(
            "⭐ Highest Rated Category",
            highest_rated_category
        )


    # ========================================================
    # CATEGORY CHARTS
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            "### 👥 Enrollment by Category"
        )


        enrollment_category_chart = (
            category_analysis
            .set_index(
                "CourseCategory"
            )[
                "Total_Enrollments"
            ]
            .sort_values(
                ascending=True
            )
        )


        st.bar_chart(
            enrollment_category_chart
        )


    with col2:

        st.markdown(
            "### 💰 Revenue by Category"
        )


        revenue_category_chart = (
            category_analysis
            .set_index(
                "CourseCategory"
            )[
                "Total_Revenue"
            ]
            .sort_values(
                ascending=True
            )
        )


        st.bar_chart(
            revenue_category_chart
        )


    # ========================================================
    # CATEGORY TABLE
    # ========================================================

    st.markdown(
        "### 📋 Category Performance Table"
    )


    category_display = (
        category_analysis
        .copy()
    )


    category_display[
        "Total_Revenue"
    ] = (
        category_display[
            "Total_Revenue"
        ]
        .apply(
            lambda x:
            f"₹{x:,.0f}"
        )
    )


    category_display[
        "Average_Rating"
    ] = (
        category_display[
            "Average_Rating"
        ]
        .round(2)
    )


    category_display[
        "Avg_Enrollment_Per_Course"
    ] = (
        category_display[
            "Avg_Enrollment_Per_Course"
        ]
        .round(1)
    )


    st.dataframe(
        category_display,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# EXECUTIVE RECOMMENDATIONS
# ============================================================

elif page == "💡 Executive Recommendations":

    st.markdown(
        '<div class="section-title">💡 Executive Recommendations</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # BEST CATEGORIES
    # ========================================================

    best_demand_category = (
        course_ml
        .groupby(
            "CourseCategory"
        )[
            "Enrollment_Count"
        ]
        .sum()
        .idxmax()
    )


    best_revenue_category = (
        course_ml
        .groupby(
            "CourseCategory"
        )[
            "Course_Revenue"
        ]
        .sum()
        .idxmax()
    )


    # ========================================================
    # BEST COURSES
    # ========================================================

    best_course = (
        course_ml
        .sort_values(
            "Enrollment_Count",
            ascending=False
        )
        .iloc[0]
    )


    best_revenue_course = (
        course_ml
        .sort_values(
            "Course_Revenue",
            ascending=False
        )
        .iloc[0]
    )


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.success(
        f"🎯 **Highest Demand Category:** "
        f"{best_demand_category}"
    )


    st.info(
        f"💰 **Highest Revenue Category:** "
        f"{best_revenue_category}"
    )


    st.success(
        f"🏆 **Most Enrolled Course:** "
        f"{best_course['CourseName']}"
    )


    st.info(
        f"💎 **Top Revenue Course:** "
        f"{best_revenue_course['CourseName']}"
    )


    st.markdown(
        "### 📌 Strategic Actions"
    )


    st.markdown("""
    **1. Focus marketing on high-demand categories**

    Increase visibility and promotional activity for categories
    with consistently high enrollment.

    **2. Optimize course pricing**

    Use the Predictive Simulator to test different prices
    before launching a course.

    **3. Leverage experienced instructors**

    Teacher experience and ratings can contribute to stronger
    course performance.

    **4. Improve low-performing courses**

    Courses with weak enrollment should be reviewed for pricing,
    course level, duration, content and instructor quality.

    **5. Use ML predictions before launching new courses**

    The Predictive Simulator can be used as a decision-support
    tool before introducing new courses.

    **6. Prioritize high-revenue categories**

    Management should focus resources on categories that combine
    strong demand with high revenue potential.
    """)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "EduPro Intelligence Hub | "
    "Machine Learning & Business Intelligence Dashboard"
)
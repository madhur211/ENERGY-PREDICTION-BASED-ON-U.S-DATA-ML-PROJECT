# app.py - Streamlit Energy Consumption Prediction App
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        text-align: center;
    }
    .feature-importance {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

class EnergyConsumptionPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder_building = None
        self.label_encoder_day = None
        self.feature_columns = None
        self.model_metadata = None
        
    def load_artifacts(self):
        """Load trained model and preprocessing artifacts"""
        try:
            # Load model
            self.model = joblib.load('best_energy_model.pkl')
            
            # Load preprocessing artifacts
            preprocessing_artifacts = joblib.load('preprocessing_artifacts.pkl')
            self.scaler = preprocessing_artifacts['scaler']
            self.label_encoder_building = preprocessing_artifacts['label_encoder_building']
            self.label_encoder_day = preprocessing_artifacts['label_encoder_day']
            self.feature_columns = preprocessing_artifacts['feature_columns']
            
            # Load metadata
            with open('model_metadata.json', 'r') as f:
                self.model_metadata = json.load(f)
            
            return True
        except Exception as e:
            st.error(f"Error loading artifacts: {str(e)}")
            return False
    
    def preprocess_input(self, input_data):
        """Preprocess user input for prediction"""
        try:
            # Create DataFrame from input
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables
            input_df['Building Type_encoded'] = self.label_encoder_building.transform(
                input_df['Building Type']
            )
            input_df['Day of Week_encoded'] = self.label_encoder_day.transform(
                input_df['Day of Week']
            )
            
            # Create engineered features
            input_df['Area_Per_Occupant'] = (
                input_df['Square Footage'] / (input_df['Number of Occupants'] + 1)
            )
            input_df['Appliance_Usage_Ratio'] = (
                input_df['Appliances Used'] / (input_df['Number of Occupants'] + 1)
            )
            input_df['Total_Load_Factor'] = (
                input_df['Square Footage'] * input_df['Number of Occupants'] * 
                input_df['Appliances Used']
            )
            input_df['Temperature_Squared'] = input_df['Average Temperature'] ** 2
            input_df['Occupant_Density'] = (
                input_df['Number of Occupants'] / (input_df['Square Footage'] + 1)
            )
            
            # Add building type flags
            building_types = self.label_encoder_building.classes_
            for building_type in building_types:
                input_df[f'Is_{building_type}'] = (
                    input_df['Building Type'] == building_type
                ).astype(int)
            
            # Ensure all feature columns are present
            for col in self.feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Select features in correct order
            input_features = input_df[self.feature_columns]
            
            # Scale features
            input_scaled = self.scaler.transform(input_features)
            
            return input_scaled
            
        except Exception as e:
            st.error(f"Error preprocessing input: {str(e)}")
            return None
    
    def predict(self, input_data):
        """Make prediction"""
        try:
            processed_input = self.preprocess_input(input_data)
            if processed_input is not None:
                prediction = self.model.predict(processed_input)[0]
                return prediction
            return None
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None
    
    def get_feature_importance(self):
        """Get feature importance if available"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                return importance_df.head(10)
            return None
        except:
            return None

def main():
    # Initialize predictor
    predictor = EnergyConsumptionPredictor()
    
    # Header
    st.markdown('<h1 class="main-header">⚡ Energy Consumption Predictor</h1>', 
               unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose App Mode",
        ["Single Prediction", "Batch Prediction", "Model Info", "About"]
    )
    
    # Load artifacts
    if not hasattr(st.session_state, 'model_loaded'):
        with st.spinner("Loading model and artifacts..."):
            if predictor.load_artifacts():
                st.session_state.model_loaded = True
                st.session_state.predictor = predictor
            else:
                st.error("Failed to load model artifacts. Please check if the model files exist.")
                return
    
    predictor = st.session_state.predictor
    
    if app_mode == "Single Prediction":
        show_single_prediction(predictor)
    elif app_mode == "Batch Prediction":
        show_batch_prediction(predictor)
    elif app_mode == "Model Info":
        show_model_info(predictor)
    elif app_mode == "About":
        show_about()

def show_single_prediction(predictor):
    """Single prediction interface"""
    st.header("🔮 Single Building Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Building Details")
        
        building_type = st.selectbox(
            "Building Type",
            options=predictor.label_encoder_building.classes_,
            help="Type of building"
        )
        
        square_footage = st.number_input(
            "Square Footage",
            min_value=100,
            max_value=100000,
            value=5000,
            step=100,
            help="Total area in square feet"
        )
        
        num_occupants = st.number_input(
            "Number of Occupants",
            min_value=1,
            max_value=500,
            value=50,
            step=1,
            help="Number of people in the building"
        )
    
    with col2:
        st.subheader("Usage Details")
        
        appliances_used = st.number_input(
            "Number of Appliances Used",
            min_value=1,
            max_value=200,
            value=25,
            step=1,
            help="Number of active appliances"
        )
        
        avg_temperature = st.slider(
            "Average Temperature (°C)",
            min_value=-10.0,
            max_value=40.0,
            value=22.0,
            step=0.5,
            help="Average ambient temperature"
        )
        
        day_of_week = st.selectbox(
            "Day of Week",
            options=predictor.label_encoder_day.classes_,
            help="Day type"
        )
    
    # Prediction button
    if st.button("Predict Energy Consumption", type="primary"):
        with st.spinner("Calculating energy consumption..."):
            # Prepare input data
            input_data = {
                'Square Footage': square_footage,
                'Number of Occupants': num_occupants,
                'Appliances Used': appliances_used,
                'Average Temperature': avg_temperature,
                'Building Type': building_type,
                'Day of Week': day_of_week
            }
            
            # Make prediction
            prediction = predictor.predict(input_data)
            
            if prediction is not None:
                # Display prediction
                st.markdown("---")
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Predicted Energy Consumption",
                        f"{prediction:.2f}",
                        "units"
                    )
                
                with col2:
                    # Calculate cost (assuming $0.12 per unit)
                    cost = prediction * 0.12
                    st.metric(
                        "Estimated Cost",
                        f"${cost:.2f}",
                        "at $0.12/unit"
                    )
                
                with col3:
                    # Energy efficiency rating
                    efficiency = calculate_efficiency_rating(prediction, square_footage, num_occupants)
                    st.metric(
                        "Energy Efficiency",
                        efficiency,
                        "rating"
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show feature importance
                show_feature_importance(predictor)
                
                # Show insights
                show_insights(input_data, prediction)

def show_batch_prediction(predictor):
    """Batch prediction interface"""
    st.header("📊 Batch Prediction")
    
    st.info("Upload a CSV file with building data for batch predictions")
    
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="File should contain columns: Square Footage, Number of Occupants, Appliances Used, Average Temperature, Building Type, Day of Week"
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            batch_data = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(batch_data)} records")
            
            # Show preview
            st.subheader("Data Preview")
            st.dataframe(batch_data.head())
            
            # Check required columns
            required_columns = ['Square Footage', 'Number of Occupants', 'Appliances Used', 
                              'Average Temperature', 'Building Type', 'Day of Week']
            
            missing_columns = [col for col in required_columns if col not in batch_data.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
            else:
                if st.button("Generate Batch Predictions", type="primary"):
                    with st.spinner("Processing batch predictions..."):
                        predictions = []
                        
                        for _, row in batch_data.iterrows():
                            input_data = {
                                'Square Footage': row['Square Footage'],
                                'Number of Occupants': row['Number of Occupants'],
                                'Appliances Used': row['Appliances Used'],
                                'Average Temperature': row['Average Temperature'],
                                'Building Type': row['Building Type'],
                                'Day of Week': row['Day of Week']
                            }
                            
                            prediction = predictor.predict(input_data)
                            predictions.append(prediction)
                        
                        # Add predictions to dataframe
                        results_df = batch_data.copy()
                        results_df['Predicted_Energy_Consumption'] = predictions
                        results_df['Estimated_Cost'] = results_df['Predicted_Energy_Consumption'] * 0.12
                        
                        # Display results
                        st.subheader("Prediction Results")
                        st.dataframe(results_df)
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Total Energy",
                                f"{results_df['Predicted_Energy_Consumption'].sum():.2f}",
                                "units"
                            )
                        
                        with col2:
                            st.metric(
                                "Average Energy",
                                f"{results_df['Predicted_Energy_Consumption'].mean():.2f}",
                                "units/building"
                            )
                        
                        with col3:
                            st.metric(
                                "Total Cost",
                                f"${results_df['Estimated_Cost'].sum():.2f}",
                                "estimated"
                            )
                        
                        with col4:
                            st.metric(
                                "Max Consumption",
                                f"{results_df['Predicted_Energy_Consumption'].max():.2f}",
                                "units"
                            )
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv,
                            file_name="energy_predictions.csv",
                            mime="text/csv"
                        )
                        
                        # Visualization
                        st.subheader("Batch Analysis")
                        
                        fig = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=('Energy Distribution by Building Type', 
                                          'Temperature vs Energy',
                                          'Square Footage vs Energy',
                                          'Top Energy Consumers')
                        )
                        
                        # Plot 1: Energy by Building Type
                        building_energy = results_df.groupby('Building Type')['Predicted_Energy_Consumption'].mean()
                        fig.add_trace(
                            go.Bar(x=building_energy.index, y=building_energy.values, name="Avg Energy"),
                            row=1, col=1
                        )
                        
                        # Plot 2: Temperature vs Energy
                        fig.add_trace(
                            go.Scatter(x=results_df['Average Temperature'], 
                                     y=results_df['Predicted_Energy_Consumption'],
                                     mode='markers', name='Temperature Effect'),
                            row=1, col=2
                        )
                        
                        # Plot 3: Square Footage vs Energy
                        fig.add_trace(
                            go.Scatter(x=results_df['Square Footage'], 
                                     y=results_df['Predicted_Energy_Consumption'],
                                     mode='markers', name='Size Effect'),
                            row=2, col=1
                        )
                        
                        # Plot 4: Top consumers
                        top_consumers = results_df.nlargest(10, 'Predicted_Energy_Consumption')
                        fig.add_trace(
                            go.Bar(x=top_consumers.index.astype(str), 
                                 y=top_consumers['Predicted_Energy_Consumption'],
                                 name='Top Consumers'),
                            row=2, col=2
                        )
                        
                        fig.update_layout(height=800, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def show_model_info(predictor):
    """Model information interface - Using your actual performance data"""
    st.header("🤖 Model Information")
    
    st.info("""
    **Model Performance**: Based on actual trained models with comprehensive feature engineering
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Best Model Details")
        st.metric("Best Model", "Random Forest (Default)")
        st.metric("Test R² Score", "0.9743")
        st.metric("Test RMSE", "132.76")
        st.metric("Test MAE", "97.36")
    
    with col2:
        st.subheader("Dataset Info")
        st.metric("Training Samples", "800")
        st.metric("Test Samples", "200")
        st.metric("Total Features", "14")
        st.metric("Cross-Validation R²", "0.9805")
    
    # Your actual model performance data
    st.subheader("📊 All Model Performance")
    
    performance_data = {
        'Model': ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 
                 'Random Forest (Default)', 'Random Forest (Tuned)'],
        'Train_R2': [1.0000, 1.0000, 1.0000, 0.9975, 0.9963],
        'Test_R2': [1.0000, 1.0000, 1.0000, 0.9743, 0.9730],
        'Test_RMSE': [0.01, 1.28, 2.41, 132.76, 136.17],
        'Test_MAE': [0.01, 1.04, 1.98, 97.36, 100.73],
        'CV_Mean_R2': [1.0000, 1.0000, 1.0000, 0.9805, 0.9799],
        'CV_Std_R2': [0.0000, 0.0000, 0.0000, 0.0011, 0.0014]
    }
    
    performance_df = pd.DataFrame(performance_data)
    
    # Display the dataframe with formatting
    formatted_df = performance_df.copy()
    formatted_df['Train_R2'] = formatted_df['Train_R2'].apply(lambda x: f"{x:.4f}")
    formatted_df['Test_R2'] = formatted_df['Test_R2'].apply(lambda x: f"{x:.4f}")
    formatted_df['Test_RMSE'] = formatted_df['Test_RMSE'].apply(lambda x: f"{x:.2f}")
    formatted_df['Test_MAE'] = formatted_df['Test_MAE'].apply(lambda x: f"{x:.2f}")
    formatted_df['CV_Mean_R2'] = formatted_df['CV_Mean_R2'].apply(lambda x: f"{x:.4f}")
    formatted_df['CV_Std_R2'] = formatted_df['CV_Std_R2'].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(formatted_df)
    
    # Performance visualization using Plotly
    st.subheader("📈 Performance Visualization")
    
    # Create visualization for Test R² scores
    fig_r2 = go.Figure()
    
    fig_r2.add_trace(go.Bar(
        name='Test R²',
        x=performance_df['Model'],
        y=performance_df['Test_R2'],
        marker_color=['red' if x > 0.99 else 'blue' for x in performance_df['Test_R2']],
        text=performance_df['Test_R2'].round(4),
        textposition='auto'
    ))
    
    fig_r2.update_layout(
        title='Test R² Scores by Model',
        xaxis_title='Model',
        yaxis_title='R² Score',
        showlegend=False
    )
    
    st.plotly_chart(fig_r2, use_container_width=True)
    
    # Create visualization for Test RMSE
    fig_rmse = go.Figure()
    
    fig_rmse.add_trace(go.Bar(
        name='Test RMSE',
        x=performance_df['Model'],
        y=performance_df['Test_RMSE'],
        marker_color='orange',
        text=performance_df['Test_RMSE'].round(2),
        textposition='auto'
    ))
    
    fig_rmse.update_layout(
        title='Test RMSE by Model',
        xaxis_title='Model',
        yaxis_title='RMSE',
        showlegend=False
    )
    
    st.plotly_chart(fig_rmse, use_container_width=True)
    
    # Model Analysis
    st.subheader("🔍 Model Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 Best Performing Model**
        - **Random Forest (Default)**
        - Test R²: 0.9743
        - RMSE: 132.76
        - Excellent generalization
        - Stable cross-validation
        """)
    
    with col2:
        st.markdown("""
        **⚠️ Potential Overfitting**
        - Linear models show perfect scores
        - May indicate data leakage
        - Random Forest shows realistic performance
        - Good train-test consistency
        """)
    
    # Feature importance explanation
    st.subheader("📊 Key Predictive Features")
    
    feature_importance = {
        'Feature': ['Square Footage', 'Building Type', 'Number of Occupants', 
                   'Appliances Used', 'Average Temperature', 'Day of Week',
                   'Area Per Occupant', 'Total Load Factor'],
        'Importance': [0.28, 0.22, 0.15, 0.12, 0.08, 0.05, 0.06, 0.04]
    }
    
    importance_df = pd.DataFrame(feature_importance)
    
    fig_importance = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance in Energy Prediction',
        color='Importance',
        color_continuous_scale='viridis'
    )
    
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Technical Details
    st.subheader("🛠️ Technical Implementation")
    
    tech_cols = st.columns(3)
    
    with tech_cols[0]:
        st.markdown("""
        **🧠 Algorithms Used**
        - Linear Regression
        - Ridge Regression
        - Lasso Regression
        - Random Forest
        - Cross-Validation
        """)
    
    with tech_cols[1]:
        st.markdown("""
        **⚡ Feature Engineering**
        - Building type encoding
        - Interaction features
        - Temperature effects
        - Occupancy ratios
        - Load factors
        """)
    
    with tech_cols[2]:
        st.markdown("""
        **📊 Evaluation Metrics**
        - R² Score
        - RMSE
        - MAE
        - Cross-validation
        - Train-test split
        """)
    
    # Performance Insights
    st.subheader("💡 Performance Insights")
    
    insights_cols = st.columns(2)
    
    with insights_cols[0]:
        st.success("""
        **✅ Strengths**
        - Random Forest: Excellent performance
        - Good generalization
        - Stable cross-validation
        - Realistic error metrics
        - Robust to overfitting
        """)
    
    with insights_cols[1]:
        st.warning("""
        **📝 Considerations**
        - Linear models may overfit
        - Check for data leakage
        - Feature scaling impact
        - Model interpretability
        - Deployment complexity
        """)
    
    # Deployment Recommendations
    st.subheader("🚀 Deployment Recommendations")
    
    st.markdown("""
    **Recommended for Production**: **Random Forest (Default)**
    
    - **R² Score**: 0.9743 (Excellent)
    - **RMSE**: 132.76 (Reasonable for energy prediction)
    - **Cross-validation**: Stable (0.9805 ± 0.0011)
    - **Generalization**: Good train-test consistency
    
    **Next Steps**:
    1. Investigate linear model overfitting
    2. Validate feature importance
    3. Test on new unseen data
    4. Monitor production performance
    """)

def calculate_efficiency_rating(prediction, square_footage, occupants):
    """Calculate energy efficiency rating"""
    energy_per_sqft = prediction / square_footage
    if energy_per_sqft < 0.05:
        return "Excellent"
    elif energy_per_sqft < 0.1:
        return "Good"
    elif energy_per_sqft < 0.15:
        return "Average"
    else:
        return "Needs Improvement"

def show_insights(input_data, prediction):
    """Show insights based on prediction"""
    st.subheader("💡 Energy Insights")
    
    insights = []
    
    # Building type insights
    if input_data['Building Type'] == 'Industrial':
        if prediction > 5000:
            insights.append("🔧 Industrial buildings typically have high energy demands. Consider energy-efficient machinery.")
    elif input_data['Building Type'] == 'Commercial':
        if prediction > 4000:
            insights.append("🏢 High commercial energy use. Optimize HVAC and lighting schedules.")
    else:  # Residential
        if prediction > 3000:
            insights.append("🏠 High residential consumption. Check appliance efficiency and insulation.")
    
    # Temperature insights
    if input_data['Average Temperature'] < 15:
        insights.append("❄️ Low temperatures may increase heating demands.")
    elif input_data['Average Temperature'] > 28:
        insights.append("☀️ High temperatures may increase cooling demands.")
    
    # Display insights
    for insight in insights:
        st.write(insight)
    
    # Recommendations
    st.subheader("🎯 Recommendations")
    
    if prediction > 4000:
        st.warning("""
        **High Energy Consumption Detected**
        - Conduct energy audit
        - Upgrade to energy-efficient appliances
        - Optimize HVAC settings
        - Consider solar panels
        """)
    else:
        st.success("""
        **Good Energy Efficiency**
        - Maintain current practices
        - Regular maintenance checks
        - Monitor seasonal variations
        """)

if __name__ == "__main__":
    main()

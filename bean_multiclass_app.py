import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Bean Classification App",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #4CAF50;
        background-color: #f0f8f0;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and preprocessors
@st.cache_resource
def load_model_artifacts():
    """Load all required model artifacts"""
    try:
        model = joblib.load('best_bean_classifier_model.pkl')
        scaler = pickle.load(open('standard_scaler.pkl', 'rb'))
        label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
        
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return model, scaler, label_encoder, metadata
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.stop()

# Load artifacts
model, scaler, label_encoder, metadata = load_model_artifacts()

# App Header
st.title("🫘 Bean Classification System")
st.markdown("### Classify dry bean varieties using computer vision features")

# Sidebar
with st.sidebar:
    st.header("📊 Model Information")
    st.info(f"""
    **Model**: {metadata['model_name']}  
    **Accuracy**: {metadata['test_accuracy']:.2%}  
    **F1 Score**: {metadata['f1_score']:.4f}  
    **Classes**: {metadata['n_classes']}  
    **Features**: {metadata['n_features']}  
    **Training Date**: {metadata['training_date']}
    """)
    
    st.header("🔍 Bean Classes")
    for bean_class in metadata['classes']:
        st.write(f"• {bean_class}")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Single Prediction", "📦 Batch Prediction", "📈 Model Info", "ℹ️ About"])

# Tab 1: Single Prediction
with tab1:
    st.header("Single Bean Classification")
    st.markdown("Enter the features of a single bean to get its classification.")
    
    col1, col2 = st.columns(2)
    
    # Feature input fields
    features = {}
    feature_list = metadata['features']
    
    # Divide features into two columns
    mid_point = len(feature_list) // 2
    
    with col1:
        st.subheader("📏 Size Features")
        for feature in feature_list[:mid_point]:
            features[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.4f",
                help=f"Enter the {feature} value"
            )
    
    with col2:
        st.subheader("📐 Shape Features")
        for feature in feature_list[mid_point:]:
            features[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.4f",
                help=f"Enter the {feature} value"
            )
    
    # Predict button
    if st.button("🔮 Classify Bean", type="primary", use_container_width=True):
        if all(v == 0.0 for v in features.values()):
            st.warning("⚠️ Please enter feature values before prediction.")
        else:
            # Create input dataframe
            input_df = pd.DataFrame([features])
            
            # Scale features
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            predicted_class = label_encoder.inverse_transform([prediction])[0]
            
            # Get probability if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_scaled)[0]
                confidence = probabilities.max() * 100
                
                # Display results
                st.markdown("---")
                st.success("✅ Classification Complete!")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2 style="text-align: center; color: #2E7D32;">Predicted Class</h2>
                        <h1 style="text-align: center; color: #1B5E20; font-size: 3rem;">{predicted_class}</h1>
                        <p style="text-align: center; font-size: 1.2rem;">
                            Confidence: <strong>{confidence:.2f}%</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability distribution
                st.markdown("### 📊 Class Probabilities")
                prob_df = pd.DataFrame({
                    'Class': label_encoder.classes_,
                    'Probability': probabilities
                }).sort_values('Probability', ascending=False)
                
                fig = px.bar(
                    prob_df,
                    x='Probability',
                    y='Class',
                    orientation='h',
                    text=prob_df['Probability'].apply(lambda x: f'{x*100:.2f}%'),
                    color='Probability',
                    color_continuous_scale='Greens'
                )
                fig.update_layout(
                    showlegend=False,
                    height=400,
                    xaxis_title="Probability",
                    yaxis_title="Bean Class"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success(f"✅ Predicted Class: **{predicted_class}**")
            
            # Show input values
            with st.expander("📋 View Input Values"):
                st.dataframe(input_df.T, use_container_width=True)

# Tab 2: Batch Prediction
with tab2:
    st.header("Batch Bean Classification")
    st.markdown("Upload a CSV file with multiple bean measurements for batch prediction.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with the same features as the training data"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Found {len(batch_df)} samples.")
            
            # Show preview
            with st.expander("👀 Preview Data"):
                st.dataframe(batch_df.head(10), use_container_width=True)
            
            # Check if all required features are present
            missing_features = set(metadata['features']) - set(batch_df.columns)
            if missing_features:
                st.error(f"❌ Missing features: {missing_features}")
            else:
                if st.button("🚀 Run Batch Prediction", type="primary"):
                    with st.spinner("Processing predictions..."):
                        # Ensure features are in correct order
                        batch_input = batch_df[metadata['features']]
                        
                        # Scale features
                        batch_scaled = scaler.transform(batch_input)
                        
                        # Make predictions
                        predictions = model.predict(batch_scaled)
                        predicted_classes = label_encoder.inverse_transform(predictions)
                        
                        # Get probabilities if available
                        if hasattr(model, 'predict_proba'):
                            probabilities = model.predict_proba(batch_scaled)
                            confidences = probabilities.max(axis=1) * 100
                        else:
                            confidences = [100.0] * len(predictions)
                        
                        # Create results dataframe
                        results_df = batch_df.copy()
                        results_df['Predicted_Class'] = predicted_classes
                        results_df['Confidence_%'] = confidences
                        
                        st.success("✅ Batch prediction completed!")
                        
                        # Display statistics
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Total Samples", len(results_df)) 
                        with col3:
                            st.metric("Average Confidence", f"{confidences.mean():.2f}%")
                        with col4:
                            st.metric("Unique Classes", len(np.unique(predicted_classes))) 
                        
                        # Class distribution
                        st.markdown("### 📊 Prediction Distribution")
                        class_counts = pd.Series(predicted_classes).value_counts()
                        
                        fig = px.pie(
                            values=class_counts.values,
                            names=class_counts.index,
                            title="Distribution of Predicted Classes"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show results table
                        st.markdown("### 📋 Prediction Results")
                        st.dataframe(
                            results_df[['Predicted_Class', 'Confidence_%']],
                            use_container_width=True
                        )
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"bean_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    else:
        st.info("👆 Please upload a CSV file to begin batch prediction.")
        
        # Show example format
        st.markdown("### 📝 Expected CSV Format")
        example_df = pd.DataFrame({feature: [0.0] for feature in metadata['features']})
        st.dataframe(example_df, use_container_width=True)
        
        # Download example template
        csv_template = example_df.to_csv(index=False)
        st.download_button(
            label="📄 Download Template CSV",
            data=csv_template,
            file_name="bean_classification_template.csv",
            mime="text/csv"
        )

# Tab 3: Model Information
with tab3:
    st.header("📈 Model Performance & Information")
    
    # Model metrics
    st.subheader("🎯 Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Accuracy",
            value=f"{metadata['test_accuracy']:.2%}",
            delta="Test Set"
        )
    
    with col2:
        st.metric(
            label="F1 Score",
            value=f"{metadata['f1_score']:.4f}",
            delta="Weighted"
        )
    
    with col3:
        st.metric(
            label="Classes",
            value=metadata['n_classes']
        )
    
    with col4:
        st.metric(
            label="Features",
            value=metadata['n_features']
        )
    
    st.markdown("---")
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        st.subheader("🔍 Feature Importance")
        
        importance_df = pd.DataFrame({
            'Feature': metadata['features'],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            importance_df.head(15),
            x='Importance',
            y='Feature',
            orientation='h',
            title="Top 15 Most Important Features",
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📋 View All Feature Importances"):
            st.dataframe(importance_df, use_container_width=True)
    
    st.markdown("---")
    
    # Model details
    st.subheader("ℹ️ Model Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Model Type**: {metadata['model_name']}  
        **Model Class**: {metadata['model_type']}  
        **Balancing Method**: {metadata['balancing_method']}  
        **Training Date**: {metadata['training_date']}
        """)
    
    with col2:
        st.markdown(f"""
        **Number of Features**: {metadata['n_features']}  
        **Number of Classes**: {metadata['n_classes']}  
        **Test Accuracy**: {metadata['test_accuracy']:.4f}  
        **F1 Score**: {metadata['f1_score']:.4f}
        """)
    
    # Feature list
    with st.expander("📝 Complete Feature List"):
        feature_df = pd.DataFrame({
            'Index': range(1, len(metadata['features']) + 1),
            'Feature Name': metadata['features']
        })
        st.dataframe(feature_df, use_container_width=True)
    
    # Class list
    with st.expander("🫘 Bean Classes"):
        class_df = pd.DataFrame({
            'Index': range(len(metadata['classes'])),
            'Class Name': metadata['classes']
        })
        st.dataframe(class_df, use_container_width=True)

# Tab 4: About
with tab4:
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ## Bean Classification System
    
    This application uses machine learning to classify dry bean varieties based on their physical characteristics 
    measured through computer vision algorithms.
    
    ### 📊 Dataset Information
    
    The model was trained on a comprehensive dataset of dry bean samples with the following features:
    
    #### Size Features:
    - **Area (A)**: The area of a bean zone and the number of pixels within its boundaries
    - **Perimeter (P)**: Bean circumference, the length of its border
    - **Major Axis Length (L)**: The distance between the ends of the longest line drawable from a bean
    - **Minor Axis Length (l)**: The longest line drawable perpendicular to the main axis
    - **Convex Area (C)**: Number of pixels in the smallest convex polygon containing the bean seed
    - **Equivalent Diameter (Ed)**: Diameter of a circle with the same area as the bean
    
    #### Shape Features:
    - **Aspect Ratio (K)**: Relationship between major and minor axes
    - **Eccentricity (Ec)**: Eccentricity of the ellipse having the same moments as the region
    - **Extent (Ex)**: Ratio of pixels in the bounding box to the bean area
    - **Solidity (S)**: Ratio of pixels in the convex shell to those in the bean
    - **Roundness (R)**: Calculated as (4πA)/(P²)
    - **Compactness (CO)**: Measures roundness as Ed/L
    - **Shape Factors (SF1-SF4)**: Additional geometric descriptors
    
    ### 🎯 Bean Classes
    
    The model can classify beans into the following varieties:
    """)
    
    for idx, bean_class in enumerate(metadata['classes'], 1):
        st.markdown(f"{idx}. **{bean_class}**")
    
    st.markdown("""
    ### 🔬 Model Performance
    
    The model has been trained and evaluated using:
    - **Stratified train-test split** to maintain class distribution
    - **Cross-validation** for robust performance estimation
    - **Multiple evaluation metrics** (Accuracy, Precision, Recall, F1-Score)
    - **Class imbalance handling** techniques for fair classification
    
    ### 🚀 How to Use
    
    #### Single Prediction:
    1. Navigate to the "Single Prediction" tab
    2. Enter the feature values for your bean sample
    3. Click "Classify Bean" to get the prediction
    4. View the predicted class and confidence scores
    
    #### Batch Prediction:
    1. Navigate to the "Batch Prediction" tab
    2. Download the template CSV file
    3. Fill in your data following the template format
    4. Upload your completed CSV file
    5. Click "Run Batch Prediction"
    6. Download the results with predictions
    
    ### 📚 Technical Details
    
    - **Machine Learning Framework**: Scikit-learn
    - **Preprocessing**: StandardScaler for feature normalization
    - **Model Type**: {model_type}
    - **Programming Language**: Python
    - **Web Framework**: Streamlit
    
    ### 📝 Notes
    
    - All feature values should be numerical
    - Features are automatically scaled using the same scaler from training
    - Predictions include confidence scores (probability estimates)
    - The model works best with data similar to its training distribution
    
    ### ⚠️ Limitations
    
    - The model's accuracy depends on the quality of input measurements
    - Computer vision measurements should follow the same methodology as training data
    - Performance may vary for bean varieties not well-represented in training data
    - Regular retraining with new data is recommended for maintaining accuracy
    
    ### 🔄 Version Information
    
    - **Model Version**: 1.0
    - **Last Updated**: {metadata['training_date']}
    - **Model Accuracy**: {metadata['test_accuracy']:.2%}
    - **F1 Score**: {metadata['f1_score']:.4f}
    
    ### 📧 Contact & Support
    
    For questions, issues, or suggestions about this application, please contact the development team.
    
    ---
    
    *Built with ❤️ using Streamlit and Scikit-learn*
    """)
    
    # System information
    with st.expander("🖥️ System Information"):
        st.code(f"""
Python Libraries Used:
- streamlit: {st.__version__}
- pandas: {pd.__version__}
- numpy: {np.__version__}
- scikit-learn: (check version)
- joblib: (check version)

Model Information:
- Model Type: {metadata['model_type']}
- Number of Parameters: {metadata['n_features']}
- Classes: {metadata['n_classes']}
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>Bean Classification System v1.0 | © 2024 | Powered by Machine Learning</p>
    <p>Model Accuracy: {:.2%} | F1 Score: {:.4f}</p>
</div>
""".format(metadata['test_accuracy'], metadata['f1_score']), unsafe_allow_html=True)
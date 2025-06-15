import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Personality Prediction Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .extrovert {
        background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
        border: 3px solid #4caf50;
        color: #2e7d32;
    }
    .introvert {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border: 3px solid #2196f3;
        color: #1565c0;
    }
    .feature-importance {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #ff4757, #ffa502, #2ed573);
    }
    .insight-box {
        background-color: #f1f3f4;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #4285f4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    model_files = {
        'Random Forest': 'models/RandomForest.pkl',
        'Logistic Regression': 'models/LogisticRegression.pkl',
        'LightGBM': 'models/LightGBM.pkl',
        'Naive Bayes': 'models/NaiveBayes.pkl'
    }
    
    for name, filename in model_files.items():
        try:
            with open(filename, 'rb') as f:
                models[name] = pickle.load(f)
        except FileNotFoundError:
            st.warning(f"Model file {filename} not found. Please ensure all model files are in the models directory.")
            models[name] = None
    
    return models

def create_feature_visualization(features, feature_names):
    """Create a radar chart for input features"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=features,
        theta=feature_names,
        fill='toself',
        name='Input Features',
        line_color='rgb(31, 119, 180)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 15]  # Updated to accommodate friends circle max of 15
            )
        ),
        showlegend=False,
        title="Feature Profile",
        title_x=0.5
    )
    
    return fig

def get_prediction_explanation(prediction, probability=None):
    """Generate explanation for the prediction"""
    if prediction == 1:  # Introvert
        explanation = """
        **🧘‍♂️ Introvert Characteristics:**
        - Energized by solitude and quiet time
        - Reflective and thoughtful in decision-making
        - Prefers smaller, intimate social groups
        - Thinks carefully before speaking
        - Enjoys deep, meaningful conversations
        - Needs time alone to recharge after social activities
        - Often described as calm, reserved, and observant
        """
        result_text = "Based on your responses, you tend to be more **introverted**. You likely enjoy quiet environments and prefer deep connections over large social gatherings."
    else:  # Extrovert
        explanation = """
        **🎉 Extrovert Characteristics:**
        - Energized by social interactions and activity
        - Outgoing and expressive in communication
        - Comfortable in large group settings
        - Tends to think out loud and process externally
        - Enjoys meeting new people and networking
        - Gains energy from being around others
        - Often described as enthusiastic, talkative, and assertive
        """
        result_text = "Based on your responses, you tend to be more **extroverted**. You likely thrive in social situations and gain energy from interacting with others."
    
    return explanation, result_text

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Personality Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Predict Extrovert vs Introvert personality based on behavioral features</p>', unsafe_allow_html=True)
    
    # Load models
    models = load_models()
    
    # Sidebar for inputs
    st.sidebar.markdown('<h2 class="sub-header">📊 Input Features</h2>', unsafe_allow_html=True)
    
    # Feature inputs based on your dataset structure - CORRECT ORDER with actual data ranges
    feature_descriptions = [
        'Time spent Alone (0-11)',
        'Social Event Attendance (0-10)', 
        'Going Outside (0-10)',
        'Friends Circle Size (0-15)',
        'Post Frequency (0-10)',
        'Stage Fear (0=No, 1=Yes)',
        'Drained after Socializing (0=No, 1=Yes)'
    ]
    
    features = []
    
    # 1. Time spent alone (range: 0-10 from your data)
    features.append(st.sidebar.slider(
        feature_descriptions[0],
        min_value=0.0, max_value=11.0, value=5.0, step=0.1,
        key='time_alone',
        help="How much time do you prefer to spend alone? (0=Never, 11=Always)"
    ))
    
    # 2. Social event attendance (range: 0-10 from your data)
    features.append(st.sidebar.slider(
        feature_descriptions[1],
        min_value=0, max_value=10, value=5, step=1,
        key='social_events',
        help="How often do you attend social events? (0=Never, 10=Very Often)"
    ))
    
    # 3. Going outside (range: 0-10 from your data)
    features.append(st.sidebar.slider(
        feature_descriptions[2],
        min_value=0, max_value=7, value=5, step=1,
        key='going_outside',
        help="How often do you go outside for activities? (0=Never, 7=Very Often)"
    ))
    
    # 4. Friends circle size (range: 0-15 from your data)
    features.append(st.sidebar.slider(
        feature_descriptions[3],
        min_value=0, max_value=15, value=7, step = 1,
        key='friends_circle',
        help="How many close friends do you have?"
    ))
    
    # 5. Post frequency (range: 0-10 from your data)
    features.append(st.sidebar.slider(
        feature_descriptions[4],
        min_value=0, max_value=10, value=5, step= 1,
        key='post_frequency',
        help="How often do you post on social media? (0=Never, 10=Very Often)"
    ))
    
    # 6. Stage fear (binary: 0 or 1)
    features.append(st.sidebar.selectbox(
        feature_descriptions[5],
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        key='stage_fear',
        help="Do you experience stage fear or public speaking anxiety?"
    ))
    
    # 7. Drained after socializing (binary: 0 or 1)
    features.append(st.sidebar.selectbox(
        feature_descriptions[6],
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        key='drained_socializing',
        help="Do you feel drained after socializing?"
    ))
    
    feature_names = ['Time Alone', 'Social Events', 'Going Outside', 'Friends Circle', 
                    'Post Freq.', 'Stage Fear', 'Drained Social.']
    
    # Model selection
    st.sidebar.markdown('<h2 class="sub-header">🤖 Model Selection</h2>', unsafe_allow_html=True)
    available_models = [name for name, model in models.items() if model is not None]
    
    if not available_models:
        st.error("No models available. Please ensure model files are in the correct directory.")
        return
    
    selected_model = st.sidebar.selectbox(
        "Choose a prediction model:",
        available_models,
        help="Select the machine learning model for prediction"
    )
    
    # Model information
    model_info = {
        'Random Forest': 'Ensemble method using multiple decision trees for robust predictions',
        'Logistic Regression': 'Linear model optimized for binary classification tasks',
        'LightGBM': 'Gradient boosting framework with high performance and efficiency',
        'Naive Bayes': 'Probabilistic classifier based on Bayes theorem with feature independence'
    }
    
    st.sidebar.info(f"**{selected_model}:** {model_info.get(selected_model, 'Advanced ML model')}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📈 Feature Visualization</h2>', unsafe_allow_html=True)
        
        # Create and display radar chart
        radar_fig = create_feature_visualization(features, feature_names)
        st.plotly_chart(radar_fig, use_container_width=True)
        
        # Feature summary table
        feature_df = pd.DataFrame({
            'Feature': ['Time Alone', 'Social Events', 'Going Outside', 'Friends Circle', 
                       'Post Frequency', 'Stage Fear', 'Drained Socializing'],
            'Value': features,
            'Description': feature_descriptions
        })
        st.dataframe(feature_df, use_container_width=True)
    
    with col2:
        st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
        
        # Prediction button
        predict_button = st.button("🔮 Predict Personality", type="primary", use_container_width=True)
        
        if predict_button:
            if selected_model and models[selected_model] is not None:
                try:
                    # Prepare input data
                    input_data = np.array(features).reshape(1, -1)
                    
                    # Make prediction
                    model = models[selected_model]
                    prediction = model.predict(input_data)[0]
                    
                    # Get prediction probability if available
                    try:
                        probabilities = model.predict_proba(input_data)[0]
                        confidence = max(probabilities) * 100
                    except:
                        confidence = None
                    
                    # Create spacing
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Display main prediction result
                    if prediction == 1:  # Introvert
                        st.markdown(
                            '<div class="prediction-box introvert">🧘‍♂️ INTROVERT</div>',
                            unsafe_allow_html=True
                        )
                    else:  # Extrovert
                        st.markdown(
                            '<div class="prediction-box extrovert">🎉 EXTROVERT</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Get explanation and result text
                    explanation, result_text = get_prediction_explanation(prediction, confidence/100 if confidence else None)
                    
                    # Show result interpretation
                    st.markdown(f"**🎯 Result:** {result_text}")
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Display metrics in organized layout
                    if confidence:
                        col_conf, col_model = st.columns([1, 1])
                        with col_conf:
                            st.metric(
                                label="Confidence",
                                value=f"{confidence:.1f}%"
                            )
                        with col_model:
                            # Shorten model names for better display
                            model_short_names = {
                                'Random Forest': 'Random Forest',
                                'Logistic Regression': 'Logistic Reg.',
                                'LightGBM': 'LightGBM',
                                'Naive Bayes': 'Naive Bayes'
                            }
                            display_name = model_short_names.get(selected_model, selected_model)
                            st.metric(
                                label="Model",
                                value=display_name
                            )
                        
                        # Add confidence progress bar
                        st.markdown("**Confidence Visualization:**")
                        st.progress(confidence/100)
                        
                        # Confidence level indicator
                        if confidence >= 80:
                            st.success("🟢 **High Confidence** - Very reliable prediction!")
                        elif confidence >= 60:
                            st.info("🟡 **Moderate Confidence** - Good prediction reliability.")
                        else:
                            st.warning("🟠 **Low Confidence** - Consider reviewing your inputs.")
                    else:
                        # Shorten model names for better display
                        model_short_names = {
                            'Random Forest': 'Random Forest',
                            'Logistic Regression': 'Logistic Reg.',
                            'LightGBM': 'LightGBM',
                            'Naive Bayes': 'Naive Bayes'
                        }
                        display_name = model_short_names.get(selected_model, selected_model)
                        st.metric(
                            label="Model",
                            value=display_name
                        )
                        st.info("Confidence score not available for this model")
                    
                    # Add spacing before detailed explanation
                    st.markdown("---")
                    
                    # Detailed personality explanation
                    st.markdown("### 📚 Personality Insights")
                    st.markdown(explanation)
                    
                    # Personal profile analysis
                    st.markdown("---")
                    st.markdown("### 🔍 Your Profile Analysis")
                    
                    # Create insights container
                    insights_container = st.container()
                    with insights_container:
                        insights = []
                        
                        # Analyze key behavioral patterns
                        if features[0] >= 7:  # High time alone
                            insights.append("🏠 **Solitude Preference:** You value personal space and quiet time")
                        elif features[0] <= 3:
                            insights.append("👥 **Social Preference:** You enjoy being around others most of the time")
                        
                        if features[5] == 1:  # Has stage fear (now index 5)
                            insights.append("🎤 **Public Speaking:** You experience anxiety in presentation situations")
                        else:
                            insights.append("🎤 **Public Speaking:** You're comfortable with public presentations")
                        
                        if features[1] >= 7:  # High social events (now index 1)
                            insights.append("🎉 **Social Activity:** You actively seek out social gatherings and events")
                        elif features[1] <= 3:
                            insights.append("🏡 **Quiet Activities:** You prefer calm, low-key social interactions")
                        
                        if features[6] == 1:  # Drained after socializing (now index 6)
                            insights.append("⚡ **Energy Pattern:** Social interactions tend to drain your energy")
                        else:
                            insights.append("⚡ **Energy Pattern:** You gain energy from social interactions")
                        
                        if features[3] >= 12:  # Large friend circle (now index 3, max is 15)
                            insights.append("🌐 **Social Network:** You maintain a wide circle of close relationships")
                        elif features[3] <= 4:  # Adjusted for max 15
                            insights.append("💎 **Close Bonds:** You prefer a smaller, intimate circle of friends")
                        
                        if features[4] >= 7:  # High posting (now index 4)
                            insights.append("📱 **Digital Presence:** You're very active on social media platforms")
                        elif features[4] <= 3:
                            insights.append("📱 **Digital Presence:** You maintain a low-key social media presence")
                        
                        # Display insights in a clean format
                        for i, insight in enumerate(insights, 1):
                            st.markdown(f"{insight}")
                    
                    # Personalized recommendations
                    st.markdown("---")
                    st.markdown("### 💡 Personalized Recommendations")
                    
                    recommendations_container = st.container()
                    with recommendations_container:
                        if prediction == 1:  # Introvert recommendations
                            recommendations = [
                                "🏠 **Workspace Design:** Create a quiet, distraction-free environment for peak productivity",
                                "📅 **Energy Management:** Schedule regular alone time to recharge between social activities",
                                "👥 **Relationship Focus:** Invest in deeper connections rather than expanding your network",
                                "🎯 **Social Preparation:** Plan ahead for social events to manage energy and reduce stress",
                                "✍️ **Communication Style:** Leverage written communication when you need time to process thoughts"
                            ]
                        else:  # Extrovert recommendations
                            recommendations = [
                                "🤝 **Collaboration:** Actively seek team-based projects and group work opportunities",
                                "🌐 **Networking:** Regularly attend professional and social networking events",
                                "👂 **Active Listening:** Balance your natural tendency to speak with focused listening",
                                "🏃 **Group Activities:** Join clubs, sports teams, or group fitness classes",
                                "📞 **Communication:** Use phone calls and face-to-face meetings over text when possible"
                            ]
                        
                        for recommendation in recommendations:
                            st.markdown(f"• {recommendation}")
                
                except Exception as e:
                    st.error(f"❌ **Error making prediction:** {str(e)}")
                    st.info("Please check that all model files are properly loaded and try again.")
            else:
                st.error("❌ **Model unavailable:** Selected model is not loaded. Please check model files.")
        
        else:
            # Show instructions when no prediction is made
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background-color: #f8f9fa; border-radius: 10px; margin: 1rem 0; border: 1px solid #dee2e6;">
                <h4 style="color: #495057; margin-bottom: 1rem;">👆 Click the button above to get your personality prediction!</h4>
                <p style="color: #6c757d; margin: 0;">Adjust the sliders in the sidebar to match your preferences, then select a model and predict.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Model performance information
        st.markdown("---")
        st.markdown('<h3 class="sub-header">📊 Model Performance</h3>', unsafe_allow_html=True)
        
        performance_data = {
            'Model': ['Random Forest', 'Logistic Regression', 'LightGBM', 'Naive Bayes'],
            'Accuracy': ['89.11%', '91.53%', '91.53%', '92.00%'],
            'Precision': ['0.9074', '0.8983', '0.8983', '0.8983'],
            'Recall': ['0.8522', '0.9217', '0.9217', '0.9217'],
            'F1-Score': ['0.8789', '0.9099', '0.9099', '0.9099']
        }
        
        perf_df = pd.DataFrame(performance_data)
        st.dataframe(perf_df, use_container_width=True)
        
        st.info("💡 **Tip:** Try different models to see how they compare for your personality profile!")
    
    # Additional information section
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown('<h3 class="sub-header">🧭 How It Works</h3>', unsafe_allow_html=True)
        st.markdown("""
        1. **Input your behavioral characteristics** using the sidebar controls
        2. **Select a machine learning model** from the dropdown
        3. **Click "Predict Personality"** to get your result
        4. **View your personality type** with confidence score and explanation
        
        The models analyze patterns in your responses to determine whether you lean more toward introversion or extroversion.
        """)
    
    with col4:
        st.markdown('<h3 class="sub-header">📚 About the Features</h3>', unsafe_allow_html=True)
        st.markdown("""
        - **Time Alone**: Preference for solitary activities
        - **Stage Fear**: Anxiety in public speaking situations
        - **Social Events**: Frequency of attending gatherings
        - **Going Outside**: Engagement in outdoor activities
        - **Drained After Socializing**: Energy levels post-social interaction
        - **Friends Circle**: Size of close social network
        - **Post Frequency**: Social media engagement level
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666; font-size: 0.9rem;">Built with Streamlit | Machine Learning for Personality Prediction</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
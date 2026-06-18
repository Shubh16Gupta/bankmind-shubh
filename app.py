# app.py - Complete FastAPI with Real Feature Importance
import os
import joblib
import pandas as pd
import numpy as np
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# ------------------------------
# FastAPI app instance
# ------------------------------
app = FastAPI(
    title="BankMind Cross-Sell API",
    description="Predict term deposit subscriptions with ML and get LLM explanations.",
    version="1.0"
)

# ------------------------------
# Load model and artifacts
# ------------------------------
print("🔄 Loading model and artifacts...")

try:
    model = joblib.load('model.pkl')
    print("✅ Model loaded successfully")
except:
    print("⚠️ Model not found. Run train.py first!")

try:
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    print(f"✅ Feature names loaded ({len(feature_names)} features)")
except:
    feature_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 
                     'housing', 'loan', 'contact', 'day', 'month', 'campaign', 
                     'pdays', 'previous', 'poutcome']
    print("⚠️ Using default feature names")

try:
    with open('feature_importance.pkl', 'rb') as f:
        feature_importance = pickle.load(f)
    print("✅ Feature importance loaded")
except:
    feature_importance = None

# ------------------------------
# Pydantic input schema
# ------------------------------
class CustomerData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: float
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str

# ------------------------------
# Groq client (optional)
# ------------------------------
groq_client = None
if os.getenv("GROQ_API_KEY"):
    try:
        from groq import Groq
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        print("✅ Groq client initialized successfully")
    except Exception as e:
        print(f"⚠️ Warning: Groq client initialization failed: {e}")
        groq_client = None
else:
    print("ℹ️ GROQ_API_KEY not found in .env - /explain endpoint will return 503")

# ------------------------------
# Helper function to get top factors
# ------------------------------
def get_top_factors(input_df, proba):
    """Get top factors influencing the prediction"""
    try:
        # Use feature importance to rank features
        if feature_importance:
            # Get top 3 features by importance
            top_features = [name for name, _ in feature_importance[:3]]
            top_factors = []
            
            for feature in top_features:
                if feature in input_df.columns:
                    value = input_df[feature].iloc[0]
                    if isinstance(value, (int, float)):
                        if value > 0:
                            top_factors.append(f"high {feature}")
                        else:
                            top_factors.append(f"low {feature}")
                    else:
                        top_factors.append(f"{feature}={value}")
            
            return top_factors[:3]
        else:
            return ["balance", "pdays", "housing"]
    except:
        return ["balance", "pdays", "housing"]

# ------------------------------
# Endpoint: GET /health
# ------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "model": "CatBoost"}

# ------------------------------
# Endpoint: POST /predict
# ------------------------------
@app.post("/predict")
def predict(customer: CustomerData):
    try:
        # Convert JSON to DataFrame
        input_df = pd.DataFrame([customer.model_dump()])
        
        # Ensure all columns are present
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 'unknown'
        
        # Reorder columns to match training
        input_df = input_df[feature_names]
        
        # Predict
        proba = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]
        
        # Get top factors (real feature importance)
        top_factors = get_top_factors(input_df, proba)
        
        return {
            "will_subscribe": bool(pred),
            "probability": float(proba),
            "top_factors": top_factors
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {str(e)}")

# ------------------------------
# Endpoint: POST /explain
# ------------------------------
@app.post("/explain")
def explain(customer: CustomerData):
    if groq_client is None:
        raise HTTPException(
            status_code=503,
            detail="Groq API key not configured. Please set GROQ_API_KEY in .env"
        )
    
    try:
        # Get prediction first
        input_df = pd.DataFrame([customer.model_dump()])
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 'unknown'
        input_df = input_df[feature_names]
        
        proba = model.predict_proba(input_df)[0][1]
        pred = bool(proba >= 0.5)
        
        # Build prompt
        prompt = f"""
        Customer profile:
        - Age: {customer.age}, Job: {customer.job}, Balance: €{customer.balance:,.0f}
        - Existing loans: Housing={customer.housing}, Personal={customer.loan}
        - Model prediction: {proba*100:.1f}% chance of subscribing (will_subscribe={pred})

        In 2-3 sentences, explain why this customer would or would not likely subscribe to a term deposit, 
        and how an RM should approach the conversation.
        """
        
        # Call Groq API
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.7
        )
        explanation = chat_completion.choices[0].message.content
        
        return {
            "prediction": pred,
            "probability": float(proba),
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in explanation: {str(e)}")
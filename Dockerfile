# FiberPulse Production — Python Dependencies
# تثبيت: pip install -r requirements.txt

# ─── Web Framework ────────────────────────────────────────────────────────
fastapi==0.111.0
uvicorn[standard]==0.29.0
python-multipart==0.0.9
aiofiles==23.2.1

# ─── Database ─────────────────────────────────────────────────────────────
sqlalchemy==2.0.30
psycopg2-binary==2.9.9        # PostgreSQL / TimescaleDB driver

# ─── Cache & Real-time ────────────────────────────────────────────────────
redis==5.0.4

# ─── ML Core ──────────────────────────────────────────────────────────────
tensorflow==2.16.1             # LSTM model
scikit-learn==1.4.2            # preprocessing, metrics, cross-validation
numpy==1.26.4
pandas==2.2.2
joblib==1.4.2                  # model serialization

# ─── XAI (Explainability) — قلب الابتكار ────────────────────────────────
shap==0.45.1                   # SHAP values للتفسير

# ─── Validation ───────────────────────────────────────────────────────────
pydantic==2.7.1

# ─── Security ─────────────────────────────────────────────────────────────
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4

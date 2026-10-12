"""
FiberPulse Production — Database Layer
TimescaleDB (PostgreSQL) — مصمم خصيصاً لـ Time-Series Rx Power
"""
from sqlalchemy import (create_engine, Column, Integer, Float, String,
                        Boolean, DateTime, Text, ForeignKey, Index)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://fiberpulse:password@localhost:5432/fiberpulse"
)

engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ─── Tables ───────────────────────────────────────────────────────────────

class Subscriber(Base):
    """جدول المشتركين — بيانات ثابتة لكل ONT"""
    __tablename__ = "subscribers"

    id            = Column(Integer, primary_key=True, index=True)
    external_id   = Column(String(50), unique=True, index=True)   # ID من نظام AT
    pon_group_id  = Column(Integer, index=True)
    wilaya_code   = Column(Integer)
    wilaya_name   = Column(String(60))
    fiber_length_km = Column(Float, default=3.0)
    splitter_ratio  = Column(String(10), default='1:32')
    olt_class       = Column(String(5), default='B+')             # B+ / C+ / C++
    helix_factor    = Column(Float, default=1.04)
    olt_lat         = Column(Float)
    olt_lng         = Column(Float)
    onu_lat         = Column(Float)
    onu_lng         = Column(Float)
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime, default=datetime.utcnow)

    measurements = relationship("RxMeasurement", back_populates="subscriber",
                                cascade="all, delete-orphan")
    predictions  = relationship("Prediction", back_populates="subscriber",
                                cascade="all, delete-orphan")
    faults       = relationship("FaultHistory", back_populates="subscriber",
                                cascade="all, delete-orphan")
    alerts       = relationship("Alert", back_populates="subscriber",
                                cascade="all, delete-orphan")


class RxMeasurement(Base):
    """
    Hypertable — يُحوَّل بعد إنشاء الجدول:
        SELECT create_hypertable('rx_measurements', 'measured_at');

    580,000 صف لـ 200 مشترك × 8 سنوات × قراءة يومية
    TimescaleDB يستعيدها في < 10ms
    """
    __tablename__ = "rx_measurements"

    id            = Column(Integer, primary_key=True)
    subscriber_id = Column(Integer, ForeignKey("subscribers.id",
                           ondelete="CASCADE"), index=True)
    measured_at   = Column(DateTime, index=True, nullable=False)
    rx_power_dbm  = Column(Float, nullable=False)
    tx_power_dbm  = Column(Float)
    ber           = Column(Float)                                  # Bit Error Rate
    wavelength_nm = Column(Integer, default=1490)
    source        = Column(String(20), default='olt')              # olt|otdr|manual

    subscriber = relationship("Subscriber", back_populates="measurements")

    __table_args__ = (
        Index("idx_rx_sub_time", "subscriber_id", "measured_at"),
    )


class Prediction(Base):
    """
    كل تنبؤ يُحفظ مع SHAP values كاملة
    بعد 3 أشهر: يُحدَّث was_correct = True/False
    → هذا يبني dataset validation حقيقي تلقائياً مع الوقت
    """
    __tablename__ = "predictions"

    id                    = Column(Integer, primary_key=True)
    subscriber_id         = Column(Integer, ForeignKey("subscribers.id",
                                   ondelete="CASCADE"), index=True)
    predicted_at          = Column(DateTime, default=datetime.utcnow)
    risk_category         = Column(String(20))    # مستقر/مراقبة/خطر/حرج
    risk_score            = Column(Float)          # 0–100
    fault_type            = Column(String(30))     # CONTAMINATION/MACROBEND/etc
    months_to_critical    = Column(Float)          # NULL = مستقر
    confidence_pct        = Column(Float)
    shap_values           = Column(JSONB)          # XAI explanation كاملة
    survival_m1           = Column(Float)
    survival_m2           = Column(Float)
    survival_m3           = Column(Float)
    was_correct           = Column(Boolean)        # NULL=لم يُتحقق بعد
    verified_at           = Column(DateTime)
    model_version         = Column(String(20))

    subscriber = relationship("Subscriber", back_populates="predictions")


class FaultHistory(Base):
    """سجل الأعطال الحقيقية — يُستخدم كـ ground truth للتدريب"""
    __tablename__ = "fault_history"

    id                    = Column(Integer, primary_key=True)
    subscriber_id         = Column(Integer, ForeignKey("subscribers.id",
                                   ondelete="CASCADE"), index=True)
    fault_date            = Column(DateTime, nullable=False)
    fault_type            = Column(String(30))
    location_type         = Column(String(30))
    estimated_distance_m  = Column(Float)
    actual_distance_m     = Column(Float)          # يُضاف بعد الإصلاح
    resolution_hours      = Column(Float)
    technician_notes      = Column(Text)
    otdr_data             = Column(JSONB)
    prediction_id         = Column(Integer, ForeignKey("predictions.id"))

    subscriber = relationship("Subscriber", back_populates="faults")


class Alert(Base):
    """تنبيهات حرجة — يُرسل عبر WebSocket + NMS"""
    __tablename__ = "alerts"

    id                = Column(Integer, primary_key=True)
    subscriber_id     = Column(Integer, ForeignKey("subscribers.id",
                               ondelete="CASCADE"), index=True)
    created_at        = Column(DateTime, default=datetime.utcnow)
    alert_type        = Column(String(30))   # CRITICAL/PON_OUTAGE/FLAPPING
    message           = Column(Text)
    is_acknowledged   = Column(Boolean, default=False)
    acknowledged_by   = Column(String(50))
    acknowledged_at   = Column(DateTime)

    subscriber = relationship("Subscriber", back_populates="alerts")


class ModelMetrics(Base):
    """سجل أداء النموذج عبر الزمن — للورقة العلمية"""
    __tablename__ = "model_metrics"

    id            = Column(Integer, primary_key=True)
    recorded_at   = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String(20))
    n_samples     = Column(Integer)
    precision     = Column(Float)
    recall        = Column(Float)
    f1_macro      = Column(Float)
    auc_macro     = Column(Float)
    mcc           = Column(Float)
    cv_folds      = Column(Integer)
    notes         = Column(Text)


# ─── Init ─────────────────────────────────────────────────────────────────

def init_db():
    """
    إنشاء كل الجداول + تفعيل TimescaleDB hypertable
    يُستدعى مرة واحدة عند أول تشغيل
    """
    Base.metadata.create_all(bind=engine)

    with engine.connect() as conn:
        try:
            conn.execute(
                "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"
            )
            conn.execute("""
                SELECT create_hypertable('rx_measurements', 'measured_at',
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 month');
            """)
            # Continuous Aggregate للتسريع (متوسط شهري تلقائي)
            conn.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS monthly_rx_avg
                WITH (timescaledb.continuous) AS
                SELECT subscriber_id,
                       time_bucket('1 month', measured_at) AS month,
                       AVG(rx_power_dbm)   AS avg_rx,
                       MIN(rx_power_dbm)   AS min_rx,
                       MAX(rx_power_dbm)   AS max_rx,
                       COUNT(*)            AS n_readings
                FROM rx_measurements
                GROUP BY subscriber_id, month;
            """)
            conn.commit()
            print("✅ TimescaleDB hypertable created")
        except Exception as e:
            print(f"⚠️  TimescaleDB init note: {e}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

"""
FiberPulse Production — FastAPI Backend
كل endpoint يمثل طبقة مفقودة في الكود الأصلي

تشغيل:
    uvicorn main:app --reload --port 8000

الوثائق:
    http://localhost:8000/docs
"""
import asyncio
from datetime import datetime
from io import StringIO
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import (Depends, FastAPI, File, HTTPException,
                     UploadFile, WebSocket, WebSocketDisconnect)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from database import (Alert, FaultHistory, ModelMetrics, Prediction,
                      RxMeasurement, Subscriber, get_db, init_db)
from model import FAULT_TYPES, SEQ_LEN, FiberPulseMLEngine

# ─── App ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FiberPulse Production API",
    description=(
        "FTTH/GPON Predictive Maintenance System\n"
        "Algérie Télécom — LSTM + SHAP XAI"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # في الإنتاج: ضع domain AT فقط
    allow_methods=["*"],
    allow_headers=["*"],
)

ml = FiberPulseMLEngine()
active_ws: list[WebSocket] = []

MAX_FILE_MB  = 5
MAX_FILE_B   = MAX_FILE_MB * 1024 * 1024


# ─── Pydantic Schemas ─────────────────────────────────────────────────────

class RxHistoryIn(BaseModel):
    rx_history: List[float] = Field(..., min_items=8, max_items=8,
                                    description="8 أشهر Rx Power بـ dBm")

    @validator('rx_history', each_item=True)
    def validate_rx(cls, v):
        if not (-60.0 <= v <= 10.0):
            raise ValueError(f"قيمة Rx خارج النطاق [-60, 10]: {v}")
        return v


class FaultRecordIn(BaseModel):
    subscriber_external_id: str
    fault_date:  datetime
    fault_type:  str
    actual_distance_m: Optional[float] = None
    resolution_hours:  Optional[float] = None
    technician_notes:  Optional[str]   = None


class AcknowledgeIn(BaseModel):
    acknowledged_by: str = Field(..., min_length=2, max_length=50)


# ─── Startup ──────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    init_db()
    try:
        ml.load()
        print("✅ ML Engine ready")
    except FileNotFoundError:
        print("⚠️  No model found — POST /api/v1/model/train to train first")


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 1 — Data Ingestion
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/subscribers/import",
          summary="استيراد مشتركين من CSV",
          tags=["Data Import"])
async def import_subscribers(
    file: UploadFile = File(...),
    db:   Session    = Depends(get_db)
):
    """
    CSV مطلوب:
        subscriber_id, pon_group_id, wilaya_code, wilaya_name,
        fiber_length_km, splitter_ratio, olt_class,
        olt_lat, olt_lng, onu_lat, onu_lng
    """
    _validate_file(file, ext='.csv')
    content = await file.read()
    df = _read_csv_safe(content)

    required = ['subscriber_id', 'pon_group_id']
    _check_columns(df, required)

    imported = skipped = 0
    for _, row in df.iterrows():
        ext_id = str(row['subscriber_id']).strip()
        if not ext_id:
            skipped += 1
            continue

        exists = db.query(Subscriber).filter(
            Subscriber.external_id == ext_id
        ).first()
        if exists:
            skipped += 1
            continue

        sub = Subscriber(
            external_id     = ext_id,
            pon_group_id    = int(row['pon_group_id']),
            wilaya_code     = int(row.get('wilaya_code', 16)),
            wilaya_name     = str(row.get('wilaya_name', 'الجزائر'))[:60],
            fiber_length_km = float(row.get('fiber_length_km', 3.0)),
            splitter_ratio  = str(row.get('splitter_ratio', '1:32'))[:10],
            olt_class       = str(row.get('olt_class', 'B+'))[:5],
            helix_factor    = float(row.get('helix_factor', 1.04)),
            olt_lat         = float(row['olt_lat']) if 'olt_lat' in row else None,
            olt_lng         = float(row['olt_lng']) if 'olt_lng' in row else None,
            onu_lat         = float(row['onu_lat']) if 'onu_lat' in row else None,
            onu_lng         = float(row['onu_lng']) if 'onu_lng' in row else None,
        )
        db.add(sub)
        imported += 1

    db.commit()
    return {"imported": imported, "skipped": skipped,
            "total_in_file": len(df)}


@app.post("/api/v1/measurements/import",
          summary="استيراد قراءات Rx Power التاريخية",
          tags=["Data Import"])
async def import_measurements(
    file: UploadFile = File(...),
    db:   Session    = Depends(get_db)
):
    """
    CSV مطلوب:
        subscriber_id, measured_at (ISO 8601), rx_power_dbm
    اختياري:
        tx_power_dbm, ber, wavelength_nm, source
    """
    _validate_file(file, ext='.csv')
    content = await file.read()
    df = _read_csv_safe(content)
    _check_columns(df, ['subscriber_id', 'measured_at', 'rx_power_dbm'])

    imported = errors = 0
    for _, row in df.iterrows():
        try:
            ext_id = str(row['subscriber_id']).strip()
            sub = db.query(Subscriber).filter(
                Subscriber.external_id == ext_id
            ).first()
            if not sub:
                errors += 1
                continue

            rx = float(row['rx_power_dbm'])
            if not (-60 <= rx <= 10):
                errors += 1
                continue

            m = RxMeasurement(
                subscriber_id = sub.id,
                measured_at   = pd.to_datetime(row['measured_at']),
                rx_power_dbm  = rx,
                tx_power_dbm  = float(row['tx_power_dbm'])
                                if 'tx_power_dbm' in row
                                and pd.notna(row['tx_power_dbm']) else None,
                wavelength_nm = int(row.get('wavelength_nm', 1490)),
                source        = str(row.get('source', 'olt'))[:20],
            )
            db.add(m)
            imported += 1

        except Exception:
            errors += 1

    db.commit()
    return {"imported": imported, "errors": errors}


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 2 — ML Training
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/model/train",
          summary="تدريب النموذج على البيانات الحقيقية",
          tags=["ML Model"])
async def train_model(
    epochs: int  = 120,
    use_cv: bool = True,
    db: Session  = Depends(get_db)
):
    """
    يبني training DataFrame من قاعدة البيانات
    ثم يدرّب LSTM بـ 5-Fold CV (use_cv=True)

    يستغرق 5-30 دقيقة حسب حجم البيانات.
    في الإنتاج: حوّله لـ Background Task (Celery/ARQ)
    """
    subscribers = (db.query(Subscriber)
                   .filter(Subscriber.is_active == True).all())

    if len(subscribers) < 30:
        raise HTTPException(400,
            f"تحتاج ≥ 30 مشترك. حالياً: {len(subscribers)}")

    rows = []
    for sub in subscribers:
        # آخر 8 قراءات شهرية
        meas = (db.query(RxMeasurement)
                .filter(RxMeasurement.subscriber_id == sub.id)
                .order_by(RxMeasurement.measured_at.desc())
                .limit(SEQ_LEN).all())

        if len(meas) < SEQ_LEN:
            continue

        rx_vals = [m.rx_power_dbm for m in reversed(meas)]

        # label حقيقي: آخر عطل موثّق
        fault_rec = (db.query(FaultHistory)
                     .filter(FaultHistory.subscriber_id == sub.id)
                     .order_by(FaultHistory.fault_date.desc())
                     .first())

        row = {f'month_{i+1}': v for i, v in enumerate(rx_vals)}
        row['fault_type']      = (fault_rec.fault_type
                                  if fault_rec else 'STABLE')
        row['actual_failure']  = 1 if fault_rec else 0
        rows.append(row)

    if len(rows) < 20:
        raise HTTPException(422,
            f"بيانات غير كافية بعد التصفية: {len(rows)} عينة")

    df = pd.DataFrame(rows)
    result = ml.train(df, epochs=epochs, use_cv=use_cv)

    # حفظ مقاييس الأداء في DB
    if result.get('cv_results'):
        cv = result['cv_results']
        metric = ModelMetrics(
            model_version = ml.version,
            n_samples     = result['n_samples'],
            f1_macro      = cv.get('f1_macro', {}).get('mean'),
            mcc           = cv.get('mcc', {}).get('mean'),
            cv_folds      = 5,
        )
        db.add(metric)
        db.commit()

    return {
        "status"   : "trained",
        "n_samples": result['n_samples'],
        "cv"       : result.get('cv_results')
    }


@app.get("/api/v1/model/metrics",
         summary="سجل أداء النموذج",
         tags=["ML Model"])
async def get_model_metrics(db: Session = Depends(get_db)):
    metrics = (db.query(ModelMetrics)
               .order_by(ModelMetrics.recorded_at.desc())
               .limit(10).all())
    return [{
        "version"   : m.model_version,
        "date"      : m.recorded_at.isoformat(),
        "n_samples" : m.n_samples,
        "f1_macro"  : m.f1_macro,
        "mcc"       : m.mcc,
    } for m in metrics]


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 3 — Prediction + XAI
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/predict/{subscriber_id}",
          summary="تنبؤ لمشترك واحد مع تفسير XAI",
          tags=["Prediction"])
async def predict_single(
    subscriber_id: str,
    db: Session = Depends(get_db)
):
    sub = _get_subscriber_or_404(subscriber_id, db)

    meas = (db.query(RxMeasurement)
            .filter(RxMeasurement.subscriber_id == sub.id)
            .order_by(RxMeasurement.measured_at.desc())
            .limit(SEQ_LEN).all())

    if len(meas) < SEQ_LEN:
        raise HTTPException(422,
            f"قراءات غير كافية: {len(meas)}/{SEQ_LEN} أشهر")

    rx_history = [m.rx_power_dbm for m in reversed(meas)]

    try:
        pred = ml.predict_single(rx_history)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    # حفظ التنبؤ
    shap_data = pred.get('xai', {})
    p_rec = Prediction(
        subscriber_id   = sub.id,
        risk_category   = pred['risk_category'],
        risk_score      = pred['risk_score'],
        fault_type      = pred['fault_type'],
        confidence_pct  = pred['fault_confidence_pct'],
        shap_values     = shap_data,
        model_version   = pred['model_version'],
    )
    db.add(p_rec)

    # تنبيه حرج → WebSocket + DB
    if pred['risk_category'] == 'حرج':
        reason = shap_data.get('top_reason', '')
        a = Alert(
            subscriber_id = sub.id,
            alert_type    = 'CRITICAL',
            message       = (f"🚨 مشترك {subscriber_id} — "
                             f"{pred['fault_type']} — {reason}")
        )
        db.add(a)
        await _broadcast({
            "type"         : "CRITICAL_ALERT",
            "subscriber_id": subscriber_id,
            "pon_group"    : sub.pon_group_id,
            "fault_type"   : pred['fault_type'],
            "reason"       : reason,
            "timestamp"    : datetime.utcnow().isoformat()
        })

    db.commit()

    return {
        "subscriber_id": subscriber_id,
        "pon_group"    : sub.pon_group_id,
        "wilaya"       : sub.wilaya_name,
        "last_rx_dbm"  : rx_history[-1],
        "prediction"   : pred
    }


@app.get("/api/v1/predict/batch/all",
         summary="تنبؤ شامل لجميع المشتركين النشطين",
         tags=["Prediction"])
async def predict_all(db: Session = Depends(get_db)):
    subscribers = (db.query(Subscriber)
                   .filter(Subscriber.is_active == True).all())

    buckets = {'حرج': [], 'خطر': [], 'مراقبة': [], 'مستقر': []}
    errors  = []

    for sub in subscribers:
        meas = (db.query(RxMeasurement)
                .filter(RxMeasurement.subscriber_id == sub.id)
                .order_by(RxMeasurement.measured_at.desc())
                .limit(SEQ_LEN).all())

        if len(meas) < SEQ_LEN:
            continue

        try:
            pred = ml.predict_single(
                [m.rx_power_dbm for m in reversed(meas)]
            )
            entry = {
                "id"         : sub.external_id,
                "pon_group"  : sub.pon_group_id,
                "wilaya"     : sub.wilaya_name,
                "risk_score" : pred['risk_score'],
                "fault_type" : pred['fault_type'],
                "location"   : pred['location'],
                "top_reason" : pred.get('xai', {}).get('top_reason', '—'),
            }
            cat = pred['risk_category']
            buckets.setdefault(cat, []).append(entry)
        except Exception as e:
            errors.append({"id": sub.external_id, "error": str(e)})

    total = sum(len(v) for v in buckets.values())
    return {
        "total_analyzed": total,
        "summary"       : {k: len(v) for k, v in buckets.items()},
        "results"       : buckets,
        "errors"        : errors
    }


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 4 — PON Topology Analysis
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/pon/{pon_group_id}",
         summary="تحليل مجموعة PON كاملة",
         tags=["Analytics"])
async def analyze_pon(pon_group_id: int, db: Session = Depends(get_db)):
    subs = (db.query(Subscriber)
            .filter(Subscriber.pon_group_id == pon_group_id,
                    Subscriber.is_active == True).all())

    if not subs:
        raise HTTPException(404, f"PON Group {pon_group_id} غير موجود")

    critical_count = 0
    details = []

    for sub in subs:
        last_pred = (db.query(Prediction)
                     .filter(Prediction.subscriber_id == sub.id)
                     .order_by(Prediction.predicted_at.desc())
                     .first())

        cat = last_pred.risk_category if last_pred else 'غير محلل'
        if cat in ('حرج', 'خطر'):
            critical_count += 1

        details.append({
            "id"        : sub.external_id,
            "risk"      : cat,
            "fault_type": last_pred.fault_type if last_pred else None,
            "score"     : last_pred.risk_score if last_pred else None,
        })

    splitter_suspected = critical_count >= 3

    return {
        "pon_group_id"        : pon_group_id,
        "total_subscribers"   : len(subs),
        "critical_count"      : critical_count,
        "splitter_fault_suspected": splitter_suspected,
        "recommendation"      : (
            "🚨 إرسال فرقة للكابينة — عطل Splitter محتمل"
            if splitter_suspected else
            "✅ الشبكة الرئيسية سليمة"
        ),
        "subscribers"         : details
    }


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 5 — Ground Truth & Model Feedback Loop
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/faults/record",
          summary="تسجيل عطل حقيقي (Ground Truth)",
          tags=["Feedback Loop"])
async def record_fault(body: FaultRecordIn, db: Session = Depends(get_db)):
    """
    يُستدعى بعد إصلاح عطل حقيقي.
    يُحدّث was_correct في Prediction المقابل.
    → يبني تدريجياً dataset validation حقيقي.
    """
    sub = _get_subscriber_or_404(body.subscriber_external_id, db)

    fault = FaultHistory(
        subscriber_id       = sub.id,
        fault_date          = body.fault_date,
        fault_type          = body.fault_type,
        actual_distance_m   = body.actual_distance_m,
        resolution_hours    = body.resolution_hours,
        technician_notes    = body.technician_notes,
    )
    db.add(fault)

    # ربط بآخر تنبؤ وتحديث was_correct
    last_pred = (db.query(Prediction)
                 .filter(Prediction.subscriber_id == sub.id)
                 .order_by(Prediction.predicted_at.desc())
                 .first())

    if last_pred:
        last_pred.was_correct = (
            last_pred.risk_category in ('حرج', 'خطر') and
            last_pred.fault_type == body.fault_type
        )
        last_pred.verified_at = datetime.utcnow()

    db.commit()
    return {"recorded": True, "prediction_updated": last_pred is not None}


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 6 — ROI & SLA Analytics
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/analytics/roi",
         summary="العائد الاقتصادي الحقيقي",
         tags=["Analytics"])
async def roi(db: Session = Depends(get_db)):
    tp = (db.query(Prediction)
          .filter(Prediction.was_correct == True,
                  Prediction.risk_category.in_(['حرج', 'خطر']))
          .count())

    fp = (db.query(Prediction)
          .filter(Prediction.was_correct == False,
                  Prediction.risk_category.in_(['حرج', 'خطر']))
          .count())

    TRUCK   = 12_000
    PENALTY = 8_000
    CHURN   = 25_000 * 0.15

    prevented   = round(tp * 0.65)
    gross       = prevented * (TRUCK + PENALTY + CHURN)
    waste       = fp * TRUCK * 0.4
    net         = max(0, gross - waste)
    monthly_200 = round(net * (200 / max(1, tp + fp)))

    return {
        "tp": tp, "fp": fp,
        "prevented_faults"    : prevented,
        "net_savings_dzd"     : net,
        "monthly_projected_dzd": monthly_200,
        "annual_projected_dzd": monthly_200 * 12,
    }


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 7 — NMS Export & Alerts
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/export/nms-json",
         summary="تصدير تنبيهات بتنسيق NMS/OSS",
         tags=["Export"])
async def nms_export(db: Session = Depends(get_db)):
    """تنسيق قابل للاستيراد في نظام AT NMS/OSS"""
    preds = (db.query(Prediction)
             .filter(Prediction.risk_category.in_(['حرج', 'خطر']))
             .order_by(Prediction.predicted_at.desc())
             .limit(200).all())

    alerts_out = []
    for p in preds:
        sub = db.query(Subscriber).get(p.subscriber_id)
        if not sub:
            continue
        alerts_out.append({
            "subscriberId"   : sub.external_id,
            "ponGroupId"     : sub.pon_group_id,
            "wilaya"         : sub.wilaya_name,
            "riskCategory"   : p.risk_category,
            "riskScore"      : p.risk_score,
            "faultType"      : p.fault_type,
            "confidence"     : p.confidence_pct,
            "xaiExplanation" : (p.shap_values or {}).get('top_reason', ''),
            "predictedAt"    : p.predicted_at.isoformat(),
        })

    return {
        "generatedAt": datetime.utcnow().isoformat(),
        "system"     : "FiberPulse",
        "version"    : "1.0.0",
        "total"      : len(alerts_out),
        "alerts"     : alerts_out
    }


@app.get("/api/v1/alerts",
         summary="التنبيهات النشطة غير المعالجة",
         tags=["Alerts"])
async def active_alerts(db: Session = Depends(get_db)):
    alerts = (db.query(Alert)
              .filter(Alert.is_acknowledged == False)
              .order_by(Alert.created_at.desc())
              .limit(100).all())

    return [{
        "id"           : a.id,
        "subscriber_id": (db.query(Subscriber).get(a.subscriber_id)
                          .external_id if a.subscriber_id else None),
        "type"         : a.alert_type,
        "message"      : a.message,
        "created_at"   : a.created_at.isoformat(),
    } for a in alerts]


@app.patch("/api/v1/alerts/{alert_id}/acknowledge",
           summary="تأكيد معالجة تنبيه",
           tags=["Alerts"])
async def ack_alert(alert_id: int, body: AcknowledgeIn,
                    db: Session = Depends(get_db)):
    a = db.query(Alert).get(alert_id)
    if not a:
        raise HTTPException(404, "التنبيه غير موجود")
    a.is_acknowledged  = True
    a.acknowledged_by  = body.acknowledged_by
    a.acknowledged_at  = datetime.utcnow()
    db.commit()
    return {"acknowledged": True}


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 8 — WebSocket Real-time Alerts
# ═══════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/alerts")
async def ws_alerts(ws: WebSocket):
    """
    تدفق مباشر للتنبيهات الحرجة للـ Frontend
    الاستخدام في FiberPulse v2.5:
        const ws = new WebSocket('ws://localhost:8000/ws/alerts');
        ws.onmessage = (e) => { const alert = JSON.parse(e.data); ... };
    """
    await ws.accept()
    active_ws.append(ws)
    try:
        while True:
            await asyncio.sleep(30)
            await ws.send_json({"type": "ping",
                                "ts": datetime.utcnow().isoformat()})
    except WebSocketDisconnect:
        pass
    finally:
        if ws in active_ws:
            active_ws.remove(ws)


# ─── Helpers ──────────────────────────────────────────────────────────────

def _validate_file(file: UploadFile, ext: str):
    if not file.filename.lower().endswith(ext):
        raise HTTPException(400, f"نوع الملف يجب أن يكون {ext}")
    if file.size and file.size > MAX_FILE_B:
        raise HTTPException(400,
            f"الملف أكبر من {MAX_FILE_MB}MB")


def _read_csv_safe(content: bytes) -> pd.DataFrame:
    try:
        text = content.decode('utf-8-sig')  # يتعامل مع BOM
        sep  = ';' if text.split('\n', 1)[0].count(';') > text.split('\n', 1)[0].count(',') else ','
        return pd.read_csv(StringIO(text), sep=sep)
    except Exception as e:
        raise HTTPException(422, f"خطأ في قراءة CSV: {e}")


def _check_columns(df: pd.DataFrame, required: list):
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(422, f"أعمدة ناقصة: {missing}")


def _get_subscriber_or_404(ext_id: str, db: Session) -> Subscriber:
    sub = db.query(Subscriber).filter(
        Subscriber.external_id == ext_id
    ).first()
    if not sub:
        raise HTTPException(404, f"المشترك '{ext_id}' غير موجود")
    return sub


async def _broadcast(payload: dict):
    dead = []
    for ws in active_ws:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        active_ws.remove(ws)

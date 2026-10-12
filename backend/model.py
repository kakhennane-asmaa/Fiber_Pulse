"""
FiberPulse Production — ML Engine
LSTM ثنائي الطبقة + SHAP XAI

الابتكار الحقيقي:
  ← LSTM يتعلم من أنماط Rx Power الزمنية (لا rules يدوية)
  ← SHAP يُفسّر كل تنبؤ: "لماذا هذا المشترك في خطر؟"
  ← 5-Fold CV يعطي validation حقيقي للورقة العلمية
"""
import numpy as np
import pandas as pd
import shap
import joblib
from pathlib import Path
from typing import Optional

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (LSTM, Dense, Dropout,
                                     BatchNormalization, Bidirectional)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, f1_score,
                              matthews_corrcoef)

# ─── Constants ────────────────────────────────────────────────────────────

MODEL_DIR   = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FAULT_TYPES = ['STABLE', 'CONTAMINATION', 'SPLICE', 'MACROBEND', 'BREAK']
N_CLASSES   = len(FAULT_TYPES)
SEQ_LEN     = 8      # 8 أشهر تاريخ
N_FEATURES  = 5      # انظر extract_features()
CRITICAL_DBM = -28.0


# ─── Feature Engineering ──────────────────────────────────────────────────

def extract_features(rx_history: list) -> np.ndarray:
    """
    تحويل 8 قراءات Rx Power إلى feature matrix غنية

    Input : [rx_1, rx_2, ..., rx_8]  (dBm)
    Output: (8, 5) array

    Feature 0 — rx_power_dbm      : القيمة المطلقة
    Feature 1 — delta_rx          : التغيير الشهري (ΔRx)
    Feature 2 — margin_to_critical: الهامش عن -28 dBm
    Feature 3 — rolling_mean_3m   : متوسط متحرك 3 أشهر
    Feature 4 — cusum             : CUSUM تراكمي للكشف عن التغيير المفاجئ
    """
    rx = np.array(rx_history, dtype=np.float32)

    delta   = np.diff(rx, prepend=rx[0])
    margin  = rx - CRITICAL_DBM

    # متوسط متحرك 3 أشهر
    roll3 = np.convolve(rx, np.ones(3)/3, mode='same')

    # CUSUM: اكتشاف نقاط التغيير
    baseline = rx[:3].mean()
    drift    = 0.15
    cusum    = np.zeros(SEQ_LEN)
    for i in range(1, SEQ_LEN):
        cusum[i] = max(0.0, cusum[i-1] + (baseline - rx[i] - drift))

    return np.stack([rx, delta, margin, roll3, cusum], axis=-1)  # (8, 5)


def prepare_dataset(df: pd.DataFrame, scaler: Optional[StandardScaler] = None,
                    fit_scaler: bool = True):
    """
    تحضير dataset كامل للتدريب أو الاختبار

    df columns مطلوبة:
        month_1 .. month_8  : قراءات Rx Power (dBm)
        fault_type          : STABLE/CONTAMINATION/SPLICE/MACROBEND/BREAK
        actual_failure      : 0 أو 1
    """
    X_list, y_fault, y_failure = [], [], []

    for _, row in df.iterrows():
        rx = [row.get(f'month_{i}') for i in range(1, SEQ_LEN + 1)]
        if any(pd.isna(v) for v in rx):
            continue
        if not all(-60 <= v <= 10 for v in rx):
            continue

        features = extract_features([float(v) for v in rx])
        X_list.append(features)

        fault = str(row.get('fault_type', 'STABLE')).strip().upper()
        y_fault.append(fault if fault in FAULT_TYPES else 'STABLE')
        y_failure.append(int(bool(row.get('actual_failure', 0))))

    if not X_list:
        raise ValueError("لا توجد عينات صالحة في DataFrame")

    X = np.array(X_list, dtype=np.float32)          # (N, 8, 5)
    y_f = np.array([FAULT_TYPES.index(f) for f in y_fault])
    y_b = np.array(y_failure, dtype=np.float32)

    # Scaling على features المستقلة
    N, T, F = X.shape
    X_flat = X.reshape(-1, F)

    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        X_scaled = scaler.fit_transform(X_flat).reshape(N, T, F)
    else:
        X_scaled = scaler.transform(X_flat).reshape(N, T, F)

    return X_scaled, y_f, y_b, scaler


# ─── Model Architecture ───────────────────────────────────────────────────

def build_lstm_model() -> Sequential:
    """
    Bidirectional LSTM ثنائي الطبقة

    لماذا Bidirectional؟
    ← يقرأ التسلسل الزمني في الاتجاهين
    ← يستخلص أنماط: "ارتفع ثم انهار" وكذلك "كان مستقراً ثم انحرف"

    Architecture:
        BiLSTM(64) → Dropout(0.3)
        BiLSTM(32) → Dropout(0.2) → BatchNorm
        Dense(64, relu) → Dense(N_CLASSES, softmax)
    """
    model = Sequential([
        Bidirectional(
            LSTM(64, return_sequences=True,
                 kernel_regularizer=l2(1e-3)),
            input_shape=(SEQ_LEN, N_FEATURES)
        ),
        Dropout(0.3),

        Bidirectional(
            LSTM(32, return_sequences=False,
                 kernel_regularizer=l2(1e-3))
        ),
        Dropout(0.2),
        BatchNormalization(),

        Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.1),
        Dense(N_CLASSES, activation='softmax', name='fault_output')
    ])

    model.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ─── Main Engine ──────────────────────────────────────────────────────────

class FiberPulseMLEngine:
    """
    المحرك الرئيسي: تدريب + تنبؤ + تفسير XAI

    الاستخدام:
        engine = FiberPulseMLEngine()
        engine.train(df)          # تدريب
        result = engine.predict_single(rx_history)  # تنبؤ مع SHAP
    """

    def __init__(self):
        self.model     : Optional[Sequential]      = None
        self.scaler    : Optional[StandardScaler]  = None
        self.explainer                             = None
        self.is_trained: bool                      = False
        self.version   : str                       = "lstm_v1"

    # ─── Training ─────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame,
              epochs: int = 120,
              use_cv: bool = True) -> dict:
        """
        تدريب مع 5-Fold Stratified Cross Validation
        use_cv=True  → للورقة العلمية (نتائج موثوقة)
        use_cv=False → للتطوير السريع
        """
        if len(df) < 30:
            raise ValueError(
                f"Dataset صغير جداً ({len(df)} عينة). "
                f"تحتاج ≥ 30 مشترك لتدريب موثوق."
            )

        X, y_f, y_b, self.scaler = prepare_dataset(df, fit_scaler=True)
        print(f"📦 Dataset: {len(X)} عينة | "
              f"{dict(zip(*np.unique(y_f, return_counts=True)))}")

        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True,
                          monitor='val_loss'),
            ModelCheckpoint(str(MODEL_DIR / 'best_model.keras'),
                            save_best_only=True, verbose=0)
        ]

        if use_cv:
            cv_results = self._cross_validate(X, y_f, callbacks, epochs)
        else:
            cv_results = None

        # النموذج النهائي على كل البيانات
        print("\n🏋️  Training final model on full dataset...")
        self.model = build_lstm_model()
        self.model.fit(
            X, y_f,
            epochs=max(30, epochs // 2),
            batch_size=min(32, max(8, len(X) // 8)),
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1
        )

        # بناء SHAP explainer على عينة ممثلة
        background_n = min(100, len(X))
        self._build_explainer(X[:background_n])

        self.is_trained = True
        self.save()

        return {"cv_results": cv_results, "n_samples": len(X)}

    def _cross_validate(self, X, y, callbacks, epochs) -> dict:
        """5-Fold CV — الأرقام التي تذهب في الورقة العلمية"""
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = {'accuracy': [], 'f1_macro': [], 'mcc': []}

        print("🔁 5-Fold Stratified Cross Validation\n")

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            m = build_lstm_model()
            m.fit(X_tr, y_tr,
                  validation_data=(X_val, y_val),
                  epochs=epochs,
                  batch_size=min(32, max(8, len(X_tr) // 8)),
                  callbacks=callbacks,
                  verbose=0)

            y_pred = np.argmax(m.predict(X_val, verbose=0), axis=1)
            acc  = float(np.mean(y_pred == y_val))
            f1   = f1_score(y_val, y_pred, average='macro', zero_division=0)
            mcc  = matthews_corrcoef(y_val, y_pred)

            scores['accuracy'].append(acc)
            scores['f1_macro'].append(f1)
            scores['mcc'].append(mcc)

            print(f"  Fold {fold+1}/5 — "
                  f"Acc: {acc:.3f} | F1: {f1:.3f} | MCC: {mcc:.3f}")

        print(f"\n📊 CV Summary:")
        summary = {}
        for k, v in scores.items():
            summary[k] = {
                'mean': round(float(np.mean(v)), 4),
                'std':  round(float(np.std(v)),  4)
            }
            print(f"  {k}: {summary[k]['mean']:.3f} ± {summary[k]['std']:.3f}")

        return summary

    # ─── SHAP XAI ─────────────────────────────────────────────────────────

    def _build_explainer(self, background: np.ndarray):
        """
        GradientExplainer: يعمل مباشرة مع LSTM Keras
        background: عينة ممثلة (50-100) تمثل التوزيع الطبيعي
        """
        print("🔍 Building SHAP GradientExplainer...")
        self.explainer = shap.GradientExplainer(self.model, background)
        print("✅ SHAP ready")

    def explain(self, rx_history: list) -> dict:
        """
        لكل تنبؤ: لماذا هذا القرار؟

        Returns:
            peak_month     : الشهر الأكثر تأثيراً في القرار
            top_reason     : السبب بالعربية
            monthly_impact : أهمية كل شهر (للرسم البياني)
            feature_impact : أهمية كل feature
        """
        if not self.explainer:
            return {"available": False}

        features = extract_features(rx_history)
        X_scaled = self.scaler.transform(
            features.reshape(-1, N_FEATURES)
        ).reshape(1, SEQ_LEN, N_FEATURES)

        # shap_vals: list طول N_CLASSES، كل عنصر (1, SEQ_LEN, N_FEATURES)
        shap_vals = self.explainer.shap_values(X_scaled)

        # تجميع الأهمية على كل classes وfeatures
        monthly_imp = np.zeros(SEQ_LEN)
        feat_imp    = np.zeros(N_FEATURES)

        for c in range(N_CLASSES):
            s = np.abs(shap_vals[c][0])     # (SEQ_LEN, N_FEATURES)
            monthly_imp += s.sum(axis=-1)
            feat_imp    += s.sum(axis=0)

        # أهم شهر
        peak_idx  = int(np.argmax(monthly_imp))
        peak_pct  = round(float(monthly_imp[peak_idx] /
                                (monthly_imp.sum() + 1e-9) * 100), 1)

        # السبب بالعربية
        delta_at_peak = rx_history[peak_idx] - rx_history[max(0, peak_idx - 1)]
        margin_now    = rx_history[-1] - CRITICAL_DBM

        if feat_imp[4] == feat_imp.max():           # CUSUM dominant
            reason = (f"تغيير مفاجئ كُشف في الشهر {peak_idx+1} "
                      f"(ΔRx = {delta_at_peak:+.2f} dBm)")
        elif feat_imp[2] == feat_imp.max():         # margin dominant
            reason = (f"الاقتراب الخطير من العتبة الحرجة "
                      f"(هامش متبقٍ = {margin_now:.1f} dBm)")
        elif feat_imp[1] == feat_imp.max():         # delta dominant
            reason = (f"تسارع التدهور في الشهر {peak_idx+1} "
                      f"(ΔRx = {delta_at_peak:+.2f} dBm/شهر)")
        else:                                        # rx dominant
            reason = (f"انخفاض تراكمي في قوة الإشارة "
                      f"(آخر قراءة: {rx_history[-1]:.1f} dBm)")

        feature_names = ['Rx Power', 'ΔRx', 'Margin to -28dBm',
                         'Rolling Mean 3m', 'CUSUM']

        return {
            "available"     : True,
            "peak_month"    : peak_idx + 1,
            "peak_impact_pct": peak_pct,
            "top_reason"    : reason,
            "monthly_impact": [round(float(v), 4) for v in monthly_imp],
            "feature_impact": dict(zip(feature_names,
                                       [round(float(v), 4) for v in feat_imp]))
        }

    # ─── Inference ────────────────────────────────────────────────────────

    def predict_single(self, rx_history: list) -> dict:
        """تنبؤ لمشترك واحد + XAI كامل"""
        if not self.is_trained:
            raise RuntimeError("النموذج غير مدرَّب — استدعِ train() أولاً")

        if len(rx_history) != SEQ_LEN:
            raise ValueError(
                f"يجب توفير {SEQ_LEN} قراءات شهرية. "
                f"مُعطى: {len(rx_history)}"
            )

        features = extract_features(rx_history)
        X = self.scaler.transform(
            features.reshape(-1, N_FEATURES)
        ).reshape(1, SEQ_LEN, N_FEATURES)

        proba     = self.model.predict(X, verbose=0)[0]   # (N_CLASSES,)
        fault_idx = int(np.argmax(proba))
        fault     = FAULT_TYPES[fault_idx]
        conf      = float(proba[fault_idx])

        # احتمال الفشل = 1 - P(STABLE)
        stable_idx   = FAULT_TYPES.index('STABLE')
        failure_prob = float(1.0 - proba[stable_idx])

        # تصنيف الخطر
        last_rx = rx_history[-1]
        margin  = last_rx - CRITICAL_DBM

        if margin <= 0 or failure_prob > 0.85:
            risk_cat = 'حرج'
        elif failure_prob > 0.65:
            risk_cat = 'خطر'
        elif failure_prob > 0.35:
            risk_cat = 'مراقبة'
        else:
            risk_cat = 'مستقر'

        # الموقع المتوقع للعطل
        location_map = {
            'MACROBEND'    : ('في المنزل عند ONT',          20),
            'CONTAMINATION': ('علبة الطابق / FAT',         150),
            'SPLICE'       : ('كابل التوزيع (BPEO)',        350),
            'BREAK'        : ('الكابل الرئيسي (Feeder)',   500),
            'STABLE'       : ('لا عطل متوقع',                0),
        }
        location_desc, uncertainty_m = location_map.get(fault, ('غير محدد', 999))

        xai = self.explain(rx_history)

        return {
            "fault_type"           : fault,
            "fault_probabilities"  : dict(zip(FAULT_TYPES, proba.tolist())),
            "fault_confidence_pct" : round(conf * 100, 1),
            "failure_probability_3m": round(failure_prob, 3),
            "risk_category"        : risk_cat,
            "risk_score"           : round(failure_prob * 100, 1),
            "location"             : location_desc,
            "uncertainty_meters"   : uncertainty_m,
            "xai"                  : xai,
            "model_version"        : self.version
        }

    def predict_batch(self, records: list) -> list:
        """تنبؤ دُفعي — كل عنصر: {'id': X, 'rx_history': [...]}"""
        return [
            {**self.predict_single(r['rx_history']), 'subscriber_id': r['id']}
            if len(r.get('rx_history', [])) == SEQ_LEN
            else {'subscriber_id': r.get('id'), 'error': 'بيانات ناقصة'}
            for r in records
        ]

    # ─── Evaluation — للورقة العلمية ──────────────────────────────────────

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        """
        تقييم شامل على test set مستقل
        المقاييس التي تذهب في جدول الورقة:
            Precision, Recall, F1-macro, AUC-macro, MCC
        """
        X, y_true, _, _ = prepare_dataset(
            test_df, scaler=self.scaler, fit_scaler=False
        )
        y_pred = np.argmax(self.model.predict(X, verbose=0), axis=1)
        y_prob = self.model.predict(X, verbose=0)

        report = classification_report(
            y_true, y_pred, target_names=FAULT_TYPES,
            output_dict=True, zero_division=0
        )

        try:
            auc = roc_auc_score(y_true, y_prob,
                                multi_class='ovr', average='macro')
        except ValueError:
            auc = None

        mcc = matthews_corrcoef(y_true, y_pred)

        return {
            "n_test_samples"      : int(len(X)),
            "classification_report": report,
            "auc_macro"           : round(float(auc), 4) if auc else None,
            "mcc"                 : round(float(mcc), 4),
            "confusion_matrix"    : confusion_matrix(y_true, y_pred).tolist()
        }

    # ─── Persistence ──────────────────────────────────────────────────────

    def save(self):
        self.model.save(str(MODEL_DIR / 'lstm_model.keras'))
        joblib.dump(self.scaler, str(MODEL_DIR / 'scaler.pkl'))
        print(f"✅ Model saved → {MODEL_DIR}")

    def load(self):
        model_f  = MODEL_DIR / 'lstm_model.keras'
        scaler_f = MODEL_DIR / 'scaler.pkl'

        if not model_f.exists():
            raise FileNotFoundError(
                "النموذج غير موجود — درّب أولاً عبر POST /api/v1/model/train"
            )

        self.model      = load_model(str(model_f))
        self.scaler     = joblib.load(str(scaler_f))
        self.is_trained = True

        # إعادة بناء explainer على بيانات خلفية وهمية إذا لم يكن موجوداً
        dummy = np.zeros((10, SEQ_LEN, N_FEATURES), dtype=np.float32)
        self.explainer = shap.GradientExplainer(self.model, dummy)
        print("✅ Model loaded")

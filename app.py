import os

# ── Paksa Transformers pakai PyTorch only, skip TensorFlow/JAX ────────────────
# Harus di-set SEBELUM import transformers apapun
os.environ["USE_TF"]                 = "0"
os.environ["USE_TORCH"]              = "1"
os.environ["TRANSFORMERS_NO_TF"]     = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"]         = "true"

from flask import Flask, redirect, request, render_template, url_for, session, flash, jsonify
import numpy as np
import pandas as pd
import json
import torch
from functools import wraps
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import warnings
warnings.filterwarnings('ignore')

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "bert-fake-news-detector-secret-key-2025"

# ─── Satu-satunya path yang di-hardcode: flask_config.json ───────────────────
# Semua path lainnya dibaca dari file ini (di-generate otomatis oleh notebook)
FLASK_CONFIG_PATH = os.path.join(os.getcwd(), 'flask_components', 'flask_config.json')


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER: CONFIG & FILE LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def get_flask_config() -> dict:
    """
    Baca flask_config.json yang di-generate oleh Cell 12 notebook.
    Berisi semua path: model, data, results, visualizations.
    """
    try:
        with open(FLASK_CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ flask_config.json tidak ditemukan: {FLASK_CONFIG_PATH}")
        print("   Pastikan sudah menjalankan Cell 12 di Jupyter notebook.")
        return {}
    except Exception as e:
        print(f"❌ Error membaca flask_config.json: {e}")
        return {}


def resolve_path(relative_path: str) -> str:
    """Ubah path relatif (dari flask_config) ke absolute path."""
    if not relative_path:
        return ''
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(os.getcwd(), relative_path)


def load_json_from_config(config_key: str, sub_key: str) -> dict:
    """
    Load file JSON berdasarkan path di flask_config.json.
    config_key : 'results_paths' | 'data_paths' | 'model_paths'
    sub_key    : nama key di dalam config_key, misal 'final_summary'
    """
    cfg  = get_flask_config()
    path = cfg.get(config_key, {}).get(sub_key, '')
    if not path:
        print(f"⚠️  Key '{config_key}.{sub_key}' tidak ada di flask_config.json")
        return {}
    abs_path = resolve_path(path)
    try:
        with open(abs_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ File tidak ditemukan: {abs_path}")
        return {}
    except Exception as e:
        print(f"❌ Error membaca {abs_path}: {e}")
        return {}


def load_csv_from_config(config_key: str, sub_key: str) -> pd.DataFrame:
    """Load file CSV berdasarkan path di flask_config.json."""
    cfg  = get_flask_config()
    path = cfg.get(config_key, {}).get(sub_key, '')
    if not path:
        return pd.DataFrame()
    abs_path = resolve_path(path)
    try:
        return pd.read_csv(abs_path)
    except FileNotFoundError:
        print(f"❌ CSV tidak ditemukan: {abs_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error membaca CSV {abs_path}: {e}")
        return pd.DataFrame()


def get_all_research_data() -> dict:
    """
    Kumpulkan SEMUA data hasil penelitian dari file output notebook.
    Ini adalah single source of truth — tidak ada nilai hardcode sama sekali.

    File yang dibaca (sesuai output Cell 3–12 di notebook):
      results/final_summary.json     → model_info, dataset_info, performance
      results/evaluation_results.json
      results/training_results.json
      results/sample_test_results.csv
      data/dataset_info.json
      data/splitting_info.json
      data/word_analysis.json
      flask_config.json               → visualization filenames, app_settings
    """
    # ── 1. final_summary.json ────────────────────────────────────────────────
    #    Berisi: model_info, dataset_info, performance (semua metrik evaluasi)
    final_summary  = load_json_from_config('results_paths', 'final_summary')
    model_info     = final_summary.get('model_info',   {})
    dataset_info   = final_summary.get('dataset_info', {})
    performance    = final_summary.get('performance',  {})

    # ── 2. evaluation_results.json ───────────────────────────────────────────
    #    Berisi: classification report lengkap per kelas
    eval_results   = load_json_from_config('results_paths', 'evaluation_results')

    # ── 3. training_results.json ─────────────────────────────────────────────
    #    Berisi: loss per epoch, training history
    training_results = load_json_from_config('results_paths', 'training_results')

    # ── 4. dataset_info.json ─────────────────────────────────────────────────
    #    Berisi: info distribusi dataset sebelum balancing
    dataset_info_raw = load_json_from_config('data_paths', 'dataset_info')
    # Merge — final_summary lebih prioritas (post-balancing)
    merged_dataset   = {**dataset_info_raw, **dataset_info}

    # ── 5. splitting_info.json ───────────────────────────────────────────────
    #    Berisi: detail train/test split (size, random_state, dll)
    splitting_info = load_json_from_config('data_paths', 'splitting_info')

    # ── 6. word_analysis.json ────────────────────────────────────────────────
    #    Berisi: top kata FAKE vs REAL, statistik panjang judul
    word_analysis  = load_json_from_config('data_paths', 'word_analysis')

    # ── 7. sample_test_results.csv ───────────────────────────────────────────
    #    Berisi: hasil prediksi test cases dari Cell 11
    sample_df      = load_csv_from_config('results_paths', 'sample_test_results')
    sample_results = sample_df.to_dict('records') if not sample_df.empty else []

    # ── 8. Visualization filenames ───────────────────────────────────────────
    #    Ambil dari flask_config.visualization_paths
    #    Di template dipakai: url_for('static', filename='visualizations/' + viz.xxx)
    cfg           = get_flask_config()
    viz_paths_raw = cfg.get('visualization_paths', {})
    viz_filenames = {
        key: os.path.basename(path)
        for key, path in viz_paths_raw.items()
    }

    # ── 9. App settings dari flask_config ────────────────────────────────────
    app_settings = cfg.get('app_settings', {})

    return {
        # Dari final_summary.json
        'model_info':        model_info,
        'dataset_info':      merged_dataset,
        'performance':       performance,

        # Dari evaluation_results.json
        'eval_results':      eval_results,

        # Dari training_results.json
        'training_results':  training_results,

        # Dari splitting_info.json
        'splitting_info':    splitting_info,

        # Dari word_analysis.json
        'word_analysis':     word_analysis,

        # Dari sample_test_results.csv
        'sample_results':    sample_results,

        # Visualization filenames
        'viz':               viz_filenames,

        # App settings
        'app_settings':      app_settings,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  BERT MODEL — LAZY LOAD
# ═══════════════════════════════════════════════════════════════════════════════

_model     = None
_tokenizer = None


def load_bert_model() -> bool:
    """
    Load IndoBERT model & tokenizer.
    Path diambil dari flask_config.json — bukan hardcode.
    """
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return True

    cfg         = get_flask_config()
    model_paths = cfg.get('model_paths', {})
    model_dir   = resolve_path(model_paths.get('bert_model',     ''))
    tok_dir     = resolve_path(model_paths.get('bert_tokenizer', ''))

    if not model_dir or not os.path.exists(model_dir):
        print(f"❌ Model dir tidak ditemukan: {model_dir}")
        return False
    if not tok_dir or not os.path.exists(tok_dir):
        print(f"❌ Tokenizer dir tidak ditemukan: {tok_dir}")
        return False

    try:
        print(f"⏳ Loading tokenizer  : {tok_dir}")
        _tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)

        print(f"⏳ Loading model      : {model_dir}")
        _model = AutoModelForSequenceClassification.from_pretrained(model_dir, from_tf=False)
        _model.eval()

        print("✅ IndoBERT model & tokenizer berhasil di-load.")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def predict_news(text: str) -> dict | None:
    """
    Inferensi BERT.
    max_length & label_map diambil dari final_summary.json — bukan hardcode.
    """
    if not load_bert_model():
        return None

    try:
        # Ambil max_length dan label_map dari output notebook
        final_summary = load_json_from_config('results_paths', 'final_summary')
        mi            = final_summary.get('model_info', {})
        max_length    = mi.get('max_length', 128)
        label_map     = mi.get('label_map',  {'0': 'FAKE', '1': 'REAL'})

        inputs = _tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = _model(**inputs).logits
            probs  = torch.nn.functional.softmax(logits, dim=-1)[0]

        prob_fake  = float(probs[0].item())
        prob_real  = float(probs[1].item())
        pred_idx   = int(torch.argmax(probs).item())
        label      = label_map.get(str(pred_idx), 'UNKNOWN')
        confidence = prob_real if pred_idx == 1 else prob_fake

        return {
            'label':      label,
            'confidence': round(confidence, 4),
            'prob_fake':  round(prob_fake,  4),
            'prob_real':  round(prob_real,  4),
        }
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  DECORATORS & CONTEXT PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            flash('Silakan login untuk mengakses halaman ini.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


@app.context_processor
def inject_globals():
    """
    Inject variabel global ke semua template Jinja2.
    Semua nilai dari file output notebook — tidak ada yang hardcode.
    """
    final_summary = load_json_from_config('results_paths', 'final_summary')
    cfg           = get_flask_config()
    app_settings  = cfg.get('app_settings', {})

    return {
        # Dari flask_config.app_settings (diisi notebook)
        'app_name':        app_settings.get('title',       'BERT Fake News Detector'),
        'app_description': app_settings.get('description', 'Deteksi berita palsu berbasis IndoBERT'),
        'app_version':     app_settings.get('version',     '1.0.0'),

        # Runtime
        'current_year':  datetime.now().year,
        'is_logged_in':  'username' in session,
        'current_user':  session.get('username', 'Guest'),

        # Dari final_summary.json (selalu tersedia di semua template)
        'performance':   final_summary.get('performance',  {}),
        'model_info':    final_summary.get('model_info',   {}),
        'dataset_info':  final_summary.get('dataset_info', {}),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def homepage():
    """Halaman beranda publik."""
    try:
        data = get_all_research_data()
        return render_template('index.html', **data)
    except Exception as e:
        print(f"❌ Homepage error: {e}")
        flash('Terjadi error saat memuat halaman.', 'danger')
        return render_template('index.html')


@app.route('/viz/<filename>')
def serve_viz(filename):
    """
    Serve file HTML visualisasi dari flask_components/visualizations/.
    Dipakai oleh iframe di index.html untuk menampilkan chart Plotly.
    Path folder diambil dari flask_config.json — bukan hardcode.
    """
    from flask import send_file, abort
    import re

    # Sanitasi nama file — hanya huruf, angka, underscore, strip, .html
    if not re.match(r'^[\w\-]+\.html$', filename):
        abort(400)

    cfg      = get_flask_config()
    viz_paths = cfg.get('visualization_paths', {})

    # Cari path yang filename-nya cocok
    target_path = None
    for key, path in viz_paths.items():
        if os.path.basename(path) == filename:
            target_path = resolve_path(path)
            break

    # Fallback: langsung construct path jika tidak ditemukan di config
    if not target_path:
        target_path = os.path.join(os.getcwd(), 'flask_components', 'visualizations', filename)

    if not os.path.exists(target_path):
        print(f"❌ Viz file tidak ditemukan: {target_path}")
        abort(404)

    return send_file(target_path, mimetype='text/html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint AJAX untuk prediksi berita palsu.
    Request  : POST JSON { "text": "judul berita" }
    Response : JSON { label, confidence, prob_fake, prob_real, text }
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Field "text" dibutuhkan.'}), 400

        text = str(data['text']).strip()
        if not text:
            return jsonify({'error': 'Teks tidak boleh kosong.'}), 400
        if len(text) > 1000:
            return jsonify({'error': 'Teks terlalu panjang (maks. 1000 karakter).'}), 400

        result = predict_news(text)
        if result is None:
            return jsonify({'error': 'Model belum siap. Coba beberapa saat lagi.'}), 503

        result['text'] = text[:200]
        return jsonify(result)

    except Exception as e:
        print(f"❌ /predict error: {e}")
        return jsonify({'error': 'Internal server error.'}), 500


# ─── Login / Logout ───────────────────────────────────────────────────────────

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if username == 'admin' and password == 'admin123':
            session['username'] = username
            flash(f'Selamat datang, {username}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Username atau password salah.', 'danger')

    return render_template('admin/login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('Anda telah logout.', 'info')
    return redirect(url_for('homepage'))


# ═══════════════════════════════════════════════════════════════════════════════
#  ADMIN ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard admin — semua data dari file output notebook."""
    try:
        data = get_all_research_data()
        data['model_loaded'] = (_model is not None)
        data['last_update']  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return render_template('dashboard/index.html', **data)
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        flash(f'Error loading dashboard: {str(e)}', 'danger')
        return render_template('dashboard/index.html',
            model_loaded=False,
            last_update=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )




@app.route('/analisis')
@login_required
def analisis():
    """Halaman hasil analisis model BERT — semua data dari file output notebook."""
    try:
        data = get_all_research_data()
        data['model_loaded'] = (_model is not None)
        data['last_update']  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return render_template('analisis/index.html', **data)
    except Exception as e:
        print(f"❌ Analisis error: {e}")
        flash(f'Error loading analisis: {str(e)}', 'danger')
        return render_template('analisis/index.html',
            model_loaded=False,
            last_update=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )


@app.route('/tentang')
@login_required
def tentang():
    """Halaman tentang penelitian."""
    try:
        data = get_all_research_data()
        return render_template('tentang/index.html', **data)
    except Exception as e:
        print(f"❌ Tentang error: {e}")
        flash(f'Error loading tentang: {str(e)}', 'danger')
        return render_template('tentang/index.html')

# ═══════════════════════════════════════════════════════════════════════════════
#  ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(error):
    return render_template('layouts/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('layouts/500.html'), 500


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("🚀  BERT Fake News Detector — Flask App")
    print("=" * 60)

    # Baca flask_config.json dan verifikasi semua file output notebook
    cfg = get_flask_config()
    if cfg:
        print("✅  flask_config.json ditemukan\n")
        all_paths = {
            **cfg.get('model_paths',         {}),
            **cfg.get('data_paths',          {}),
            **cfg.get('results_paths',       {}),
            **cfg.get('visualization_paths', {}),
        }
        ok = 0
        for name, path in all_paths.items():
            abs_path = resolve_path(path)
            exists   = os.path.exists(abs_path)
            print(f"  {'✅' if exists else '❌'}  {name:<30} {path}")
            if exists:
                ok += 1
        print(f"\n  {ok}/{len(all_paths)} file output notebook tersedia")
    else:
        print("❌  flask_config.json tidak ditemukan!")
        print("   → Jalankan Cell 12 di Jupyter notebook terlebih dahulu.")

    print("=" * 60)

    # Preload model saat startup agar prediksi pertama tidak lambat
    load_bert_model()

    print("=" * 60)
    print("  Login  : admin / admin123")
    print("  URL    : http://localhost:5000")
    print("  Predict: POST http://localhost:5000/predict")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)
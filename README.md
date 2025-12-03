# Encryption Algorithm Classification using Machine Learning

A comprehensive research project that implements multiple lightweight cryptographic algorithms, generates performance datasets, and uses machine learning to classify encryption algorithms based on their system-level performance characteristics.

## 📁 Project Structure

```
research/
├── dataset/                    # Generated encrypted datasets
│   ├── plaintexts/            # Original binary files (16KB-2048KB)
│   ├── AES/                   # AES-encrypted files
│   ├── PRESENT/               # PRESENT-encrypted files
│   ├── SIMON/                 # SIMON64/128-encrypted files
│   ├── XTEA/                  # XTEA-encrypted files
│   ├── PRINCE/                # PRINCE-encrypted files
│   ├── RECTANGLE/             # RECTANGLE-80-encrypted files
│   ├── ASCONv2/               # ASCON-128 v2 AEAD-encrypted files
│   ├── LEA/                   # LEA-128-encrypted files
│   └── MSEA/                  # MSEA-96-encrypted files
├── metrics/                   # Performance metrics and analysis
├── dataset_generator.py       # Core encryption and data generation
├── app.py                     # Machine learning classification
├── main.tex                   # LaTeX research paper
└── README.md                  # This file
```

## 🔐 Core Components

### 1. Dataset Generator (`dataset_generator.py`)

**Purpose**: Implements 9 lightweight cryptographic algorithms and generates performance datasets for machine learning analysis.

#### Implemented Algorithms:

| Algorithm     | Block Size | Key Size | Mode | Description                        |
| ------------- | ---------- | -------- | ---- | ---------------------------------- |
| **AES**       | 128-bit    | 128-bit  | CTR  | Industry standard (PyCryptodome)   |
| **PRESENT**   | 64-bit     | 80-bit   | CTR  | Lightweight block cipher           |
| **SIMON**     | 64-bit     | 128-bit  | CTR  | NSA lightweight cipher             |
| **XTEA**      | 64-bit     | 128-bit  | CTR  | Extended Tiny Encryption Algorithm |
| **PRINCE**    | 64-bit     | 128-bit  | CTR  | Low-latency block cipher           |
| **RECTANGLE** | 64-bit     | 80-bit   | CTR  | Bit-slice oriented cipher          |
| **ASCONv2**   | -          | 128-bit  | AEAD | Authenticated encryption (winner)  |
| **LEA**       | 128-bit    | 128-bit  | CTR  | Korean block cipher                |
| **MSEA**      | 96-bit     | 96-bit   | CTR  | Modified SEA algorithm             |

#### Key Features:

- **Custom CTR Implementation**: Generic counter mode for all block ciphers
- **Performance Monitoring**: Uses `psutil` to capture system metrics
- **Multiple File Sizes**: Processes 16KB, 64KB, 256KB, 512KB, 1024KB, 2048KB files
- **Metrics Collected**:
  - Execution time (ms)
  - CPU usage (user/system time)
  - Memory consumption (RSS before/after)
  - Throughput (MB/s)
  - File metadata

#### Usage:

```bash
python dataset_generator.py --repeats 3 --filter_sizes_kb "16,64,256,512,1024,2048"
```

#### Output Files:

- `dataset/<ALGO>/ALGO_<filename>.bin` - Encrypted files
- `metrics/perf_metrics.csv` - System performance data
- `metrics/dataset_manifest.csv` - File inventory with labels

### 2. Machine Learning Classifier (`app.py`)

**Purpose**: Trains and evaluates machine learning models to classify encryption algorithms based on performance characteristics.

#### Models Implemented:

| Model                | Type              | Key Features                     |
| -------------------- | ----------------- | -------------------------------- |
| **Random Forest** 🥇 | Ensemble          | 93.6% F1-score, best consistency |
| **Decision Tree** 🥈 | Tree-based        | 93.5% F1-score, interpretable    |
| **LightGBM** 🥉      | Gradient Boosting | 93.4% F1-score, fast inference   |
| **SVM**              | Support Vector    | 62.0% F1-score, poor performance |
| **KNN**              | Distance-based    | 65.6% F1-score, inconsistent     |

#### Analysis Features:

- **Stratified Cross-Validation**: Ensures balanced training/testing
- **Per-Size Analysis**: Evaluates performance across different file sizes
- **Comprehensive Metrics**: F1-score, Precision, Recall, Accuracy
- **Algorithm Difficulty Ranking**: Identifies hard-to-classify algorithms

#### Key Findings:

- **Easiest to Classify**: AES (98.7%), ASCONv2 (97.0%), PRESENT (94.0%)
- **Most Challenging**: SIMON (61.6%), PRINCE (65.2%), XTEA (70.9%)
- **Optimal File Sizes**: 64KB-512KB provide best classification accuracy
- **Random Forest Winner**: 72.2% perfect classifications, only 1.9% failure rate

#### Usage:

```bash
python app.py
```

#### Output Files:

- `metrics/model_overall_results.csv` - Overall model rankings
- `metrics/tables_by_size/F1_*.csv` - F1-scores per file size
- `metrics/tables_by_size/Precision_*.csv` - Precision metrics
- `metrics/tables_by_size/Recall_*.csv` - Recall metrics
- `metrics/tables_by_size/Accuracy_*.csv` - Accuracy per model

### 3. Dataset Folder (`dataset/`)

**Structure**: Organized by encryption algorithm with consistent naming conventions.

#### Plaintexts (`dataset/plaintexts/`):

- **60 binary files** total (10 files × 6 sizes)
- **Size range**: 16KB to 2048KB
- **Format**: `<SIZE>KB_<INDEX>.bin` (e.g., `1024KB_5.bin`)
- **Content**: Random binary data for encryption testing

#### Algorithm Folders:

Each algorithm folder contains 60 encrypted files:

- **Naming**: `<ALGO>_<ORIGINAL_FILENAME>.bin`
- **Example**: `AES_1024KB_5.bin` (AES encryption of `1024KB_5.bin`)
- **Total Files**: 540 encrypted files (9 algorithms × 60 files)

#### Special Cases:

- **ASCONv2**: Outputs include authentication tags (ciphertext || tag)
- **File Sizes**: Encrypted files may differ slightly due to padding/tags

## 🚀 Quick Start

### Prerequisites:

```bash
pip install pycryptodome psutil scikit-learn lightgbm pandas numpy
```

### 1. Generate Dataset:

```bash
# Generate all encrypted files and performance metrics
python dataset_generator.py --repeats 3

# Filter specific file sizes only
python dataset_generator.py --filter_sizes_kb "64,256,512"
```

### 2. Train ML Models:

```bash
# Run comprehensive model evaluation
python app.py
```

### 3. View Results:

- Check `metrics/model_overall_results.csv` for model rankings
- Review `main.tex` for detailed analysis and tables
- Examine per-size performance in `metrics/tables_by_size/`

## 📊 Key Results

### 🏆 Best Model: Random Forest

- **F1-Score**: 93.6%
- **Accuracy**: 93.5%
- **Perfect Classifications**: 72.2%
- **Failure Rate**: 1.9%
- **Consistency**: Excellent across all file sizes

### 📈 Performance by File Size:

| File Size | Random Forest Accuracy |
| --------- | ---------------------- |
| 16 KB     | 77.8% (Challenging)    |
| 64 KB     | 100% (Optimal)         |
| 256 KB    | 100% (Optimal)         |
| 512 KB    | 100% (Optimal)         |
| 1024 KB   | 92.6% (Very Good)      |
| 2048 KB   | 85.2% (Good)           |

### 🔐 Algorithm Classification Difficulty:

- **Easy**: AES, ASCONv2, PRESENT (>94% success rate)
- **Moderate**: LEA, RECTANGLE, MSEA (80-85% success rate)
- **Challenging**: XTEA, PRINCE, SIMON (<71% success rate)

## 🎯 Applications

### Research:

- **Cryptographic Analysis**: Performance profiling of lightweight ciphers
- **Algorithm Comparison**: Systematic evaluation of encryption efficiency
- **Machine Learning**: Novel application of ML to cryptographic classification

### Practical:

- **Malware Analysis**: Identify encryption algorithms in unknown binaries
- **Forensics**: Classify encrypted data without keys
- **Performance Optimization**: Select optimal algorithms for specific use cases

## 📚 Technical Details

### Machine Learning Pipeline:

1. **Feature Extraction**: System performance metrics (time, CPU, memory)
2. **Preprocessing**: Normalization and stratified sampling
3. **Model Training**: 5-fold cross-validation with hyperparameter tuning
4. **Evaluation**: Comprehensive metrics across multiple file sizes
5. **Analysis**: Statistical significance and consistency testing

### Cryptographic Implementation:

- **Pure Python**: All algorithms implemented from scratch (except AES)
- **CTR Mode**: Generic counter mode implementation for block ciphers
- **Key Generation**: Cryptographically secure random keys per encryption
- **Nonce Management**: Proper nonce sizing for different block sizes

### Performance Monitoring:

- **High Precision**: Multiple runs averaged for stable measurements
- **System Metrics**: CPU usage, memory consumption, execution time
- **Throughput Calculation**: Megabytes per second processing rate
- **Resource Tracking**: Memory delta and system resource utilization

## 🔧 Configuration

### Dataset Generation:

- **Repeats**: Adjust `--repeats` for measurement precision (default: 3)
- **File Sizes**: Filter with `--filter_sizes_kb` for specific sizes
- **Output Paths**: Customize `--out_root` and `--metrics_dir`

### Machine Learning:

- **Models**: Modify `models` dictionary in `app.py`
- **Hyperparameters**: Tune parameters in model definitions
- **Cross-Validation**: Adjust `StratifiedKFold` parameters
- **Metrics**: Add custom evaluation metrics

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@article{encryption_classification_2025,
  title={Machine Learning Classification of Lightweight Cryptographic Algorithms},
  author={[Your Name]},
  journal={[Journal/Conference]},
  year={2025},
  note={Random Forest achieves 93.6\% F1-score in encryption algorithm classification}
}
```

## 📄 License

This research project is provided for educational and research purposes. Please ensure compliance with local cryptographic regulations and export controls.

---

**Generated**: October 3, 2025  
**Dataset**: 9 Algorithms × 6 File Sizes × 10 Samples = 540 encrypted files  
**Best Model**: Random Forest (93.6% F1-score, 72.2% perfect classification rate)

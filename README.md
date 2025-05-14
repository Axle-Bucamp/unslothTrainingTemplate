## 🧰 Parquet Summary Tool

A simple and effective utility for summarizing Parquet datasets—designed for LLM dataset preprocessing, exploration, and quality inspection.

---

### 📌 Features

* ✅ Pretty display of the **top N rows** of your dataset.
* 📊 Global stats including:

  * Row and column counts
  * Null value inspection
  * Duplicate detection
  * Column data types
* 🪶 Lightweight and CLI-friendly.
* 🤖 Great for use in LLM pipelines or data validation scripts.

---

### 📦 Installation

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

---

### 🔧 Usage

```bash
python parquet_summary.py --file <path_to_file> --top <n>
```

**Arguments:**

| Flag     | Description                           | Default |
| -------- | ------------------------------------- | ------- |
| `--file` | Path to the Parquet file (required)   | —       |
| `--top`  | Number of top rows to show (optional) | 5       |

---

### ✅ Example

```bash
python parquet_summary.py --file ./data/sample.parquet --top 5
```

Output:

```text
📂 Loading Parquet file: ./data/sample.parquet

🔍 Top 5 Rows:
+------------+---------------+--------+
| column_1   | column_2      | ...    |
+------------+---------------+--------+

📊 Global Dataset Stats:
- Total rows: 10000
- Total columns: 6

📉 Null Values:
+----------+--------+
| Column   | Nulls  |
+----------+--------+

📦 Duplicate Rows:
- Duplicate rows: 0

🔠 Data Types:
+----------+--------+
| Column   | DType  |
+----------+--------+
```

---

### 📁 `requirements.txt`

```text
pandas
pyarrow
tabulate
```

---

### 🧠 Why?

Before training a large language model, inspecting and validating your dataset can help avoid:

* Training on malformed data
* Inconsistent column types
* Text with excessive nulls or duplication

---

### 🚀 License

MIT — free to use and modify.

---

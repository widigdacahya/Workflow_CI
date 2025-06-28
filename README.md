### 📝 Catatan Mengenai Struktur Folder
Sesuai dengan kriteria submission, folder untuk menyimpan skrip training seharusnya bernama `MLProject`. Namun, karena sistem operasi yang saya gunakan untuk development bersifat *case-insensitive*, hal ini menyebabkan konflik nama dengan file `MLProject` yang juga diwajibkan ada di direktori root.

Untuk menghindari konflik teknis dan memastikan kelancaran development baik di lokal maupun di server CI, maka folder tersebut saya ganti namanya menjadi **`modelling_files`**. Semua fungsionalitas dan path di dalam file `MLProject` dan workflow CI telah disesuaikan dengan nama folder baru ini. Keputusan ini diambil untuk menjaga stabilitas dan fungsionalitas proyek.

## 📂 Struktur Repository

Struktur repository untuk mengikuti standar MLflow Project dan alur kerja GitHub Actions.

```
Workflow_CI/
├── .github/
│   └── workflows/
│       └── main.yml          # File definisi workflow GitHub Actions
├── MLProject                 # File "SOP" atau entry point untuk MLflow Project
├── modelling_files/          # Folder berisi semua logika dan dependensi training
│   ├── conda.yaml            # Definisi environment Conda (daftar library)
│   └── modelling.py          # Skrip utama untuk training & hyperparameter tuning
└── preprocessing/
    └── MedicalCost_preprocessing/
        └── insurance_processed.csv # Dataset bersih yang siap digunakan
```

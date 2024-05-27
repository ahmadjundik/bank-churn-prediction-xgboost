# Hapus environment virtual yang ada
rm -rf venv

# Buat environment virtual baru
python -m venv venv

# Aktifkan environment virtual
source venv/bin/activate  # Pada Windows: venv\Scripts\activate

# Instal dependensi dari requirements.txt
pip install --upgrade pip  # Memperbarui pip
pip install -r requirements.txt



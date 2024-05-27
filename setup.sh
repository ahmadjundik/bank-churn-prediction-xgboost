# Hapus environment virtual yang ada
rm -rf venv

# Buat environment virtual baru
python -m venv venv

# Aktifkan environment virtual
source venv/bin/activate  # Pada Windows: venv\Scripts\activate

# Instal dependensi dari requirements.txt
pip install -r requirements.txt

pip install numpy==1.23.3


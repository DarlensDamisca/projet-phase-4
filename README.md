# 🩺 Deteksyon Nemoni ak Entèlijans Atifisyèl

## Sistèm otomatik pou detekte nemoni nan radyografi pwatrin yo

---

## 📋 Deskripsyon Pwojè a

Pwojè sa a devlope yon sistèm entèlijans atifisyèl ki ka otomatikman detekte nemoni nan imaj radyografi pwatrin yo. Objektif prensipal la se ede klinik ak lopital nan zòn riral Ayiti kote pa gen ase radyològ espesyalis.

### 🎯 Objektif yo
- Aksè pi rapid nan dyagnostik nemoni yo
- Redui tan pou tann rezilta yo
- Sipòte doktè yo nan zòn ki manke espesyalis
- Amelyore presizyon dyagnostik yo

---

## 📊 Done yo ak Pèfòmans

### Dataset
- **Total Imaj**: 5,856 radyografi pwatrin
- **Klas Normal**: 1,583 imaj
- **Klas Nemoni**: 4,273 imaj
- **Sous**: Chest X-Ray Images (Pneumonia) sou Kaggle

### Pi Bon Rezilta (VGG16 Transfer Learning)
- **Presizyon**: 91.51%
- **AUC**: 0.9461
- **Sensitivite**: 97.95% (detekte nemoni)
- **Spesifisite**: 80.77% (idantifye moun ki sèn)

---

## 🚀 Kòmanse Rapid

### Kondisyon yo
```bash
Python 3.8+
TensorFlow 2.x
Keras
NumPy
Pandas
Matplotlib
Seaborn
Pillow
OpenCV
Scikit-learn
```

### Enstalasyon
```bash
# Clone repository a
git clone https://github.com/your-username/deteksyon-nemoni-ai.git
cd deteksyon-nemoni-ai



# Telechaje dataset la sou Kaggle
# Mete dosye "chest_xray" nan racine pwojè a
```

### Itilizasyon
```python
# Chaje modèl pi bon an
import tensorflow as tf
model = tf.keras.models.load_model('vgg16_nemoni_final.keras')

# Fè prediksyon sou nouvo imaj
from PIL import Image
import numpy as np

def predi_nemoni(chemen_imaj):
    img = Image.open(chemen_imaj)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        return f"Nemoni - Konfyans: {prediction[0][0]:.2%}"
    else:
        return f"Normal - Konfyans: {1-prediction[0][0]:.2%}"

# Egzanp itilizasyon
rezilta = predi_nemoni("path/to/chest_xray.jpg")
print(rezilta)
```

---

## 🏗️ Achitèkti Modèl yo

Nou teste 4 modèl diferan:

### 1. CNN de Baz
- 3 kouch konvolusyon ak max pooling
- 2 kouch dans ak dropout
- **Rezilta**: 82.05% presizyon, 0.926 AUC

### 2. VGG16 Transfer Learning ⭐ (Pi Bon)
- Pre-antrene sou ImageNet
- Fine-tuning ak kouch pèsonalize
- **Rezilta**: 91.51% presizyon, 0.946 AUC

### 3. ResNet50 Transfer Learning
- Achitèkti residual ak skip connections
- **Rezilta**: 82.05% presizyon, 0.904 AUC

### 4. EfficientNetB0 Transfer Learning
- Modèl optimize pou efikasite
- **Rezilta**: Pa travay byen (37.5% presizyon)

---

## 📁 Òganizasyon Dosye yo

```
deteksyon-nemoni-ai/
├── README.md
|
├── phase4.ipynb                 # Notebook prensipal
├── models/
│   ├── vgg16_nemoni_final.keras
│   ├── vgg16_nemoni_final.weights.h5
│   └── vgg16_nemoni_final_architecture.json
├── data/
│
│      
```

---

## 🔬 Metòd ak Teknoloji yo

### Preparasyon Done
- Redimansyone imaj yo nan 224x224 piksèl
- Nòmalizasyon valè yo ant 0 ak 1
- Ogmantasyon done (rotasyon, deplase, zoom)
- Divizyon: antrennman (5,216) / validasyon (16) / tès (624)

### Antrennman
- **Optimizer**: Adam
- **Loss Function**: Binary crossentropy
- **Callbacks**: EarlyStopping, ReduceLROnPlateau
- **Learning Rate**: 0.0001 pou transfer learning

### Evalyasyon
- Matris konfizyon
- Rapò klasifikasyon
- Koub ROC ak AUC
- Analiz erè yo

---

## ⚠️ Limit ak Defi yo

### Defi Teknik
- Dezekilib klas yo (74.3% nemoni, 25.7% nòmal)
- Risk overfitting sou done validasyon
- EfficientNetB0 pa travay jan nou te espere

### Limit Praktik
- Bezwen entènèt pou aksè sistèm nan
- Kalite imaj yo enpòtan pou bon rezilta
- Bezwen fòmasyon pou pèsonèl yo

---

## 🚀 Pwochen Etap yo

- [ ] Teste nan anviwonman klinik reyèl
- [ ] Ajoute plis done 
- [ ] Optimize modèl la pou aparèy mobil
- [ ] Kreye yon aplikasyon web senp


---

## 🤝 Kontribisyon

Nou aksepte kontribisyon yo! Tanpri:

1. Fork repository a
2. Kreye yon branch pou feature nou an (`git checkout -b feature/nouvo-feature`)
3. Commit chanjman yo (`git commit -m 'Ajoute nouvo feature'`)
4. Push nan branch la (`git push origin feature/nouvo-feature`)
5. Kreye yon Pull Request

---

## 📜 Lisans

Pwojè sa a sou lisans MIT - gade fichye [LICENSE](LICENSE) pou plis detay.



---

## 🙏 Remèsiman

- Dataset sou Kaggle pou done yo
- TensorFlow ak Keras pou framework yo
- Kominote open source la pou bibliyotèk ak zouti yo

---

## 📧 Kontak


## 👤 Ote
- **Nom :** **Darlens DAMISCA**
- **Email :** Bdamisca96@gmail.com 
- **LinkedIn :** www.linkedin.com/in/darlens-damisca-dev0529

### 📂 Repositories
- **GitHub Principal :** https://github.com/DarlensDamisca/projet-phase-4.git

---

*Pwojè sa a fèt pou ede sistèm sante an Ayiti ak teknoloji AI.*
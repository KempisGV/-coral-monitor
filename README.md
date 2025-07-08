# 🪸 Coral Health Monitor

Monitor de salud de corales mediante inteligencia artificial

---

## 🚀 Demo en Línea

Puedes probar la app directamente en la web:  
👉 **[https://coral-monitor-production.up.railway.app/](https://coral-monitor-production.up.railway.app/)**

---

## 📦 Requisitos para ejecución local

- Python 3.10 o superior  
- Pip (gestor de paquetes de Python)  
- Se recomienda usar [venv](https://docs.python.org/3/library/venv.html) o [conda](https://docs.conda.io/)

---

## 🔥 Instalación y ejecución local

1. **Clona el repositorio:**

   ```bash
   git clone https://github.com/KempisGV/-coral-monitor.git
   cd -coral-monitor

2. **Crea y activa un entorno virtual (opcional pero recomendado):**

   ```bash
   python -m venv venv
   # En Windows:
    env\Scripts\activate
   # En Mac/Linux:
    source venv/bin/activat

3. **Instala las dependencias:**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt

3. **Ejecuta la aplicación:**

   ```bash
   streamlit run coral_app.py

# Snow-Rainbow-Classifier 🌈❄️

An AI-powered Telegram bot built with Python and Machine Learning that can classify images and detect spam emails using trained machine learning models.

## 🚀 Features

This project includes three different Machine Learning systems:

### 1. Snow-Rainbow Classification 🌈❄️
- Identifies whether an image contains a Rainbow or Snow.
- Uses the K-Nearest Neighbors (KNN) algorithm.
- Processes images using OpenCV.

### 2. Cat-Dog Classification 🐶🐈
- Classifies images into two categories: Cats and Dogs.
- Uses KNN for image classification.
- Extracts image features using NumPy and OpenCV.

### 3. Spam Email Detection ✉️
- Detects whether an email is Spam or Not Spam.
- Uses Logistic Regression.
- Analyzes word frequency features.
- Trained using an email dataset.

## 🛠️ Technologies Used

- Python
- NumPy
- OpenCV
- Pandas
- Scikit-learn
- Telegram Bot API
- Machine Learning
- K-Nearest Neighbors (KNN)
- Logistic Regression

## 📂 Project Structure

```text
Snow-Rainbow-Classifier/
│
├── main.py
├── requirements.txt
├── .gitignore
│
└── data/
    ├── rainbow/
    ├── snow/
    ├── cat/
    ├── dog/
    └── emails.csv
```

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Snow-Rainbow-Classifier.git
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Your Telegram Bot Token

Set your Telegram Bot Token as an environment variable.

**Windows PowerShell:**

```powershell
$env:TELEGRAM_BOT_TOKEN="YOUR_BOT_TOKEN"
```

### 4. Run the Project

```bash
python main.py
```

## 🧠 Machine Learning Algorithms

| Feature | Algorithm |
|---|---|
| Rainbow / Snow | K-Nearest Neighbors |
| Cat / Dog | K-Nearest Neighbors |
| Spam Detection | Logistic Regression |

## 📊 Image Processing

Images are processed using OpenCV:

- Grayscale conversion
- Image resizing to 200 × 200
- Flattening image arrays
- Machine Learning classification

## 🔐 Security

- Telegram Bot Token is stored in an environment variable.
- No personal file paths are hardcoded.
- Sensitive credentials should never be uploaded to GitHub.

## 📌 Future Improvements

- Improve classification accuracy.
- Add more image categories.
- Develop a graphical user interface.
- Deploy the Telegram bot online.
- Explore Deep Learning models such as CNNs.

## 👨‍💻 Author

Developed as a Python and Machine Learning project.

---

⭐ If you find this project interesting, consider giving it a star!


import numpy as np
import cv2
import os
import pandas as pd
import re

from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters
)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not TOKEN:
    raise RuntimeError("Telegram bot token is not configured.")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

RAINBOW_PATH = DATA_DIR / "rainbow"
SNOW_PATH = DATA_DIR / "snow"

CAT_PATH = DATA_DIR / "cat"
DOG_PATH = DATA_DIR / "dog"

EMAILS_PATH = DATA_DIR / "emails.csv"


def load_images(path, label, x, y):

    if not path.exists():
        print(f"Dataset directory not found: {path.name}")
        return

    for filename in os.listdir(path):

        file_path = path / filename

        img = cv2.imread(str(file_path), 0)

        if img is None:
            continue

        img = cv2.resize(img, (200, 200))
        img = img.flatten()

        x.append(img)
        y.append(label)


def train_rainbow_snow():

    x = []
    y = []

    load_images(RAINBOW_PATH, 0, x, y)
    load_images(SNOW_PATH, 1, x, y)

    x = np.array(x)
    y = np.array(y)

    if len(x) == 0:
        raise ValueError("Rainbow/Snow dataset is empty.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    knn = KNeighborsClassifier(n_neighbors=7)

    knn.fit(x_train, y_train)

    score = knn.score(x_test, y_test)

    print(f"Rainbow-Snow Accuracy: {score * 100:.2f}%")

    return knn


def train_cat_dog():

    x = []
    y = []

    load_images(CAT_PATH, 0, x, y)
    load_images(DOG_PATH, 1, x, y)

    x = np.array(x)
    y = np.array(y)

    if len(x) == 0:
        raise ValueError("Cat/Dog dataset is empty.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    knn = KNeighborsClassifier(n_neighbors=7)

    knn.fit(x_train, y_train)

    score = knn.score(x_test, y_test)

    print(f"Cat-Dog Accuracy: {score * 100:.2f}%")

    return knn


def spamD():

    data = pd.read_csv(EMAILS_PATH)

    data.drop(
        columns=["Email No."],
        inplace=True
    )

    data.drop_duplicates(inplace=True)

    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    feature_names = x.columns.tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    lng = LogisticRegression(max_iter=1000)

    lng.fit(x_train, y_train)

    score = lng.score(x_test, y_test)

    print(f"Spam Accuracy: {score * 100:.2f}%")

    def email_to_features(email):

        words = re.findall(
            r"[a-zA-Z]+",
            email.lower()
        )

        word_counts = Counter(words)

        features = [
            word_counts.get(word.lower(), 0)
            for word in feature_names
        ]

        return np.array([features], dtype=float)

    return lng, email_to_features


print("Training Rainbow-Snow model...")

rainbow_snow_knn = train_rainbow_snow()

print("Training Cat-Dog model...")

cat_dog_knn = train_cat_dog()

print("Training Spam model...")

spam_model, email_to_features = spamD()

print("All models trained successfully!")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    button1 = InlineKeyboardButton(
        text="Snow-Rainbow Analysis 🌩️",
        callback_data="rainbow_snow"
    )

    button2 = InlineKeyboardButton(
        text="Cat-Dog Analysis 🐶🐈",
        callback_data="cat_dog"
    )

    button3 = InlineKeyboardButton(
        text="Spam Email Detection ✉️",
        callback_data="spam"
    )

    keyboard = [
        [button1],
        [button2],
        [button3]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        f"Hello {update.effective_user.first_name}! "
        "How Can I Help You?",
        reply_markup=reply_markup
    )


async def button_click(update: Update, context: ContextTypes.DEFAULT_TYPE):

    query = update.callback_query

    await query.answer()

    if query.data == "rainbow_snow":

        context.user_data["model"] = "rainbow_snow"

        await query.edit_message_text(
            "🌩️ Snow-Rainbow Analysis\n\n"
            "Send Me a Picture to Analysis..."
        )

    elif query.data == "cat_dog":

        context.user_data["model"] = "cat_dog"

        await query.edit_message_text(
            "🐶🐈 Cat-Dog Analysis\n\n"
            "Send me a Picture to Analysis..."
        )

    elif query.data == "spam":

        context.user_data["model"] = "spam"

        await query.edit_message_text(
            "✉️ Spam Email Detection\n\n"
            "Send me the email text..."
        )


async def analyze_image(update: Update, context: ContextTypes.DEFAULT_TYPE):

    model_type = context.user_data.get("model")

    if model_type is None:

        await update.message.reply_text(
            "First of All Click on a Button❌"
        )

        return

    if model_type == "spam":

        await update.message.reply_text(
            "Please Send Me The Email Text ✉️"
        )

        return

    photo = update.message.photo[-1]

    file = await context.bot.get_file(photo.file_id)

    user_id = update.effective_user.id

    image_path = BASE_DIR / f"user_{user_id}.jpg"

    try:

        await file.download_to_drive(str(image_path))

        img = cv2.imread(str(image_path), 0)

        if img is None:

            await update.message.reply_text(
                "I Can't Read the Photo❌"
            )

            return

        img = cv2.resize(img, (200, 200))
        img = img.flatten()

        if model_type == "rainbow_snow":

            prediction = rainbow_snow_knn.predict([img])[0]

            probabilities = rainbow_snow_knn.predict_proba([img])[0]

            confidence = max(probabilities) * 100

            result = "🌈 Rainbow" if prediction == 0 else "❄️ Snow"

        elif model_type == "cat_dog":

            prediction = cat_dog_knn.predict([img])[0]

            probabilities = cat_dog_knn.predict_proba([img])[0]

            confidence = max(probabilities) * 100

            result = "🐈 Cat" if prediction == 0 else "🐶 Dog"

        else:

            result = "Unknown"
            confidence = 0

        await update.message.reply_text(
            f"🔎 Result\n\n"
            f"{result}\n\n"
            f"Confidence: {confidence:.2f}%"
        )

    finally:

        if image_path.exists():
            image_path.unlink()


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):

    model_type = context.user_data.get("model")

    if model_type == "spam":

        email = update.message.text

        new_x = email_to_features(email)

        prediction = spam_model.predict(new_x)[0]

        probabilities = spam_model.predict_proba(new_x)[0]

        classes = spam_model.classes_

        prediction_index = list(classes).index(prediction)

        confidence = probabilities[prediction_index] * 100

        if prediction == 1:

            result = "NOT SPAM ✅"

        else:

            result = "Spam🚨"

        await update.message.reply_text(
            f"🔎 Result\n\n"
            f"{result}\n\n"
            f"Confidence: {confidence:.2f}%"
        )

        return

    await update.message.reply_text(
        "Please Send Me Photo 📸"
    )


app = (
    ApplicationBuilder()
    .token(TOKEN)
    .build()
)

app.add_handler(CommandHandler("start", start))

app.add_handler(CallbackQueryHandler(button_click))

app.add_handler(
    MessageHandler(filters.PHOTO, analyze_image)
)

app.add_handler(
    MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
)


if __name__ == "__main__":

    print("Bot is running...")

    app.run_polling()

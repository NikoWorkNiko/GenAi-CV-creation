# AI-Powered CV tailoring

This is a Django-based web application that helps users optimize their CV for a specific job posting using Google’s Gemini API.
Users can upload their CV, paste a link to a job posting, and the tool will:
1. Parse the existing CV and the jop posting.
2. Allow manual edits or additions.
3. Suggest tailored improvements to align the CV with the job ad.
4. Automatically generate a PDF version of the optimized CV.

## Setup Instructions

Follow these steps to set up the development environment and run the project.

### 1. Clone the Repository

Use git clone command or github interface's green button "clone".

### 2. Set Up Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```
### 4. API-Key

Open [config.yaml](GenAi_Django/config/config.yaml) and insert your API key in the defined cell.

### 5. Run app in terminal

Apply database migrations
```bash
python manage.py makemigrations
```

```bash
python manage.py migrate
```

Start the development server at http://127.0.0.1:8000/
```bash
python manage.py runserver
```


To stop the server close the terminal or use **control + C** key combination.

## Tech Stack
- Framework: Django (Python) -> [settings.py](GenAi_Django/settings.py)
- AI Engine: Gemini API (by Google)
- Frontend: [HTML](templates), [CSS](static/css), [JavaScript](static/js)
- PDF Generation: LaTeX into [PDF-API](https://latexonline.cc/)
- File Handling: Static folder [json](static/json)

## AI Prompts & Logic
- All prompt logic is defined in [views.py](CV/views.py). It generates suggestions dynamically based on the job ad and the user’s current CV to ensure optimal alignment.
- [urls.py](GenAi_Django/urls.py) defines the available page routes


from django.http import JsonResponse
from django.shortcuts import render
import os
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

from GenAi_Django.forms import DateiUploadForm, JobPostingForm
from PyPDF2 import PdfReader
import docx2txt
from GenAi_Django.utils import UnifiedChatClient
import json
from django.shortcuts import render, redirect
import requests
from bs4 import BeautifulSoup
import re
import traceback
from deepmerge import always_merger
from deepmerge import Merger
import copy
from datetime import datetime


def home(request):
    return render(request, 'home.html')


def extract_text_from_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == '.pdf':
            reader = PdfReader(filepath)
            return "\n".join(p.extract_text() for p in reader.pages if p.extract_text())

        elif ext == '.docx':
            return docx2txt.process(filepath)

        elif ext == '.txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()

        else:
            return f"Unsupported file type: {ext}"

    except Exception as e:
        return f"Error reading file: {e}"


def step1(request):
    text_vorschau = None
    personal_json = None
    professional_json = None

    if request.method == 'POST':
        print(">>> POST received")

        form = DateiUploadForm(request.POST, request.FILES)

        if form.is_valid():
            print(">>> Form is valid")

            datei = request.FILES['datei']
            ext = os.path.splitext(datei.name)[1].lower()
            print(f">>> Uploaded file extension: {ext}")

            if ext not in ['.pdf', '.docx', '.txt']:
                print(">>> Invalid file type")
                form.add_error('datei', 'Nur PDF, DOCX oder TXT erlaubt.')
            else:
                uploads_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
                os.makedirs(uploads_dir, exist_ok=True)
                filepath = os.path.join(uploads_dir, datei.name)

                with open(filepath, 'wb+') as destination:
                    for chunk in datei.chunks():
                        destination.write(chunk)

                try:
                    full_text = extract_text_from_file(filepath)
                    print(">>> Text extracted (preview):")
                    print(full_text[:500])
                    text_vorschau = full_text[:1000]
                except Exception as e:
                    print(">>> ERROR extracting text:", e)
                    form.add_error('datei', f"Fehler beim Lesen der Datei: {e}")
                    return render(request, 'step1.html', {'form': form})

                json_dir = os.path.join(settings.MEDIA_ROOT, 'json')
                os.makedirs(json_dir, exist_ok=True)

                personal_info_output = os.path.join(json_dir, 'personal_info.json')
                professional_info_output = os.path.join(json_dir, 'professional_info.json')

                try:
                    client = UnifiedChatClient()

                    # === PERSONAL INFO ===
                    personal_format_path = os.path.join(settings.BASE_DIR, 'static', 'json', 'personal_info_template.json')
                    with open(personal_format_path, 'r', encoding='utf-8') as f:
                        personal_format = f.read()

                    response_personal = client.chat([
                        {
                            "role": "system",
                            "content": (
                                "Extract structured personal information from the following resume text. "
                                "Use the given JSON format. Return ONLY valid JSON with no placeholders or markdown:\n\n"
                                f"{personal_format}"
                            )
                        },
                        {
                            "role": "user",
                            "content": f"Resume Text:\n\n{full_text}"
                        }
                    ])

                    print(">>> LLM response (personal):")
                    print(response_personal)

                    match = re.search(r'\{.*\}', response_personal, re.DOTALL)
                    if match:
                        personal_json = json.loads(match.group())
                        with open(personal_info_output, 'w', encoding='utf-8') as f:
                            json.dump(personal_json, f, ensure_ascii=False, indent=2)
                    else:
                        raise ValueError("Could not extract JSON from personal LLM response.")

                    # === PROFESSIONAL INFO ===
                    professional_format_path = os.path.join(settings.BASE_DIR, 'static', 'json', 'professional_info_template.json')
                    with open(professional_format_path, 'r', encoding='utf-8') as f:
                        professional_format = f.read()

                    response_professional = client.chat([
                        {
                            "role": "system",
                            "content": (
                                "You are an assistant that extracts structured professional data from a resume."
                                "Use the following JSON format. Overwrite ALL existing fields."
                                "Use "" for missing strings. Do not use the string 'None'. Always include all required keys."
                                "Return ONLY valid JSON with no markdown or extra text:\n\n"
                                f"{professional_format}"
                            )
                        },
                        {
                            "role": "user",
                            "content": f"Resume Text:\n\n{full_text}"
                        }
                    ])

                    print(">>> LLM response (professional):")
                    print(response_professional)

                    match = re.search(r'\{.*\}', response_professional, re.DOTALL)
                    if match:
                        professional_json = json.loads(match.group())

                        if os.path.exists(professional_info_output):
                            os.remove(professional_info_output)

                        with open(professional_info_output, 'w', encoding='utf-8') as f:
                            json.dump(professional_json, f, ensure_ascii=False, indent=2)
                    else:
                        raise ValueError("Could not extract JSON from professional LLM response.")

                    return redirect('step2')

                except Exception as e:
                    print(">>> ERROR in LLM or JSON block:")
                    traceback.print_exc()
                    personal_json = {"error": str(e)}

        else:
            print(">>> Form invalid:")
            print(form.errors)

    else:
        form = DateiUploadForm()

    return render(request, 'step1.html', {
        'form': form,
        'text': text_vorschau,
        'extracted': personal_json,
        'step': 1
    })


def extract_job_info_with_llm(text, template_path):
    with open(template_path, 'r', encoding='utf-8') as f:
        template_dict = json.load(f)

    # Wir extrahieren nur den inneren Block (der eigentlich relevant ist)
    job_posting_template = template_dict["job_posting_information"]
    job_posting_str = json.dumps(job_posting_template, indent=2, ensure_ascii=False)

    client = UnifiedChatClient()

    response = client.chat([
        {
            "role": "system",
            "content": (
                "You are an assistant that extracts structured job posting information from job postings "
                "The user will provide raw posting text. "
                "Your job is to extract the job postings requirements and other information"
                "Assume level of skill profficiency (zero, low, medium, high) on factors: years of experience, company size, job position title"
                f"in the following JSON format:\n\n{job_posting_str}"
                "\nRespond ONLY with valid JSON matching this format. Do not include any explanations."
            )
        },
        {
            "role": "user",
            "content": f"Text to extract values from:\n{text}"
        }
    ])

    cleaned_response = response.strip().removeprefix("```json").removesuffix("```").strip()

    # Nur das innere Objekt wird ersetzt
    updated_data = {
        "job_posting_information": json.loads(cleaned_response)
    }
    return updated_data


def extract_visible_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        for tag in soup(['script', 'style', 'nav', 'footer', 'head']):
            tag.decompose()

        text = soup.get_text(separator=' ')
        cleaned_text = ' '.join(text.split())
        return cleaned_text
    except Exception as e:
        print(f"Failed to extract text from URL: {e}")
        return ""


def step2(request):
    extracted_text = None
    filled_json = None
    pdf_error = None

    if request.method == 'POST':
        print(">>> POST received")

        form = JobPostingForm(request.POST, request.FILES)

        if form.is_valid():
            print(">>> Form is valid")

            job_url = form.cleaned_data['job_url']
            job_pdf = form.cleaned_data['job_pdf']

            # === 1. Extract text ===
            if job_url:
                extracted_text = extract_visible_text_from_url(job_url)
                print(">>> Text extracted from URL")

            elif job_pdf:
                ext = os.path.splitext(job_pdf.name)[1].lower()
                if ext == '.pdf':
                    try:
                        reader = PdfReader(job_pdf)
                        extracted_text = "\n".join(page.extract_text() or "" for page in reader.pages)
                        print(">>> Text extracted from PDF")
                    except Exception as e:
                        pdf_error = f"Error reading PDF: {e}"
                        print(pdf_error)
                else:
                    pdf_error = f"Unsupported file type: {ext}"
                    print(pdf_error)

            # === 2. LLM call & JSON save ===
            if extracted_text:
                # Save raw job posting text for future use
                text_output_path = os.path.join(settings.MEDIA_ROOT, 'json', 'job_posting_raw.txt')
                with open(text_output_path, 'w', encoding='utf-8') as f:
                    f.write(extracted_text)

                template_path = os.path.join(settings.BASE_DIR, 'static', 'json', 'job_posting_information_template.json')
                output_path = os.path.join(settings.MEDIA_ROOT, 'json', 'job_posting_information.json')

                try:
                    filled_json = extract_job_info_with_llm(extracted_text, template_path)

                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(filled_json, f, ensure_ascii=False, indent=2)

                    print(">>> JSON written successfully")
                    return redirect('step2b')

                except Exception as e:
                    pdf_error = f"Error during LLM processing: {e}"
                    print(pdf_error)

    else:
        form = JobPostingForm()

    return render(request, 'step2.html', {
        'form': form,
        'extracted_text': extracted_text,
        'pdf_error': pdf_error,
        'json_output': json.dumps(filled_json, indent=2, ensure_ascii=False) if filled_json else None,
        'step': 2
    })

# === Functions Step 2b Extract Company values ===

def extract_company_values_with_llm(text, template_path):
    import json
    from GenAi_Django.utils import UnifiedChatClient

    with open(template_path, 'r', encoding='utf-8') as f:
        template_dict = json.load(f)

    # Get expected keys
    expected_keys = template_dict["company_values_information"].keys()
    company_values_str = json.dumps(template_dict["company_values_information"], indent=2, ensure_ascii=False)

    client = UnifiedChatClient()
    response = client.chat([
        {
            "role": "system",
            "content": (
                "You are an assistant that extracts structured company values, job-related values, and desired character traits "
                "from company websites or career pages. The user will provide raw page text. "
                "Your job is to extract the core cultural values, expectations, and soft factors "
                f"in the following JSON format:\n\n{company_values_str}"
                "\nRespond ONLY with valid JSON matching this format. Do not include any explanations or markdown."
            )
        },
        {
            "role": "user",
            "content": f"Text to extract values from:\n{text}"
        }
    ])

    cleaned_response = response.strip().removeprefix("```json").removesuffix("```").strip()
    parsed = json.loads(cleaned_response)

    # Keep only expected keys
    return {key: parsed.get(key, []) for key in expected_keys}


# full step 2b - company values - using helper-functions
def step2b_company_values_view(request):
    extracted_text = None
    company_values = None
    url_error = None

    if request.method == 'POST':
        print(">>> POST received for Step 2b")

        form = JobPostingForm(request.POST, request.FILES)

        if form.is_valid():
            print(">>> Step 2b: Form is valid")

            company_url = form.cleaned_data['job_url']
            if company_url:
                print(f">>> Step 2b: extracted company_url = {company_url}")
                extracted_text = extract_visible_text_from_url(company_url)
                print(">>> Step 2b: Text extracted from URL")

                if extracted_text:
                    text_output_path = os.path.join(settings.MEDIA_ROOT, 'json', 'company_values_raw.txt')
                    with open(text_output_path, 'w', encoding='utf-8') as f:
                        f.write(extracted_text)

                    template_path = os.path.join(settings.BASE_DIR, 'static', 'json', 'company_value_information_template.json')
                    job_path = os.path.join(settings.MEDIA_ROOT, 'json', 'job_posting_information.json')

                    try:
                        company_values = extract_company_values_with_llm(extracted_text, template_path)

                        # Load and merge into job_posting_information.json
                        with open(job_path, 'r', encoding='utf-8') as f:
                            job_data = json.load(f)

                        job_info = job_data.get("job_posting_information", {})
                        job_info["company_specific_values"] = company_values.get("company_specific_values", [])
                        job_info["desired_character_traits"] = company_values.get("desired_character_traits", [])

                        job_data["job_posting_information"] = job_info

                        with open(job_path, 'w', encoding='utf-8') as f:
                            json.dump(job_data, f, ensure_ascii=False, indent=2)

                        print(">>> Step 2b: Company values merged and saved")
                        return redirect('step3')  # Or 'step3b' if more appropriate

                    except Exception as e:
                        url_error = f"Error during LLM processing: {e}"
                        print(url_error)
            else:
                return redirect('step3')

    else:
        form = JobPostingForm()

    return render(request, 'step2b.html', {
        'form': form,
        'extracted_text': extracted_text,
        'url_error': url_error,
        'json_output': json.dumps(company_values, indent=2, ensure_ascii=False) if company_values else None,
        'step': 2
    })

def step3(request):
    json_path = os.path.join(settings.MEDIA_ROOT, 'json', 'personal_info.json')  # ⬅ CHANGED

    if request.method == 'POST':
        updated_data = {
            "first_name": request.POST.get("first_name", ""),
            "last_name": request.POST.get("last_name", ""),
            "email": request.POST.get("email", ""),
            "phone": request.POST.get("phone", ""),
            "address": request.POST.get("address", "")
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=2)

        return redirect('step3a_edu')

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        data = {}

    return render(request, 'step3.html', {'user': data, 'step':3})


def calculate_keyword_match_percentage(comparison_json: dict) -> float:
    table = comparison_json.get("comparison_table", [])
    if not table:
        return -1

    matched_count = sum(
        1 for row in table
        if row.get("matched_user_skill", "").strip().lower() != "none"
    )
    total_requirements = len(table)
    return round((matched_count / total_requirements) * 100, 2) if total_requirements > 0 else 0.0


# === Helper Functions Step 3a===

merger = Merger(
    [(dict, ["merge"]), (list, ["override"]), (set, ["union"])],
    ["override"],
    ["override"]
)

def load_json_section(file_path, section):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get(section, {})
    except Exception as e:
        print(f"Failed to load JSON section {section}:", e)
        return {}

def save_json_section(file_path, section, new_data):
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {}

        data[section] = new_data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save JSON section {section}:", e)

def get_summary_from_llm(data, topic):
    try:
        client = UnifiedChatClient()
        response = client.chat([
            {
                "role": "system",
                "content": (
                    f"You are an assistant summarizing structured professional resume data related to {topic}. "
                    "Summarize the user’s background and suggest what might be missing."
                    "Talk directly to the user, not in third person."
                )
            },
            {
                "role": "user",
                "content": json.dumps(data)
            }
        ])
        return response.strip()
    except Exception as e:
        traceback.print_exc()
        return f"Failed to summarize {topic}: {e}"

def get_patch_from_llm(data, instruction, topic):
    try:
        client = UnifiedChatClient()
        response = client.chat([
            {
                "role": "system",
                "content": (
                    f"You are an assistant that updates structured professional profile data for the section: {topic}. "
                    "Update the user's professional_info JSON based on their instruction. "
                    "First, explain what you changed in 1-2 short bullet points. Then return the updated JSON as a code block."
                )
            },
            {
                "role": "user",
                "content": f"Current JSON:\n{json.dumps(data)}\nInstruction:\n{instruction}"
            }
        ])

        explanation_match = re.split(r'```(?:json)?', response.strip())
        explanation_text = explanation_match[0].strip() if explanation_match else "No explanation."
        json_match = re.search(r'\{.*\}', response, re.DOTALL)

        if json_match:
            patch = json.loads(json_match.group())
            return explanation_text, patch
        else:
            return "Error: could not extract JSON block from response.", None

    except Exception as e:
        traceback.print_exc()
        return f"AI error: {e}", None

def apply_patch_to_section(original, patch):
    merged = copy.deepcopy(original)
    merger.merge(merged, patch)
    return merged

# === Refactored Step 3a Education View ===

def step3a_edu(request):
    always_merger = Merger(
        [(dict, ["merge"]), (list, ["override"]), (set, ["union"])],
        ["override"],
        ["override"]
    )

    json_path = os.path.join(settings.MEDIA_ROOT, 'json', 'professional_info.json')
    preview = None
    summary = ""
    prof_data = {}

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            prof_data = json.load(f)
    except Exception as e:
        print("Failed to load JSON:", e)
        prof_data = {}

    if request.method == 'POST':
        if 'send_instruction' in request.POST:
            instruction = request.POST.get('instruction', '').strip()
            try:
                client = UnifiedChatClient()
                education_data = prof_data.get("professional_info", {}).get("education", [])

                response = client.chat([
                    {
                        "role": "system",
                        "content": (
                            "You are an assistant that updates a user's education background. "
                            "Focus ONLY on modifying the 'education' section. "
                            "First, explain what you changed in 1–2 short bullet points. Then return the updated JSON as a code block."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Current education section:\n{json.dumps(education_data)}\nInstruction:\n{instruction}"
                    }
                ])

                explanation_match = re.split(r'```(?:json)?', response.strip())
                explanation_text = explanation_match[0].strip() if explanation_match else "No explanation."
                json_match = re.search(r'\[.*\]', response, re.DOTALL)

                if json_match:
                    updated_edu = json.loads(json_match.group())
                    merged = copy.deepcopy(prof_data)
                    merged["professional_info"]["education"] = updated_edu
                    request.session['preview_json'] = merged
                    request.session.modified = True
                    preview = explanation_text
                else:
                    preview = "Error: could not extract JSON block from response."

            except Exception as e:
                traceback.print_exc()
                preview = f"AI error: {e}"

        elif 'apply_preview' in request.POST:
            try:
                preview_json = request.session.pop('preview_json', None)
                if preview_json is None:
                    raise ValueError("No preview data in session.")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(preview_json, f, ensure_ascii=False, indent=2)
                return redirect('step3a_edu')
            except Exception as e:
                preview = f"Could not save preview: {e}"

        elif 'final_save' in request.POST:
            return redirect('step3a_work')

    if request.method == 'GET':
        try:
            client = UnifiedChatClient()
            education_data = prof_data.get("professional_info", {}).get("education", [])
            response = client.chat([
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that summarizes a user's education background. "
                        "Summarize strengths and suggest missing or unclear information."
                        "Structure the suggestions nicely and clearly using bullet points."
                        "Talk directly to the user, not in third person."
                    )
                },
                {
                    "role": "user",
                    "content": json.dumps(education_data)
                }
            ])
            summary = response.strip()
        except Exception as e:
            traceback.print_exc()
            summary = f"Failed to summarize education data: {e}"

    return render(request, 'step3a_edu.html', {
        'prof': prof_data.get('professional_info', {}),
        'education': prof_data.get('professional_info', {}).get('education', []),
        'summary': summary,
        'preview': preview,
        'step': 3
    })


def step3a_work(request):

    always_merger = Merger(
        [(dict, ["merge"]), (list, ["override"]), (set, ["union"])],
        ["override"],
        ["override"]
    )

    json_path = os.path.join(settings.MEDIA_ROOT, 'json', 'professional_info.json')
    preview = None
    summary = ""
    prof_data = {}

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            prof_data = json.load(f)
    except Exception as e:
        print("Failed to load JSON:", e)
        prof_data = {}

    if request.method == 'POST':
        if 'send_instruction' in request.POST:
            instruction = request.POST.get('instruction', '').strip()
            try:
                client = UnifiedChatClient()
                response = client.chat([
                    {
                        "role": "system",
                        "content": (
                            "You are an assistant that updates a user's professional background/work experience. "
                            "Focus ONLY on modifying the 'work_experience' section. Don't add information about university degrees or school"
                            "First, explain what you changed in 1–2 short bullet points. Then return the updated JSON as a code block."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Current work_experience section:\n{json.dumps(prof_data.get('professional_info', {}).get('work_experience', []))}\nInstruction:\n{instruction}"
                    }
                ])

                explanation_match = re.split(r'```(?:json)?', response.strip())
                explanation_text = explanation_match[0].strip() if explanation_match else "No explanation."
                json_match = re.search(r'\[.*\]', response, re.DOTALL)

                if json_match:
                    preview_patch = {"professional_info": {"work_experience": json.loads(json_match.group())}}
                    merged = copy.deepcopy(prof_data)
                    always_merger.merge(merged, preview_patch)
                    request.session['preview_json'] = merged
                    request.session.modified = True
                    preview = explanation_text
                else:
                    preview = "Error: could not extract JSON block from response."

            except Exception as e:
                traceback.print_exc()
                preview = f"AI error: {e}"

        elif 'apply_preview' in request.POST:
            try:
                preview_json = request.session.pop('preview_json', None)
                if preview_json is None:
                    raise ValueError("No preview data in session.")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(preview_json, f, ensure_ascii=False, indent=2)
                return redirect('step3a_work')
            except Exception as e:
                preview = f"Could not save preview: {e}"

        elif 'final_save' in request.POST:
            return redirect('step3a_other')

    if request.method == 'GET':
        try:
            client = UnifiedChatClient()
            work_exp = prof_data.get("professional_info", {}).get("work_experience", [])
            response = client.chat([
                {
                    "role": "system",
                    "content": (
                        "You are an assistant summarizing a user's work experience. "
                        "Summarize strengths and suggest missing or unclear information."
                        "Structure the suggestions nicely and clearly using bullet points."
                        "Talk directly to the user, not in third person."
                    )
                },
                {
                    "role": "user",
                    "content": json.dumps(work_exp)
                }
            ])
            summary = response.strip()
        except Exception as e:
            traceback.print_exc()
            summary = f"Failed to summarize work experience: {e}"

    return render(request, 'step3a_work.html', {
        'prof': prof_data.get('professional_info', {}),
        'work_experience': prof_data.get('professional_info', {}).get('work_experience', []),
        'summary': summary,
        'preview': preview,
        'step': 3
    })


def step3a_other(request):
    always_merger = Merger(
        [(dict, ["merge"]), (list, ["override"]), (set, ["union"])],
        ["override"],
        ["override"]
    )

    json_path = os.path.join(settings.MEDIA_ROOT, 'json', 'professional_info.json')
    preview = None
    summary = ""
    prof_data = {}

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            prof_data = json.load(f)
    except Exception as e:
        print("Failed to load JSON:", e)
        prof_data = {}

    # Define allowed keys for this section
    allowed_keys = {"engagement", "skills", "job_requirements"}

    if request.method == 'POST':
        if 'send_instruction' in request.POST:
            instruction = request.POST.get('instruction', '').strip()
            try:
                client = UnifiedChatClient()

                section_data = {
                    "engagement": prof_data.get("professional_info", {}).get("engagement", []),
                    "skills": prof_data.get("professional_info", {}).get("skills", {}),
                    "job_requirements": {
                        "misc": prof_data.get("professional_info", {}).get("job_requirements", {}).get("misc", {})
                    }
                }

                response = client.chat([
                    {
                        "role": "system",
                        "content": (
                            "You are an assistant that updates a user's engagment, skills and job requirements. "
                            "Focus ONLY on engagement, skills, and job_requirements. Dont add information about university degrees or work_experience."
                            "First, explain what you changed in 1–2 short bullet points. Then return the updated JSON as a code block."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Current JSON:\n{json.dumps(section_data)}\nInstruction:\n{instruction}"
                    }
                ])

                explanation_match = re.split(r'```(?:json)?', response.strip())
                explanation_text = explanation_match[0].strip() if explanation_match else "No explanation."
                json_match = re.search(r'\{.*\}', response, re.DOTALL)

                if json_match:
                    patch_data = json.loads(json_match.group())
                    filtered_patch = {k: v for k, v in patch_data.items() if k in allowed_keys}

                    merged = copy.deepcopy(prof_data)
                    for key, val in filtered_patch.items():
                        if key == "job_requirements":
                            merged["professional_info"].setdefault("job_requirements", {}).update(val)
                        else:
                            merged["professional_info"][key] = val

                    request.session['preview_json'] = merged
                    request.session.modified = True
                    preview = explanation_text
                else:
                    preview = "Error: could not extract JSON block from response."

            except Exception as e:
                traceback.print_exc()
                preview = f"AI error: {e}"

        elif 'apply_preview' in request.POST:
            try:
                preview_json = request.session.pop('preview_json', None)
                if preview_json is None:
                    raise ValueError("No preview data in session.")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(preview_json, f, ensure_ascii=False, indent=2)
                return redirect('step3a_other')
            except Exception as e:
                preview = f"Could not save preview: {e}"

        elif 'final_save' in request.POST:
            return redirect('step3b')

    if request.method == 'GET':
        try:
            client = UnifiedChatClient()
            subset = {
                "engagement": prof_data.get("professional_info", {}).get("engagement", []),
                "skills": prof_data.get("professional_info", {}).get("skills", {}),
                "job_requirements": {
                    "misc": prof_data.get("professional_info", {}).get("job_requirements", {}).get("misc", {})
                }
            }
            response = client.chat([
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that reviews engagement, skills, and job preferences. "
                        "Summarize strengths and suggest missing or unclear information."
                        "Structure the suggestions nicely and clearly using bullet points."
                        "Don't provide suggestions on the job_requirements part"
                        "Talk directly to the user, not in third person."
                    )
                },
                {
                    "role": "user",
                    "content": json.dumps(subset)
                }
            ])
            summary = response.strip()
        except Exception as e:
            traceback.print_exc()
            summary = f"Failed to summarize profile: {e}"

    return render(request, 'step3a_other.html', {
        'prof': prof_data.get('professional_info', {}),
        'summary': summary,
        'preview': preview,
        'step': 3
    })


def step3b(request):

    json_path = os.path.join(settings.MEDIA_ROOT, 'json', 'job_posting_information.json')
    text_path = os.path.join(settings.MEDIA_ROOT, 'json', 'job_posting_raw.txt')
    summary = ""

    # === Load job JSON ===
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except:
        full_data = {"job_posting_information": {}}

    job = full_data.get("job_posting_information", {})

    # === Generate summary from raw job posting text ===
    if os.path.exists(text_path):
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()

            client = UnifiedChatClient()
            response = client.chat([
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that summarizes job postings for users. "
                        "Generate a short summary of the job."
                    )
                },
                {
                    "role": "user",
                    "content": raw_text
                }
            ])
            summary = response.strip()
        except Exception as e:
            summary = f"Summary failed: {e}"

    # === Handle POST submission ===
    if request.method == 'POST':
        # === required_education[] ===
        education = []
        i = 0
        while request.POST.get(f'edu_{i}_degree'):
            education.append({
                "degree": request.POST.get(f'edu_{i}_degree'),
                "field_of_study": request.POST.get(f'edu_{i}_field'),
                "minimum_level": request.POST.get(f'edu_{i}_level'),
                "preferred_institutions": [s.strip() for s in request.POST.get(f'edu_{i}_preferred', '').split(',')]
            })
            i += 1

        # === required_work_experience[] ===
        work = []
        i = 0
        while request.POST.get(f'work_{i}_position'):
            work.append({
                "position": request.POST.get(f'work_{i}_position'),
                "industry": request.POST.get(f'work_{i}_industry'),
                "years_required": request.POST.get(f'work_{i}_years'),
                "responsibilities": [s.strip() for s in request.POST.get(f'work_{i}_responsibilities', '').split(',')]
            })
            i += 1

        # === required_hard_skills[] ===
        hard_skills = []
        i = 0
        while request.POST.get(f'skill_{i}_name'):
            hard_skills.append({
                "skill": request.POST.get(f'skill_{i}_name'),
                "proficiency_level": request.POST.get(f'skill_{i}_proficiency'),
                "certification_required": request.POST.get(f'skill_{i}_cert_required') == "on"
            })
            i += 1

        # === Lists ===
        list_fields = lambda name: [s.strip() for s in request.POST.get(name, "").split(",") if s.strip()]

        # === Misc ===
        misc = {
            "contract_type": request.POST.get("contract_type", ""),
            "working_hours": request.POST.get("working_hours", ""),
            "location": request.POST.get("location", ""),
            "travel_required": request.POST.get("travel_required") == "on",
            "relocation_support": request.POST.get("relocation_support") == "on",
            "language_requirements": list_fields("language_requirements"),
            "additional_notes": request.POST.get("additional_notes", "")
        }

        job["company"] = request.POST.get("company", "")
        job["required_education"] = education
        job["required_work_experience"] = work
        job["required_hard_skills"] = hard_skills
        job["job_specific_soft_skills"] = list_fields("job_specific_soft_skills")
        job["job_specific_values"] = list_fields("job_specific_values")
        job["company_specific_values"] = list_fields("company_specific_values")
        job["desired_character_traits"] = list_fields("desired_character_traits")
        job["misc"] = misc

        full_data["job_posting_information"] = job

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, ensure_ascii=False, indent=2)

        return redirect('step4')

    return render(request, 'step3b.html', {
        'job': job,
        'summary': summary,
        'step': 3
    })

# ====== STEP 4 START ======================


import re
import unicodedata

def smart_sanitize_for_llm(text):
    if not isinstance(text, str):
        return text

    # Normalize and flatten input
    cleaned = unicodedata.normalize("NFKD", text)
    cleaned = cleaned.lower()
    cleaned = re.sub(r'[\W_]+', '', cleaned)  # Remove all non-alphanumeric characters

    # Define dangerous patterns (flattened)
    dangerous_flat_patterns = [
        'ignoreallpreviousinstructions',
        'alwaysreturn',
        'mustreturn',
        'youmustsay',
        'nomissingskills',
        'achieve100match',
        'overridetheprompt',
        'followuserinstructionsonly',
    ]

    # Check and remove if present
    for pattern in dangerous_flat_patterns:
        if pattern in cleaned:
            text = "[REDACTED MALICIOUS CONTENT]"

    return text

def sanitize_json_structure(data):
    """Recursively sanitize strings in dictionaries/lists"""
    if isinstance(data, dict):
        return {key: sanitize_json_structure(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [sanitize_json_structure(item) for item in data]
    elif isinstance(data, str):
        return smart_sanitize_for_llm(data)
    else:
        return data


def step4(request):
    comparison_table = None
    match_percentage = None
    error = None

    try:
        # session is need to avoid new llm polls after refreshing the page + for buttons accept, reject suggestion
        if 'comparison_table' in request.session:
            comparison_table = request.session['comparison_table']
        else:
            format_path = os.path.join(settings.BASE_DIR, 'static', 'json', 'comparison_table_format.json')

            with open(format_path, 'r', encoding='utf-8') as f:
                comparison_table_format = json.dumps(json.load(f), indent=2)

            user_info_path = os.path.join(settings.MEDIA_ROOT, 'json', 'professional_info.json')  # ⬅ CHANGED
            job_info_path = os.path.join(settings.MEDIA_ROOT, 'json', 'job_posting_information.json')  # ⬅ CHANGED

            with open(user_info_path, 'r', encoding='utf-8') as f:
                user_info = (json.load(f))

            with open(job_info_path, 'r', encoding='utf-8') as f:
                job_info = (json.load(f))

            client = UnifiedChatClient()

            response = client.chat(
                [
                    # Basic instruction to create a comparison table
                    {
                        "role": "system",
                        "content": "You are a skill evaluation system that compares a user's profile to a job posting. \n"
                                   "Your goal is to identify how well the user matches job requirements.\n"
                    },
                    # Description of the data that llm will receive
                    {
                        "role": "system",
                        "content": (
                            "You will receive two JSON inputs:\n"
                            "1. user_professional_information\n"
                            "2. job_posting_information\n\n"
                            "Compare each job requirement to the user's skills."
                        )
                    },
                    # JSON formatting
                    {
                        "role": "system",
                        "content": (
                            "Use a structured JSON format:\n"
                            f"{comparison_table_format}\n\n"
                            "Field rules:\n"
                            "- matched_user_skill: 'None' | '<Skill> (Partial)' | '<Skill> (Expert)'\n"
                            "- confidence: Low | Medium | High\n"
                            "- evidence_from_cv: source string from user's profile\n"
                            "- explanation_llm: justify your classification\n"
                            "- category: one of skill-level-0 to skill-level-4\n"
                            "- actions: list of strings: ['edit', 'prompt', 'add']"
                        )
                    },
                    {
                        "role": "system",
                        "content": (
                            "'explanation_llm' field which you generate for each comparison, which should include:\n\n"
                            "1. **Source of the skill match:** Indicate where the skill or related mention was found in the user's professional profile. "
                            "Be specific: e.g., 'Mentioned in project descriptions', 'Listed under skills', 'Found in work experience at Company X', etc.\n"
                            "Don't ask user about the correctness of assessment, just provide your best assessment\n\n"

                            "2. **Reason for skill level choice** (None | Partial | Expert):\n"
                            "- 'Expert': Clear and strong evidence of frequent use, deep experience, or formal responsibility involving the skill.\n"
                            "- 'Partial': Mentioned, but not central; or appears indirectly through tooling or in a minor context.\n"
                            "- 'None': No evidence found in the profile.\n\n"

                            "3. **Assumption reasoning when matched_user_skill is 'None':**\n"
                            "If no match is found, infer a skill that the user might possess based on their expertise \n"
                            "in strongly related technologies or based on user's previous positions or "
                            "based on the company user worked for and famous for using these instruments. "
                            "State your assumption clearly and use word 'likely' in it. For example:\n"
                            "'No direct mention of CMake, but you are an expert in C++ and have worked on large-scale C++ projects — it's 'likely' they you've used CMake or similar build systems.'\n"
                            "If there is no matching skill end with a question: 'Is this assumption correct?'\n"
                            "Don't ask user 'Is this assumption correct?' if you cannot infer any likely skill match\n\n"                            
                            "Don't ask user 'Is this assumption correct?' if you need extra information to infer a like skill match\n\n"                            
                            

                            "Write in a formal tone. Make the explanation informative and traceable.\n"
                            "Use 1–2 concise sentences per point."
                            "Address user directly instead of talking about him in third person"
                        )
                    },

                    # Skill evaluation method
                    {
                        "role": "system",
                        "content": (
                            "Evaluation Method:\n"
                            "- skill-level-0: no evidence of skill\n"
                            "- skill-level-1: mentioned once, low certainty\n"
                            "- skill-level-2: mentioned in projects or context, medium proficiency\n"
                            "- skill-level-3: listed as core skill with usage\n"
                            "- skill-level-4: deep expertise with multiple confirmations\n"
                            "- Confidence reflects the *reliability* of this classification\n"
                        )
                    },
                    {
                        "role": "system",
                        "content": "DO NOT include any markdown formatting.\n"
                                   " DO NOT include explanations. Return ONLY valid JSON.\n"
                    },
                    {
                        "role": "user",
                        "content": (
                            f"user_professional_information:\n{json.dumps(user_info)}\n"
                            f"job_posting_information:\n{json.dumps(job_info)}"
                        )
                    }
                ]
            )

            response_clean = response.strip().removeprefix("```json").removesuffix("```").strip()
            parsed = json.loads(response_clean)

            if isinstance(parsed, list):
                comparison_table = parsed
            elif isinstance(parsed, dict) and "comparison_table" in parsed:
                comparison_table = parsed["comparison_table"]
            else:
                raise ValueError("Unexpected LLM response format.")

            # Save original
            request.session['comparison_table'] = comparison_table
            # Save to JSON file
            output_path = os.path.join(settings.MEDIA_ROOT, 'json', 'comparison_table.json')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"comparison_table": comparison_table}, f, ensure_ascii=False, indent=2)


        # Recalculate match %
        match_percentage = calculate_keyword_match_percentage({"comparison_table": comparison_table})

    except Exception as e:
        error = f"Fehler beim Laden oder Berechnen: {e}"

    return render(request, 'step4.html', {
        'comparison_table': comparison_table,
        'match_percentage': match_percentage,
        'error': error,
        'step': 4
    })

def save_comparison_edits(request):
    if request.method == 'POST':
        total_rows = int(request.POST.get('total_rows', 0))
        updated_table = []

        for i in range(total_rows):
            job_req = request.POST.get(f'job_req_{i}')
            matched_skill = request.POST.get(f'matched_skill_{i}')

            updated_table.append({
                "job_requirement": job_req,
                "matched_user_skill": matched_skill,
                # Optionally re-add other fields if needed from session
            })

        # Update session
        if 'comparison_table' in request.session:
            # Preserve existing structure where possible
            existing_table = request.session['comparison_table']
            for i, row in enumerate(existing_table):
                if i < total_rows:
                    row['matched_user_skill'] = updated_table[i]['matched_user_skill']
            request.session['comparison_table'] = existing_table
            request.session.modified = True

        return redirect('step5')  # Continue to next step

    return redirect('step4')

def compile_latex(latex_content: str, output_path: str):
    response = requests.get(
        "https://latexonline.cc/compile",
        params={"text": latex_content, "compiler": "pdflatex"}
    )

    if response.status_code != 200 or not response.content.startswith(b'%PDF'):
        print("API response (truncated):", response.text[:1000])
        return False, response.text[:3000]  # return the error
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)
        return True, None

def rerun_comparison(request):
    if 'comparison_table' in request.session:
        del request.session['comparison_table']
    if 'match_percentage' in request.session:
        del request.session['match_percentage']
    return redirect('step4')  # redirect to the same step that triggers LLM again

@csrf_exempt
def autosave_skill_edit(request):
    if request.method == 'POST':
        job_req = request.POST.get('job_requirement')
        matched_skill = request.POST.get('matched_skill')

        comparison_table = request.session.get('comparison_table', [])
        for row in comparison_table:
            if row['job_requirement'] == job_req:
                row['matched_user_skill'] = matched_skill
        request.session['comparison_table'] = comparison_table
        request.session.modified = True
        return JsonResponse({'status': 'success'})

    return JsonResponse({'status': 'invalid'}, status=400)


@csrf_exempt
def add_suggested_skill(request):
    if request.method == 'POST':
        job_req = request.POST.get('job_requirement')

        comparison_table = request.session.get('comparison_table', [])
        for row in comparison_table:
            if row['job_requirement'] == job_req:
                if row['matched_user_skill'] == "None":
                    # Add suggested skill
                    row['matched_user_skill'] = f"{job_req} (Partial)"
                    row['confidence'] = "Medium"
                    row['explanation_llm'] += " ✅ Based on your profile, we assumed you likely know this skill."
                    row['highlight'] = "green"

        # Save updated session
        request.session['comparison_table'] = comparison_table
        # Save to JSON
        output_path = os.path.join(settings.MEDIA_ROOT, 'json', 'comparison_table.json')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({"comparison_table": comparison_table}, f, ensure_ascii=False, indent=2)


    return redirect('step4')


@csrf_exempt
def ignore_suggested_skill(request):
    if request.method == 'POST':
        job_req = request.POST.get('job_requirement')

        comparison_table = request.session.get('comparison_table', [])
        for row in comparison_table:
            if row['job_requirement'] == job_req:
                if row['matched_user_skill'] == "None":
                    row['explanation_llm'] += " ❌ You chose to ignore this suggested skill."
                    row['highlight'] = "red"

        request.session['comparison_table'] = comparison_table

    return redirect('step4')


# ====== STEP 4 END ===========
def step5(request):
    error = None
    pdf_filename = None

    try:
        # === Paths ===
        personal_info_path = os.path.join(settings.MEDIA_ROOT, 'json', 'personal_info.json')
        job_info_path = os.path.join(settings.MEDIA_ROOT, 'json', 'job_posting_information.json')
        professional_info_path = os.path.join(settings.MEDIA_ROOT, 'json', 'professional_info.json')
        comparison_table_path = os.path.join(settings.MEDIA_ROOT, 'json', 'comparison_table.json')

        # === Load data ===
        with open(personal_info_path, 'r', encoding='utf-8') as f:
            personal_info = json.load(f)

        with open(job_info_path, 'r', encoding='utf-8') as f:
            job_info = json.load(f)

        with open(professional_info_path, 'r', encoding='utf-8') as f:
            professional_info = json.load(f)

        with open(comparison_table_path, 'r', encoding='utf-8') as f:
            comparison_table = json.load(f)


        # === Call Gemini ===
        client = UnifiedChatClient()
        response = client.chat([
            {
            "role": "system",
            "content": (
                "You are an advanced AI resume generator acting as a professional career consultant and CV writer. "
                "Your job is to transform structured resume data into a polished, compelling, full-page LaTeX resume, following the Harvard CV layout and modern placement standards.\n\n"

                "=== OBJECTIVE ===\n"
                "Generate a professional LaTeX resume tailored for human recruiters. "
                "It should be achievement-oriented, clearly formatted, and use appropriate language and structure. The layout must resemble a Harvard-style resume.\n"
                "The CV should be tailored perfectly to the job posting, highlighting relevant skills and experiences. For this, put more emphasis on the provided comparison_table \n"

                "=== CONTEXT ===\n"
                "You will receive the following structured inputs:\n"
                "1. user_personal_information: Contact details (name, email, phone, address).\n"
                "2. job_posting_information: Structured job description and expectations.\n"
                "3. comparison_table: Skill-level matches between user and job.\n"
                "4. professional_info: Full resume data (education, experience, leadership, skills, etc.).\n\n"

                "=== WORKFLOW ==="
                "\n"
                "1. Replace all placeholders: <<FULL_NAME>>, <<ADDRESS>>, <<EMAIL>>, <<PHONE>>, <<EDUCATION>>, <<EXPERIENCE>>, <<LEADERSHIP>>, <<SKILLS>>.\n"
                "2. For each bullet point:\n"
                "   - Begin with an approved action verb.\n"
                "   - Highlight outcomes, skills, and quantifiable results.\n"
                "   - Format as phrases, not full sentences (no periods, no personal pronouns).\n"
                "3. Include activities and recognition outside of core job experience when meaningful.\n"
                "4. Include full-page content with minimal whitespace.\n"
                "5. Organize skills into technical, languages, interests, tools, and certifications if applicable.\n\n"

                "=== ACTION VERBS ===\n"
                "You must begin each bullet point with one of the following professionally accepted action verbs:\n"
                "Accelerated, Accomplished, Achieved, Acted, Adapted, Added, Addressed, Administered, Advised, Allocated, "
                "Analyzed, Appraised, Approved, Arbitrated, Arranged, Assembled, Assessed, Assigned, Assisted, Attained, "
                "Audited, Authored, Balanced, Broadened, Budgeted, Built, Calculated, Cataloged, Centralized, Chaired, "
                "Changed, Clarified, Classified, Coached, Collaborated, Collected, Communicated, Compiled, Completed, "
                "Composed, Computed, Conceived, Conceptualized, Concluded, Conducted, Consolidated, Constructed, Contracted, "
                "Controlled, Convinced, Coordinated, Corresponded, Counseled, Created, Critiqued, Customized, Defined, "
                "Delegated, Delivered, Demonstrated, Demystified, Derived, Designed, Determined, Developed, Devised, Diagnosed, "
                "Directed, Discovered, Dispatched, Documented, Drafted, Earned, Edited, Educated, Enabled, Encouraged, Energized, "
                "Engineered, Enhanced, Enlisted, Established, Evaluated, Examined, Executed, Expanded, Expedited, Explained, "
                "Extracted, Fabricated, Facilitated, Familiarized, Fashioned, Forecasted, Formed, Formulated, Founded, Gained, "
                "Gathered, Generated, Guided, Handled, Headed, Identified, Illustrated, Impacted, Implemented, Improved, Increased, "
                "Influenced, Informed, Initiated, Inspected, Installed, Instituted, Instructed, Integrated, Interpreted, Interviewed, "
                "Introduced, Invented, Investigated, Launched, Lectured, Led, Liaised, Maintained, Managed, Marketed, Mastered, "
                "Maximized, Mediated, Minimized, Modeled, Moderated, Monitored, Motivated, Negotiated, Operated, Optimized, "
                "Orchestrated, Organized, Originated, Overhauled, Oversaw, Participated, Performed, Persuaded, Planned, Predicted, "
                "Prepared, Presented, Prioritized, Processed, Produced, Programmed, Projected, Promoted, Proposed, Proved, Provided, "
                "Publicized, Published, Purchased, Recommended, Reconciled, Recorded, Recruited, Redesigned, Reduced, Referred, "
                "Regulated, Rehabilitated, Reinforced, Remodeled, Reorganized, Repaired, Reported, Represented, Researched, "
                "Resolved, Retrieved, Reviewed, Revised, Revitalized, Rewrote, Scheduled, Screened, Selected, Served, Shaped, "
                "Simplified, Sold, Solved, Spearheaded, Specified, Spoke, Standardized, Steered, Stimulated, Streamlined, "
                "Strengthened, Structured, Studied, Suggested, Summarized, Supervised, Supported, Surpassed, Surveyed, "
                "Synthesized, Systematized, Tabulated, Taught, Tested, Trained, Translated, Unified, Updated, Upgraded, "
                "Utilized, Validated, Verified, Visualized, Wrote\n\n"
                "Only use verbs from this list. Avoid repetition of the same verb in a section.\n\n"

                "=== OUTPUT FORMAT ===\n"
                "- Return a single valid LaTeX document — no markdown, no explanations, No illegal LaTeX commands like '\n' '&' \n"
                "- DO NOT leave any placeholder (e.g. <<FULL_NAME>>).\n"
                "- Make sure the LaTeX document always fills at least 3/4 of the page, but never exceeds one page.\n"
                "- Do not use the following character in the text: '&'. Use 'and' instead, as using '&' will lead to an error.\n"
                "- Even in Names like 'R&D', do not use '&' in the output format text. Instead opt for workarounds like 'RnD' or 'and'.\n"
                "- Use this LaTeX template structure:\n\n"

                "\\documentclass[11pt]{article}\n"
                "\\usepackage[margin=1in]{geometry}\n"
                "\\usepackage{enumitem}\n"
                "\\usepackage{titlesec}\n"
                "\\usepackage{hyperref}\n"
                "\\usepackage{parskip}\n"
                "\\titleformat{\\section}{\\bfseries\\large}{}{0em}{}[\\titlerule]\n"
                "\\begin{document}\n\n"
                "\\begin{center}\n"
                "    {\\LARGE \\textbf{<<FULL_NAME>>}} \\\\\n"
                "    <<ADDRESS>> - <<EMAIL>> - <<PHONE>>\n"
                "\\end{center}\n\n"
                "\\section*{Education}\n"
                "<<EDUCATION>>\n\n"
                "\\section*{Experience}\n"
                "<<EXPERIENCE>>\n\n"
                "\\section*{Leadership \\& Activities}\n"
                "<<LEADERSHIP>>\n\n"
                "\\section*{Skills}\n"
                "<<SKILLS>>\n\n"
                "\\end{document}\n\n"

                "Now return only the final LaTeX."
            )
            },
            {
                "role": "user",
                "content": (
                    f"user_personal_information:\n{json.dumps(personal_info, indent=2)}\n\n"
                    f"job_posting_information:\n{json.dumps(job_info, indent=2)}\n\n"
                    f"comparison_table:\n{json.dumps(comparison_table, indent=2)}\n\n"
                    f"professional_info:\n{json.dumps(professional_info, indent=2)}"
                )
            }
        ])


        # === Split Gemini Response ===
        latex_output = response.strip()

        # === Generate PDF ===
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"cv_{timestamp}.pdf"
        output_path = os.path.join(settings.MEDIA_ROOT, "generated_cvs", pdf_filename)
        print(latex_output)
        compile_latex(latex_output, output_path)

        success, error_output = compile_latex(latex_output, output_path)

        # === If compilation fails, repair via Gemini ===
        if not success and error_output:
            print("Initial compilation failed, attempting to repair LaTeX...")
            repair_prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are a LaTeX repair assistant. The user will give you broken LaTeX and compiler error messages. "
                        "Fix only the syntax and structural errors without changing formatting, content, or layout. "
                        "Return ONLY valid LaTeX code. No markdown. No explanations."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"The following LaTeX code failed to compile:\n\n{latex_output}\n\n"
                        f"Here is the compiler error message:\n\n{error_output}"
                    )
                }
            ]
            fixed_latex = client.chat(repair_prompt).strip()

            success, second_error = compile_latex(fixed_latex, output_path)

            if not success:
                raise RuntimeError(f"Second compilation failed:\n{second_error}")
            else:
                latex_output = fixed_latex  # Save the corrected version

    except Exception as e:
        error = f"Error while creating CV: {e}"

    return render(request, 'step5.html', {
        'pdf_filename': pdf_filename,
        'MEDIA_URL': settings.MEDIA_URL,
        'error': error,
        'step': 5
    })

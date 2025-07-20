from django import forms


class DateiUploadForm(forms.Form):
    """
    Einfaches Upload-Formular für eine Datei.
    """
    datei = forms.FileField(
        label="Choose a file (PDF, DOCX, TXT)"
    )


class JobPostingForm(forms.Form):
    job_url = forms.URLField(
        required=False,
        label="Job URL",
        widget=forms.URLInput(attrs={
            'class': 'form-control',
            'placeholder': 'https://www.linkedin.com/jobs/....',
            'lang': 'en'
        })
    )

    job_pdf = forms.FileField(
        required=False,
        label="...or upload a PDF",
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'lang': 'en',
        })
   )
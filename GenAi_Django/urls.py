"""
URL configuration for GenAi_Django project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from CV import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),
    path('step1/', views.step1, name='step1'),
    path('step2/', views.step2, name='step2'),
    path('step2b/', views.step2b_company_values_view, name='step2b'),
    path('step3/', views.step3, name='step3'),
    path('step3a/education/', views.step3a_edu, name='step3a_edu'),
    path('step3a/work/', views.step3a_work, name='step3a_work'),
    path('step3a/other/', views.step3a_other, name='step3a_other'),
    path('step3b/', views.step3b, name='step3b'),
    path('step4/', views.step4, name='step4'),
    path('step4/rerun/', views.rerun_comparison, name='rerun_comparison'),

    path('autosave-skill/', views.autosave_skill_edit, name='autosave_skill_edit'),

    path('save-comparison-edits/', views.save_comparison_edits, name='save_comparison_edits'),

    path('add-skill/', views.add_suggested_skill, name='add_suggested_skill'),
    path('ignore-skill/', views.ignore_suggested_skill, name='ignore_suggested_skill'),
    path('step5/', views.step5, name='step5'),
]

# Serve media files during development
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
from django.urls import path
from search import views

urlpatterns = [
    path('', views.search, name='search'),
    path('remove-history/', views.remove_history, name='remove_history'),
    path('clear-history/', views.clear_history, name='clear_history'),
    path('autocomplete/', views.autocomplete, name='autocomplete'),
]

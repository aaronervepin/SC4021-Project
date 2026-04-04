from django.apps import AppConfig


class SearchConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'search'

    def ready(self):
        try:
            from search.tfidf_engine import get_engine
            get_engine()
            print("[TF-IDF] Engine pre-built successfully.")
        except Exception as e:
            print(f"[TF-IDF] Failed to pre-build engine: {e}")

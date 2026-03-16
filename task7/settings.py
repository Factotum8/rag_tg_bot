from pydantic_settings import BaseSettings

class AppSettings(
    BaseSettings,
):
    openai_api_key: str = "token"
    telegram_token: str = "token"

    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    data_path: str = "../task7/bad_knowledge_base/star_wars_planets_dataset.json"
    chroma_dir: str = "../task7/chroma_starwars"
    collection_name: str = "starwars_planets"

    class Config:
        env_file = "../.env"
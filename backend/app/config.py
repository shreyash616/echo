from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    # AcoustID (audio fingerprinting) — https://acoustid.org/login
    acoustid_api_key: str = ""

    # Spotify Developer App (for 1-click playlist creation) — https://developer.spotify.com/dashboard
    spotify_client_id: str = ""
    spotify_redirect_uri: str = ""

    # ML model paths
    faiss_index_path: str = "../ml/inference/music_index.faiss"
    track_metadata_path: str = "../ml/inference/track_metadata.json"

    # Limits & Candidate Pool (S1)
    max_recommendations: int = 20
    max_audio_size_mb: int = 30
    candidates_clap: int = 300
    candidates_feel: int = 3000


settings = Settings()

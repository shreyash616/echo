from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    # AcoustID (audio fingerprinting) — https://acoustid.org/login
    acoustid_api_key: str = ""

    # Last.fm API — https://www.last.fm/api/account/create (free)
    lastfm_api_key: str = ""

    # ML model paths
    onnx_model_path: str = "../ml/inference/music_encoder.onnx"
    faiss_index_path: str = "../ml/inference/music_index.faiss"
    track_metadata_path: str = "../ml/inference/track_metadata.json"

    # Limits
    max_recommendations: int = 20
    max_audio_size_mb: int = 30


settings = Settings()

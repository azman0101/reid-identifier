from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    frigate_url: str = "http://localhost:5000"
    external_url: str = "http://localhost:5000"
    model_path: str = "/models/person-reidentification-retail-0288.xml"
    gallery_dir: str = "/models/gallery"
    log_level: str = "INFO"
    unknown_dir: str = "/models/unknown"
    device_name: str = "GPU"

    # New settings for Reranking and Self-Learning
    use_rerank: bool = True
    rerank_k: int = 5
    self_learning_threshold: float = 0.80
    max_gallery_per_identity: int = 15
    use_tta: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

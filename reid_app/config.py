from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    frigate_url: str = "http://localhost:5000"
    external_url: str = "http://localhost:5000"
    model_path: str = "/models/person-reidentification-retail-0288.xml"
    gallery_dir: str = "/models/gallery"
    unknown_dir: str = "/models/unknown"
    device_name: str = "GPU"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import dotenv_values
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    def dotenv_values(path, encoding='utf-8-sig'):
        values = {}
        env_path = Path(path)
        if not env_path.exists():
            return values
        for raw_line in env_path.read_text(encoding=encoding).splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            if value and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            values[key] = value
        return values

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
IN_DATA_DIR = DATA_DIR / "in_data"
REFERENCE_DIR = DATA_DIR / "reference"
MATCHING_DIR = DATA_DIR / "matching"
ENV_PATH = ROOT_DIR / ".env"


@dataclass(frozen=True)
class ServerSettings:
    host: str
    port: int

    @property
    def browser_url(self):
        host = '127.0.0.1' if self.host in {'0.0.0.0', '::'} else self.host
        if ':' in host and not host.startswith('['):
            host = f'[{host}]'
        return f'http://{host}:{self.port}'


def load_env_file(path: str | Path | None = None) -> dict[str, str]:
    env_path = Path(path) if path is not None else ENV_PATH
    values: dict[str, str] = {}
    if not env_path.exists():
        return values

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        values[key] = value

    return values


def get_env_value(key: str, default: str | None = None) -> str | None:
    values = load_env_file()
    return os.getenv(key, values.get(key, default))


def server_settings(env_path: str | Path | None = None, environ: dict | None = None):
    values = {**dotenv_values(Path(env_path) if env_path is not None else ENV_PATH, encoding='utf-8-sig'), **(os.environ if environ is None else environ)}
    host = str(values.get('OCR_WEB_HOST') or '0.0.0.0').strip()
    if not host or any(character.isspace() for character in host) or '://' in host or '/' in host:
        raise ValueError('OCR_WEB_HOST: укажите IP-адрес или имя хоста без http:// и пути')
    try:
        port = int(values.get('OCR_WEB_PORT') or '8017')
    except (ValueError, TypeError) as error:
        raise ValueError('OCR_WEB_PORT: требуется целое число от 1 до 65535') from error
    if not 1 <= port <= 65535:
        raise ValueError('OCR_WEB_PORT: требуется целое число от 1 до 65535')
    return ServerSettings(host, port)


settings = server_settings()
WEB_HOST = settings.host
WEB_PORT = settings.port

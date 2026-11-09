"""Install the pinned official tokenizer data, without executing downloaded code."""
import io
import json
import zipfile
from pathlib import Path

import httpx

from src.agent.data import file_hash

URL = 'https://cdn.deepseek.com/api-docs/deepseek_v4_tokenizer.zip'
SHA256 = '89085f12ef79460ac5f66d1119325ddfc694b4ab209d80bbd81d35f081dc9614'
TARGET = Path(__file__).resolve().parents[1]/'models/agent_tokenizer/tokenizer.json'


def main():
    if TARGET.exists():
        if file_hash(TARGET) != SHA256:
            raise ValueError('Existing tokenizer differs from the pinned official file.')
    else:
        with httpx.Client(timeout=45) as client:
            response = client.get(URL)
            response.raise_for_status()
        if len(response.content) > 15_000_000:
            raise ValueError('Tokenizer archive exceeds the size limit.')
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            info = archive.getinfo('deepseek_v4_tokenizer/tokenizer.json')
            if info.file_size > 10_000_000:
                raise ValueError('Tokenizer exceeds the size limit.')
            data = archive.read(info)
        import hashlib
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise ValueError('Official tokenizer hash changed; review before updating the pin.')
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        with TARGET.open('xb') as stream:
            stream.write(data)
    print(json.dumps({'tokenizer':str(TARGET),'sha256':SHA256,'source':URL}))

if __name__ == '__main__':
    main()

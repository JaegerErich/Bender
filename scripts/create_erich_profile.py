"""
One-off: Create profile for Erich Jaeger with password "Bender".
Uses same hashing as ui_streamlit so login works.
"""
import hashlib
import json
import secrets
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROFILE_DIR = BASE_DIR / "data" / "profiles"
PBKDF2_ITERATIONS = 100_000


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_bytes(16)
    else:
        salt = salt if isinstance(salt, bytes) else bytes.fromhex(salt)
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return salt.hex(), key.hex()


def main():
    user_id = "erich_jaeger"
    display_name = "Erich Jaeger"
    password = "Bender"

    salt_hex, hash_hex = hash_password(password)
    profile = {
        "user_id": user_id,
        "display_name": display_name,
        "age": 16,
        "equipment": ["None"],
        "equipment_setup_done": True,
        "password_salt": salt_hex,
        "password_hash": hash_hex,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    path = PROFILE_DIR / f"{user_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    print(f"Created {path}")
    print("Username: Erich Jaeger")
    print("Password: Bender")


if __name__ == "__main__":
    main()

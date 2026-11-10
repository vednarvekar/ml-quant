"""Upstox OAuth2 (authorization-code) helper.

Token expires 03:30 IST next day, so run `uv run python -m mlq.upstox.auth` each morning.
Secrets come from .env (never committed); token saved to .secrets/.
"""
import json
import os
import sys
from datetime import datetime, timezone
from urllib.parse import urlencode, urlparse, parse_qs

import httpx
from dotenv import load_dotenv

from mlq.config import ROOT, SECRETS, TOKEN_FILE

AUTH_URL = "https://api.upstox.com/v2/login/authorization/dialog"
TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"


def _env():
    load_dotenv(ROOT / ".env")
    try:
        return (os.environ["UPSTOX_API_KEY"], os.environ["UPSTOX_API_SECRET"],
                os.environ["UPSTOX_REDIRECT_URI"])
    except KeyError as e:
        sys.exit(f"Missing {e} in .env (see .env.example)")


def login_url() -> str:
    key, _, redirect = _env()
    return f"{AUTH_URL}?" + urlencode(
        {"client_id": key, "redirect_uri": redirect, "response_type": "code"})


def exchange_code(code: str) -> dict:
    key, secret, redirect = _env()
    r = httpx.post(
        TOKEN_URL,
        headers={"accept": "application/json"},
        data={"code": code, "client_id": key, "client_secret": secret,
              "redirect_uri": redirect, "grant_type": "authorization_code"},
        timeout=30,
    )
    r.raise_for_status()
    tok = r.json()
    tok["saved_at"] = datetime.now(timezone.utc).isoformat()
    SECRETS.mkdir(exist_ok=True)
    TOKEN_FILE.write_text(json.dumps(tok))
    return tok


def load_token() -> str:
    """Prefer the 1-year read-only Analytics token (UPSTOX_ANALYTICS_TOKEN in .env);
    fall back to the daily OAuth token (needed only for trading APIs)."""
    load_dotenv(ROOT / ".env")
    if os.environ.get("UPSTOX_ANALYTICS_TOKEN"):
        return os.environ["UPSTOX_ANALYTICS_TOKEN"]
    if not TOKEN_FILE.exists():
        sys.exit("No token. Run: uv run python -m mlq.upstox.auth")
    return json.loads(TOKEN_FILE.read_text())["access_token"]


def main():
    print("1) Open this URL, log in, approve:\n\n", login_url(), "\n")
    pasted = input("2) Paste the full redirected URL (or just the code): ").strip()
    code = parse_qs(urlparse(pasted).query).get("code", [pasted])[0]
    exchange_code(code)
    print("Token saved to .secrets/ (valid until ~03:30 IST tomorrow).")


if __name__ == "__main__":
    main()

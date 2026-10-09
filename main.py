from functools import wraps
from datetime import datetime
import hmac
import ipaddress
import os
from pathlib import Path
import re
import secrets
from threading import Lock
import time

from argofamiglia import ArgoFamiglia
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory, session
from google import genai
from google.genai.errors import APIError
from google.genai import types
import httpx
import requests


PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")
MAX_REQUEST_BYTES = 262_144
API_CONFIG = {
    "google": {
        "apiKey": os.environ.get("GEMINI_API_KEY", ""),
        "models": {
            "gemini-2.5": "gemini-2.5-flash",
            "gemini-2.5-chat": "gemini-2.5-flash",
        },
    }
}
SECRET_PLACEHOLDER = "replace-with-a-long-random-secret"
LOGIN_ATTEMPTS = {}
LOGIN_ATTEMPTS_LOCK = Lock()
LOGIN_ATTEMPT_LIMIT = 5
LOGIN_COOLDOWN_SECONDS = 900
LOCAL_AUTH_BYPASS = os.environ.get("LOCAL_AUTH_BYPASS", "").lower() == "true"

app = Flask(__name__, static_folder=None)
app.config.update(
    SECRET_KEY=os.environ.get("SESSION_SECRET"),
    MAX_CONTENT_LENGTH=MAX_REQUEST_BYTES,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "false").lower() == "true",
    PERMANENT_SESSION_LIFETIME=60 * 60 * 24 * 14,
)


@app.after_request
def add_response_headers(response):
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


def api_error(message: str, status: int):
    return jsonify({"error": message}), status


def csrf_is_valid() -> bool:
    submitted_token = request.headers.get("X-CSRF-Token", "")
    stored_token = session.get("csrf_token", "")
    return bool(submitted_token and stored_token and hmac.compare_digest(submitted_token, stored_token))


def is_security_configured() -> bool:
    secret_key = app.config["SECRET_KEY"]
    app_password = os.environ.get("APP_PASSWORD", "")
    return bool(secret_key and len(secret_key) >= 32 and secret_key != SECRET_PLACEHOLDER and len(app_password) >= 12)


def is_gemini_configured() -> bool:
    api_key = API_CONFIG["google"]["apiKey"].strip()
    return bool(api_key and api_key != "paste-your-new-gemini-api-key-here")


def is_local_auth_bypass_request() -> bool:
    remote_addr = request.remote_addr
    return bool(
        LOCAL_AUTH_BYPASS
        and remote_addr
        and ipaddress.ip_address(remote_addr).is_loopback
    )


def require_authentication(handler):
    @wraps(handler)
    def wrapped(*args, **kwargs):
        if is_local_auth_bypass_request():
            return handler(*args, **kwargs)
        if not session.get("authenticated"):
            return api_error("Accedi per continuare.", 401)
        if not csrf_is_valid():
            return api_error("Sessione scaduta. Ricarica la pagina e accedi di nuovo.", 403)
        return handler(*args, **kwargs)

    return wrapped


@app.get("/")
def index():
    return send_from_directory(PROJECT_DIR, "index.html")


@app.get("/manifest.webmanifest")
def manifest():
    return send_from_directory(PROJECT_DIR, "manifest.webmanifest", mimetype="application/manifest+json")


@app.get("/sw.js")
def service_worker():
    response = send_from_directory(PROJECT_DIR, "sw.js", mimetype="application/javascript")
    response.headers["Service-Worker-Allowed"] = "/"
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/icons/<path:filename>")
def icons(filename: str):
    return send_from_directory(PROJECT_DIR / "icons", filename)


@app.get("/healthz")
def health_check():
    return jsonify({"status": "ok"})


@app.get("/api/config")
def get_config():
    local_auth_bypass = is_local_auth_bypass_request()
    has_server_secrets = is_security_configured()
    csrf_token = ""
    if has_server_secrets:
        csrf_token = session.setdefault("csrf_token", secrets.token_urlsafe(32))
    return jsonify({
        "authenticated": local_auth_bypass or bool(session.get("authenticated")),
        "localAuthBypass": local_auth_bypass,
        "configured": is_gemini_configured(),
        "setupRequired": not local_auth_bypass and not has_server_secrets,
        "csrfToken": csrf_token,
    })


@app.post("/api/login")
def login():
    expected_password = os.environ.get("APP_PASSWORD", "")
    if not is_security_configured():
        return api_error("Configura SESSION_SECRET e una APP_PASSWORD di almeno 12 caratteri nelle variabili segrete dell'hosting.", 503)
    if not csrf_is_valid():
        return api_error("Ricarica la pagina e riprova.", 403)

    client_key = request.remote_addr or "unknown"
    now = time.monotonic()
    with LOGIN_ATTEMPTS_LOCK:
        attempts, blocked_until = LOGIN_ATTEMPTS.get(client_key, (0, 0))
        if now < blocked_until:
            return api_error("Troppi tentativi. Attendi qualche minuto e riprova.", 429)
        if blocked_until:
            LOGIN_ATTEMPTS.pop(client_key, None)

    payload = request.get_json(silent=True)
    password = payload.get("password") if isinstance(payload, dict) else None
    if not isinstance(password, str) or not hmac.compare_digest(password, expected_password):
        with LOGIN_ATTEMPTS_LOCK:
            attempts, _ = LOGIN_ATTEMPTS.get(client_key, (0, 0))
            attempts += 1
            blocked_until = now + LOGIN_COOLDOWN_SECONDS if attempts >= LOGIN_ATTEMPT_LIMIT else 0
            LOGIN_ATTEMPTS[client_key] = (attempts, blocked_until)
        return api_error("Password non corretta.", 401)

    with LOGIN_ATTEMPTS_LOCK:
        LOGIN_ATTEMPTS.pop(client_key, None)
    session.clear()
    session["authenticated"] = True
    session.permanent = True
    csrf_token = secrets.token_urlsafe(32)
    session["csrf_token"] = csrf_token
    return jsonify({
        "authenticated": True,
        "configured": is_gemini_configured(),
        "csrfToken": csrf_token,
    })


@app.post("/api/logout")
@require_authentication
def logout():
    session.clear()
    return jsonify({"authenticated": False})


@app.post("/api/gemini")
@require_authentication
def generate_content():
    api_key = API_CONFIG["google"]["apiKey"]
    if not is_gemini_configured():
        return api_error("Gemini non è configurato: aggiungi GEMINI_API_KEY nelle variabili segrete dell'hosting.", 503)

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return api_error("Formato della richiesta non valido.", 400)

    model = payload.get("model")
    system_instruction = payload.get("systemInstruction")
    contents = payload.get("contents")
    if not isinstance(model, str) or model not in API_CONFIG["google"]["models"]:
        return api_error("Il modello Gemini richiesto non è supportato.", 400)
    if not isinstance(system_instruction, str) or not isinstance(contents, list) or not contents:
        return api_error("La richiesta non contiene un messaggio valido.", 400)

    try:
        sdk_contents = [types.Content.model_validate(item) for item in contents]
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,
            max_output_tokens=2200,
        )
    except (ValueError, TypeError) as error:
        return api_error(str(error), 400)

    try:
        with genai.Client(api_key=api_key) as client:
            response = client.models.generate_content(
                model=API_CONFIG["google"]["models"][model],
                contents=sdk_contents,
                config=config,
            )
        answer = (response.text or "").strip()
        if not answer:
            return api_error("Gemini ha restituito una risposta vuota. Riprova.", 502)
        return jsonify({"text": answer})
    except (APIError, httpx.HTTPError) as error:
        message = str(error).replace(api_key, "[chiave nascosta]")
        app.logger.warning("Richiesta Gemini non riuscita: %s", message)
        return api_error(message or "La richiesta a Gemini non è riuscita.", 502)


def argo_date(value: object) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if not value:
        return ""
    try:
        return datetime.fromisoformat(value[:10]).date().isoformat()
    except ValueError:
        try:
            return datetime.strptime(value, "%d/%m/%Y").date().isoformat()
        except ValueError:
            return ""


@app.post("/api/argo/sync")
@require_authentication
def sync_argo():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return api_error("Formato della richiesta non valido.", 400)

    credentials = {
        "school": payload.get("schoolCode"),
        "username": payload.get("username"),
        "password": payload.get("password"),
    }
    if any(not isinstance(value, str) or not value.strip() for value in credentials.values()):
        return api_error("Inserisci codice scuola, username e password Argo.", 400)
    if len(credentials["school"]) > 40 or len(credentials["username"]) > 256 or len(credentials["password"]) > 1024:
        return api_error("Uno dei dati inseriti supera la lunghezza consentita.", 400)

    try:
        argo = ArgoFamiglia(
            credentials["school"].strip(),
            credentials["username"].strip(),
            credentials["password"],
        )
        dashboard = argo.dashboard()
    except ValueError as error:
        if str(error) == "Wrong credentials":
            return api_error("Codice scuola, username o password Argo non corretti.", 401)
        app.logger.warning("Connessione Argo non riuscita: %s", type(error).__name__)
        return api_error("Argo non ha restituito una risposta utilizzabile. Riprova più tardi.", 502)
    except requests.RequestException as error:
        app.logger.warning("Connessione Argo non riuscita: %s", type(error).__name__)
        return api_error("Non è stato possibile raggiungere Argo. Controlla la connessione e riprova.", 502)
    except (KeyError, TypeError) as error:
        app.logger.warning("Risposta Argo non valida: %s", type(error).__name__)
        return api_error("La risposta di Argo non è nel formato atteso. La libreria potrebbe non essere compatibile.", 502)

    data = dashboard.get("data") if isinstance(dashboard, dict) else None
    records = data.get("dati") if isinstance(data, dict) else None
    if not isinstance(records, list):
        app.logger.warning("Risposta Argo priva dell'elenco dati della dashboard.")
        return api_error("Argo ha risposto senza i dati del registro.", 502)

    assignments = []
    reminders = []
    for record in records:
        if not isinstance(record, dict):
            continue
        register = record.get("registro", [])
        if isinstance(register, list):
            for subject_entry in register:
                if not isinstance(subject_entry, dict):
                    continue
                subject = subject_entry.get("materia")
                homework = subject_entry.get("compiti", [])
                if not isinstance(homework, list):
                    continue
                for item in homework:
                    if not isinstance(item, dict):
                        continue
                    title = item.get("compito")
                    due = argo_date(item.get("dataConsegna"))
                    if isinstance(title, str) and title.strip() and due:
                        assignments.append({
                            "subject": str(subject or "Materia"),
                            "title": title.strip(),
                            "due": due,
                        })

        class_reminders = record.get("promemoria", [])
        if isinstance(class_reminders, list):
            for item in class_reminders:
                if not isinstance(item, dict):
                    continue
                due = argo_date(item.get("datGiorno") or item.get("datEvento"))
                note = item.get("desAnnotazioni") or item.get("desPromemoria") or item.get("desTitolo")
                if not due or not isinstance(note, str) or not note.strip():
                    continue
                start_time = item.get("oraInizio", "")
                reminders.append({
                    "date": due,
                    "time": start_time[:5] if isinstance(start_time, str) and re.match(r"^\d{2}:\d{2}", start_time) else "",
                    "title": note.strip(),
                    "detail": str(item.get("docente") or "").strip(),
                })

    return jsonify({"assignments": assignments, "reminders": reminders})


@app.errorhandler(413)
def request_too_large(_error):
    app.logger.warning("Rejected an oversized request (HTTP %s).", _error.code)
    return api_error("La richiesta è troppo grande.", 413)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "18765"))
    print("Diario scolastico avviato.")
    print(f"Indirizzo locale: http://127.0.0.1:{port}")
    print("Per l'accesso Internet da telefono e PC, pubblica il progetto su un hosting HTTPS.")
    app.run(host="127.0.0.1", port=port)

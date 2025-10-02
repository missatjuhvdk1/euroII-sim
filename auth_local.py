import os
import re
import time
import secrets
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
import smtplib
 

import bcrypt
import extra_streamlit_components as stx
import streamlit as st
from dotenv import load_dotenv
from jose import jwt, JWTError
from filelock import FileLock
from streamlit.components.v1 import html as st_html

# =========================
# Configuration
# =========================
load_dotenv(override=True)

APP_NAME = os.getenv("APP_NAME", "Rekenmodule")

# Base URL used in links sent by email. Required if your app is not hosted at root.
# Example: https://app.example.com
APP_BASE_URL = os.getenv("APP_BASE_URL", "https://runs-eden-beans-ebook.trycloudflare.com/")

def _get_or_create_cookie_secret() -> str:
    # Prefer env var if provided
    env_secret = os.getenv("COOKIE_SECRET")
    if env_secret:
        return env_secret
    # Persist a generated secret to a file so refreshes keep tokens valid
    secret_file = os.getenv(
        "COOKIE_SECRET_FILE",
        os.path.join(os.getcwd(), "dist", "auth_cookie_secret.key"),
    )
    try:
        os.makedirs(os.path.dirname(secret_file) or ".", exist_ok=True)
        if os.path.exists(secret_file):
            with open(secret_file, "r", encoding="utf-8") as f:
                s = f.read().strip()
                if s:
                    return s
    except Exception:
        pass
    # Generate and store
    s = secrets.token_urlsafe(64)
    try:
        with open(secret_file, "w", encoding="utf-8") as f:
            f.write(s)
    except Exception:
        pass
    return s

COOKIE_SECRET = _get_or_create_cookie_secret()
USERS_JSON_PATH = os.getenv("USERS_JSON_PATH", os.path.join(os.getcwd(), "users.json"))
USERS_LOCK_PATH = USERS_JSON_PATH + ".lock"

# Abstract API Email Verification removed

# Allowed domains (dev includes gmail.com by default)
ALLOWED_EMAIL_DOMAINS = {"euroliquids.com", "ttd.com", "gmail.com"}

# SMTP (optional but recommended for production)
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "0") or 0)
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_STARTTLS = bool(os.getenv("SMTP_STARTTLS", "1") == "1")
MAIL_FROM = os.getenv("MAIL_FROM", "")

# Optional hardcoded dev SMTP (used only if env is empty). Fill these to send via Gmail/other.
DEV_SMTP_HOST = ""  # optional local/dev SMTP host
DEV_SMTP_PORT = 0
DEV_SMTP_USERNAME = ""
DEV_SMTP_PASSWORD = ""
DEV_SMTP_STARTTLS = True
DEV_MAIL_FROM = ""

JWT_ALG = "HS256"
COOKIE_NAME = os.getenv("COOKIE_NAME", "auth_session")
# Default to a very long-lived cookie (10 years) unless overridden
COOKIE_TTL = int(os.getenv("COOKIE_TTL", str(60 * 60 * 24 * 365 * 10)))
COOKIE_SETTINGS = dict(
    key=COOKIE_NAME,
    expires=COOKIE_TTL,
    secure=bool(os.getenv("COOKIE_SECURE", "1") == "1"),
    same_site=os.getenv("COOKIE_SAMESITE", "Lax"),
)

PASSWORD_POLICY = {
    "min_length": 10,
    "require_upper": True,
    "require_lower": True,
    "require_digit": True,
    "require_symbol": True,
}

# Token TTLs
VERIFICATION_TOKEN_TTL_SECONDS = int(os.getenv("VERIFICATION_TOKEN_TTL_SECONDS", str(60 * 60 * 24)))       # 24 uur
RESET_TOKEN_TTL_SECONDS = int(os.getenv("RESET_TOKEN_TTL_SECONDS", str(60 * 30)))                           # 30 min

# =========================
# Data model
# =========================
@dataclass
class User:
    email: str
    role: str

# =========================
# Utilities
# =========================
def _cookie_manager():
    key = "_cookie_manager_instance"
    if key not in st.session_state:
        st.session_state[key] = stx.CookieManager()
    cm = st.session_state[key]
    # Force render/sync once; returns None before the component mounts
    try:
        _ = cm.get_all()
    except Exception:
        pass
    return cm

def _cookie_bootstrap_ready() -> bool:
    """Ensure the CookieManager component has mounted and synced.

    On a fresh app run (e.g., after full browser refresh), Streamlit runs the
    Python script before the front-end component has returned cookie data.
    We render the component and perform a single rerun to allow it to sync.
    """
    cm = _cookie_manager()
    try:
        cookies = cm.get_all()
    except Exception:
        cookies = None
    # None => component not yet ready; {} => ready but no cookies
    if cookies is None and not st.session_state.get("__cookie_bootstrap_once__", False):
        st.session_state["__cookie_bootstrap_once__"] = True
        # Trigger a fast rerun so the component can provide cookies on next pass
        _rerun()
        st.stop()
    return cookies is not None

def _now_ts() -> int:
    return int(time.time())

def _email_valid(email: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", (email or "").strip()))

def _email_domain_allowed(email: str) -> bool:
    try:
        domain = (email or "").strip().split("@", 1)[1].lower()
    except Exception:
        return False
    return domain in ALLOWED_EMAIL_DOMAINS

def _password_valid(pw: str) -> Tuple[bool, str]:
    if len(pw) < PASSWORD_POLICY["min_length"]:
        return False, f"Wachtwoord moet minstens {PASSWORD_POLICY['min_length']} tekens lang zijn."
    if PASSWORD_POLICY["require_upper"] and not re.search(r"[A-Z]", pw):
        return False, "Minstens één hoofdletter vereist."
    if PASSWORD_POLICY["require_lower"] and not re.search(r"[a-z]", pw):
        return False, "Minstens één kleine letter vereist."
    if PASSWORD_POLICY["require_digit"] and not re.search(r"\d", pw):
        return False, "Minstens één cijfer vereist."
    if PASSWORD_POLICY["require_symbol"] and not re.search(r"[^A-Za-z0-9]", pw):
        return False, "Minstens één speciaal teken vereist."
    return True, ""

def _load_users() -> Dict[str, Any]:
    if not os.path.exists(USERS_JSON_PATH):
        return {"roles": {"admin": [], "user": []}, "local_users": []}
    with FileLock(USERS_LOCK_PATH):
        try:
            with open(USERS_JSON_PATH, "r", encoding="utf-8") as f:
                return json_load_safely(f.read())
        except Exception:
            return {"roles": {"admin": [], "user": []}, "local_users": []}

def _save_users(data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(USERS_JSON_PATH) or ".", exist_ok=True)
    with FileLock(USERS_LOCK_PATH):
        with open(USERS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

def _smtp_healthcheck() -> None:
    try:
        import ssl, smtplib
        ctx = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
            s.ehlo()
            s.starttls(context=ctx)
            s.ehlo()
            s.login(SMTP_USERNAME, SMTP_PASSWORD)
        st.success("SMTP inlog OK")
    except Exception as e:
        st.error(f"SMTP fout: {type(e).__name__}: {e}")


def json_load_safely(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Invalid users.json structure")
        data.setdefault("roles", {"admin": [], "user": []})
        local = data.setdefault("local_users", [])
        if isinstance(local, list):
            cleaned = []
            for u in local:
                if not isinstance(u, dict):
                    continue
                email = str(u.get("email", "")).strip()
                role = str(u.get("role", "user")).lower() or "user"
                pw_hash = u.get("password_hash")
                if isinstance(pw_hash, bytes):
                    try:
                        pw_hash = pw_hash.decode("utf-8", errors="ignore")
                    except Exception:
                        pw_hash = ""
                elif not isinstance(pw_hash, str):
                    pw_hash = ""
                email_verified = bool(u.get("email_verified", False))
                verification_token = str(u.get("verification_token") or "")
                verification_expires = int(u.get("verification_expires") or 0)
                reset_token = str(u.get("reset_token") or "")
                reset_expires = int(u.get("reset_expires") or 0)
                cleaned.append({
                    "email": email,
                    "role": role,
                    "password_hash": pw_hash,
                    "email_verified": email_verified,
                    "verification_token": verification_token,
                    "verification_expires": verification_expires,
                    "reset_token": reset_token,
                    "reset_expires": reset_expires,
                })
            data["local_users"] = cleaned
        return data
    except Exception:
        return {"roles": {"admin": [], "user": []}, "local_users": []}

def _issue_jwt(email: str, role: str, ttl: int) -> str:
    now = _now_ts()
    payload = {
        "sub": email,
        "role": role,
        "iat": now,
        "exp": now + ttl,
        "jti": secrets.token_urlsafe(16),
        "app": APP_NAME,
    }
    return jwt.encode(payload, COOKIE_SECRET, algorithm=JWT_ALG)

def _verify_jwt(token: str) -> Optional[User]:
    try:
        payload = jwt.decode(token, COOKIE_SECRET, algorithms=[JWT_ALG])
        email = payload.get("sub")
        role = payload.get("role")
        if not email or not role:
            return None
        return User(email=email, role=str(role))
    except JWTError:
        return None

def _get_cookie() -> Optional[str]:
    cm = _cookie_manager()
    try:
        # Force a sync of cookies so that on first render the
        # component has a chance to load existing cookies.
        try:
            _ = cm.get_all()
        except Exception:
            pass
        return cm.get(COOKIE_NAME)
    except TypeError:
        try:
            return cm.get(cookie=COOKIE_NAME)
        except TypeError:
            return None

def _js_set_cookie(token: str, ttl: int) -> None:
    """Client-side fallback: set cookie via inline JS.

    This complements CookieManager for environments where its setter
    might be ignored. Runs in the current render and should be fast.
    """
    secure_flag = bool(os.getenv("COOKIE_SECURE", "1") == "1")
    same_site_flag = os.getenv("COOKIE_SAMESITE", "Lax")
    expires_dt = datetime.utcnow() + timedelta(seconds=ttl)
    # RFC1123/GMT format for Expires
    expires_str = expires_dt.strftime("%a, %d %b %Y %H:%M:%S GMT")
    # Build cookie attributes
    attrs = ["Path=/", f"Max-Age={ttl}", f"Expires={expires_str}"]
    if same_site_flag:
        attrs.append(f"SameSite={same_site_flag}")
    if secure_flag:
        attrs.append("Secure")
    cookie_js = (
        f"document.cookie = '{COOKIE_NAME}=' + encodeURIComponent(%TOKEN%) + '; "
        + "; ".join(attrs) + "';"
    )
    # Inject token safely
    safe_token = token.replace("\\", "\\\\").replace("'", "\\'")
    js = "<script>" + cookie_js.replace("%TOKEN%", "'" + safe_token + "'") + "</script>"
    try:
        st_html(js, height=0)
    except Exception:
        pass

def _set_cookie(token: str, ttl: int) -> None:
    cm = _cookie_manager()
    # Try multiple signatures for compatibility across versions of extra_streamlit_components
    # Also try to set a long-lived expiry where supported.
    expires_days = max(1, int(ttl // (60 * 60 * 24)))
    expires_at = datetime.utcnow() + timedelta(seconds=ttl)
    secure_flag = bool(os.getenv("COOKIE_SECURE", "1") == "1")
    same_site_flag = os.getenv("COOKIE_SAMESITE", "Lax")
    for args, kwargs in [
        # Prefer explicit modern attributes first
        ((), {"key": COOKIE_NAME, "value": token, "path": "/", "expires_days": expires_days, "same_site": same_site_flag, "secure": secure_flag}),
        ((), {"cookie": COOKIE_NAME, "value": token, "path": "/", "expires_days": expires_days, "same_site": same_site_flag, "secure": secure_flag}),
        # Alternate parameter names sometimes used
        ((), {"key": COOKIE_NAME, "value": token, "path": "/", "expires_days": expires_days, "samesite": same_site_flag, "secure": secure_flag}),
        ((), {"cookie": COOKIE_NAME, "value": token, "path": "/", "expires_days": expires_days, "samesite": same_site_flag, "secure": secure_flag}),
        # Some versions support expires_at instead of expires_days
        ((), {"key": COOKIE_NAME, "value": token, "path": "/", "expires_at": expires_at, "same_site": same_site_flag, "secure": secure_flag}),
        ((), {"cookie": COOKIE_NAME, "value": token, "path": "/", "expires_at": expires_at, "same_site": same_site_flag, "secure": secure_flag}),
        # Fallbacks without same_site/secure
        ((), {"key": COOKIE_NAME, "value": token, "path": "/", "expires_days": expires_days}),
        ((), {"cookie": COOKIE_NAME, "value": token, "path": "/", "expires_days": expires_days}),
        ((COOKIE_NAME, token), {}),
        ((), {"key": COOKIE_NAME, "value": token}),
        ((), {"cookie": COOKIE_NAME, "value": token}),
    ]:
        try:
            cm.set(*args, **kwargs)
            # Cache in session as a short-term fallback in case the
            # front-end hasn't persisted the cookie before a rerun.
            try:
                st.session_state["__auth_token__"] = token
            except Exception:
                pass
            # Also set via JS as a robust client-side fallback
            _js_set_cookie(token, ttl)
            return
        except TypeError:
            continue

def _clear_cookie() -> None:
    cm = _cookie_manager()
    try:
        cm.delete(COOKIE_NAME)
    except TypeError:
        try:
            cm.delete(cookie=COOKIE_NAME)
        except TypeError:
            pass
    # Also clear session fallback
    try:
        if "__auth_token__" in st.session_state:
            del st.session_state["__auth_token__"]
    except Exception:
        pass

def _users_find(email: str) -> Optional[Dict[str, Any]]:
    data = _load_users()
    for u in data.get("local_users", []):
        if (u.get("email") or "").lower() == (email or "").lower():
            return u
    return None

def _users_count() -> int:
    data = _load_users()
    return len(data.get("local_users", []))

def _store_users(data: Dict[str, Any], uobj: Dict[str, Any]) -> None:
    updated = False
    for i, u in enumerate(data.get("local_users", [])):
        if (u.get("email") or "").lower() == (uobj.get("email") or "").lower():
            data["local_users"][i] = uobj
            updated = True
            break
    if not updated:
        data.setdefault("local_users", []).append(uobj)
    _save_users(data)

def _upsert_user(email: str, password_plain: str, role: str, email_verified: bool = False) -> None:
    data = _load_users()
    hashed = bcrypt.hashpw(password_plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    u = _users_find(email) or {"email": email}
    u["password_hash"] = hashed
    u["role"] = role
    u.setdefault("email_verified", email_verified)
    u.setdefault("verification_token", "")
    u.setdefault("verification_expires", 0)
    u.setdefault("reset_token", "")
    u.setdefault("reset_expires", 0)
    _store_users(data, u)

    roles = data.setdefault("roles", {"admin": [], "user": []})
    if role not in roles:
        roles[role] = []
    for rname, lst in roles.items():
        if rname == role:
            if email not in lst:
                lst.append(email)
        else:
            if email in lst:
                lst.remove(email)
    _save_users(data)

def _effective_role(email: str) -> str:
    data = _load_users()
    roles = data.get("roles", {})
    if email in roles.get("admin", []):
        return "admin"
    if email in roles.get("user", []):
        return "user"
    u = _users_find(email)
    if u and u.get("role"):
        return str(u["role"]).lower()
    return "user"

def _verification_needed(email: str) -> bool:
    u = _users_find(email)
    if not u:
        return True
    return bool(not u.get("email_verified", False))

def _create_token() -> str:
    return secrets.token_urlsafe(32)

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _send_email(to_addr: str, subject: str, html: str) -> bool:
    # Build message once, so we can reuse for .eml fallback
    msg = EmailMessage()
    from_addr = MAIL_FROM or DEV_MAIL_FROM
    msg["From"] = from_addr or "no-reply@example.com"
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content("HTML e-mail vereist. Gebruik een moderne e-mail client.")
    msg.add_alternative(html, subtype="html")

    # Resolve SMTP configuration: prefer explicit env SMTP_*; otherwise, only use DEV_* if fully specified
    use_dev = not (SMTP_HOST and SMTP_USERNAME and SMTP_PASSWORD and MAIL_FROM)
    host = (DEV_SMTP_HOST if use_dev else SMTP_HOST) or ""
    port = (DEV_SMTP_PORT if use_dev else SMTP_PORT)
    user = (DEV_SMTP_USERNAME if use_dev else SMTP_USERNAME) or ""
    pwd = (DEV_SMTP_PASSWORD if use_dev else SMTP_PASSWORD) or ""
    use_starttls = (DEV_SMTP_STARTTLS if use_dev else SMTP_STARTTLS)

    if host and from_addr:
        try:
            import ssl
            with smtplib.SMTP(host, port, timeout=30) as s:
                # Be explicit with EHLO/TLS to satisfy stricter servers (e.g., Gmail)
                try:
                    s.ehlo()
                except Exception:
                    pass
                if use_starttls:
                    try:
                        ctx = ssl.create_default_context()
                        s.starttls(context=ctx)
                        s.ehlo()
                    except Exception as e:
                        print(f"[mailer] STARTTLS error: {e}")
                        raise
                if user:
                    s.login(user, pwd)
                s.send_message(msg)
            return True
        except Exception as e:
            print(f"[mailer] send error: {e}")
            # fall through to .eml/console fallback

    # Fallback: write .eml file + console output for dev
    try:
        out_dir = os.path.join(os.getcwd(), "dist", "emails")
        os.makedirs(out_dir, exist_ok=True)
        safe_subj = re.sub(r"[^\w\-]+", "_", subject)[:60]
        fname = f"{int(time.time())}_{safe_subj}.eml"
        fpath = os.path.join(out_dir, fname)
        with open(fpath, "wb") as f:
            f.write(bytes(msg))
        print(f"[mailer] wrote .eml to: {fpath}")
    except Exception as e:
        print(f"[mailer] .eml write failed: {e}")

    print("=== EMAIL (console fallback) ===")
    print(f"TO: {to_addr}")
    print(f"SUBJECT: {subject}")
    print(html)
    print("=== END EMAIL ===")
    return False

def _build_app_url_with_query(params: Dict[str, str]) -> str:
    # Prefer explicit APP_BASE_URL when provided.
    base = APP_BASE_URL.rstrip("/")
    query = "&".join([f"{k}={v}" for k, v in params.items()])
    if base:
        return f"{base}/?{query}"
    # Fallback: relative link. Works if app is mounted at /
    return f"/?{query}"

# _abstract_verify_email removed

def _send_verification_email(email: str, token: str) -> bool:
    verify_url = _build_app_url_with_query({"verify": token})
    subject = f"{APP_NAME}: Verifieer je e-mailadres"
    html = f"""
    <p>Bevestig je e-mailadres voor <b>{APP_NAME}</b>.</p>
    <p>Klik op onderstaande knop om je account te verifiëren.</p>
    <p><a href="{verify_url}" target="_blank" rel="noopener"
          style="display:inline-block;padding:10px 16px;border-radius:6px;background:#2f54eb;color:#fff;text-decoration:none">
         E-mailadres verifiëren
       </a></p>
    <p>Werkt de knop niet? Kopieer deze link: <br>{verify_url}</p>
    <p>Deze link verloopt over 24 uur.</p>
    """
    return _send_email(email, subject, html)

def _send_reset_email(email: str, token: str) -> bool:
    reset_url = _build_app_url_with_query({"reset": token})
    subject = f"{APP_NAME}: Wachtwoord herstellen"
    html = f"""
    <p>Er is een verzoek gedaan om het wachtwoord te herstellen voor <b>{APP_NAME}</b>.</p>
    <p>Als jij dit was, klik dan op onderstaande knop om een nieuw wachtwoord in te stellen.</p>
    <p><a href="{reset_url}" target="_blank" rel="noopener"
          style="display:inline-block;padding:10px 16px;border-radius:6px;background:#2f54eb;color:#fff;text-decoration:none">
         Nieuw wachtwoord instellen
       </a></p>
    <p>Werkt de knop niet? Kopieer deze link: <br>{reset_url}</p>
    <p>Deze link verloopt over 30 minuten.</p>
    """
    return _send_email(email, subject, html)

# =========================
# UI + Auth flows
# =========================
def _auth_forms() -> Optional[User]:
    st.markdown("## Rekenmodule Login")
    # Flash message from previous action (e.g., after rerun)
    _flash = st.session_state.pop("_flash_notice", None)
    if isinstance(_flash, dict):
        lvl = str(_flash.get("level", "info")).lower()
        msg = str(_flash.get("msg", "")).strip()
        code = _flash.get("code")
        if msg:
            if lvl == "success":
                st.success(msg)
            elif lvl == "warning":
                st.warning(msg)
            elif lvl == "error":
                st.error(msg)
            else:
                st.info(msg)
        if code:
            try:
                st.code(str(code), language="text")
            except Exception:
                st.write(str(code))
            # If code looks like a URL, also show a clickable link
            try:
                _code_str = str(code)
                if _code_str.startswith("http://") or _code_str.startswith("https://") or _code_str.startswith("/?"):
                    st.markdown(f"[Open link]({_code_str})")
            except Exception:
                pass
    tabs = st.tabs(["Inloggen", "Registreren", "Wachtwoord vergeten"])  # type: ignore[attr-defined]

    # Inloggen tab
    with tabs[0]:
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("E-mail", key="login_email")
            pw = st.text_input("Wachtwoord", type="password", key="login_pw")
            submitted = st.form_submit_button("Inloggen")
        if submitted:
            if not _email_valid(email):
                st.error("Ongeldig e-mailadres.")
            elif not _email_domain_allowed(email):
                st.error("Alleen e-mails van euroliquids.com en ttd.com zijn toegestaan.")
            else:
                rec = _users_find(email)
                if not rec:
                    st.error("Onbekende gebruiker of fout wachtwoord.")
                else:
                    pw_hash = rec.get("password_hash")
                    if isinstance(pw_hash, bytes):
                        pw_hash_str = pw_hash.decode("utf-8", errors="ignore")
                    elif isinstance(pw_hash, str):
                        pw_hash_str = pw_hash
                    else:
                        pw_hash_str = None
                    try:
                        ok = bool(pw_hash_str) and bcrypt.checkpw(
                            pw.encode("utf-8"), pw_hash_str.encode("utf-8")
                        )
                    except Exception:
                        ok = False
                    if not ok:
                        st.error("Onbekende gebruiker of fout wachtwoord. Indien het probleem blijft, reset het wachtwoord.")
                    else:
                        if not bool(rec.get("email_verified", False)):
                            st.warning("E-mailadres is nog niet geverifieerd.")
                            if st.button("Verificatie e-mail opnieuw sturen", key="resend_verify_login"):
                                token = rec.get("verification_token") or _create_token()
                                rec["verification_token"] = token
                                rec["verification_expires"] = int((_utcnow() + timedelta(seconds=VERIFICATION_TOKEN_TTL_SECONDS)).timestamp())
                                data = _load_users()
                                _store_users(data, rec)
                                sent = _send_verification_email(email, token)
                                if sent:
                                    st.info("Verificatie e-mail verzonden.")
                                else:
                                    st.info("Verificatielink is gegenereerd. Controleer de console of e-mailconfiguratie.")
                                    verify_url = _build_app_url_with_query({"verify": token})
                                    st.code(verify_url, language="text")
                                    st.markdown(f"[Open link]({verify_url})")
                            st.stop()
                        role = _effective_role(email)
                        ttl = COOKIE_TTL
                        token = _issue_jwt(email, role, ttl)
                        st.session_state["_login_toast"] = {"email": email, "role": role, "ts": time.time()}
                        _set_cookie(token, ttl)
                        st.success("Ingelogd. Vernieuwen...")
                        _rerun()

    # Registreren tab
    with tabs[1]:
        with st.form("register_form", clear_on_submit=False):
            email_r = st.text_input("E-mail", key="reg_email")
            pw_r = st.text_input("Wachtwoord", type="password", key="reg_pw")
            pw_r2 = st.text_input("Herhaal wachtwoord", type="password", key="reg_pw2")
            submitted_r = st.form_submit_button("Registreren")
        if submitted_r:
            if not _email_valid(email_r):
                st.error("Ongeldig e-mailadres.")
            elif not _email_domain_allowed(email_r):
                st.error("Alleen e-mails van euroliquids.com en ttd.com zijn toegestaan.")
            else:
                if pw_r != pw_r2:
                    st.error("Wachtwoorden komen niet overeen.")
                else:
                    ok_pw, msg_pw = _password_valid(pw_r)
                    if not ok_pw:
                        st.error(msg_pw)
                    else:
                        # First registered user becomes admin
                        role = "admin" if _users_count() == 0 else "user"
                        _upsert_user(email_r, pw_r, role, email_verified=False)

                        # Issue verification token and send mail
                        rec = _users_find(email_r)
                        if rec:
                            token = _create_token()
                            rec["verification_token"] = token
                            rec["verification_expires"] = int((_utcnow() + timedelta(seconds=VERIFICATION_TOKEN_TTL_SECONDS)).timestamp())
                            data = _load_users()
                            _store_users(data, rec)
                            sent = _send_verification_email(email_r, token)
                            if sent:
                                st.session_state["_flash_notice"] = {
                                    "level": "success",
                                    "msg": "Account aangemaakt. Controleer je inbox om je e-mailadres te verifiëren. Na verificatie kun je inloggen.",
                                }
                            else:
                                verify_url = _build_app_url_with_query({"verify": token})
                                st.session_state["_flash_notice"] = {
                                    "level": "warning",
                                    "msg": "Account aangemaakt. Verificatielink is gegenereerd. Controleer de e-mailconfiguratie. Gebruik indien nodig onderstaande link.",
                                    "code": verify_url,
                                }

                        # Do not auto-login before verification; refresh to clear form and show flash
                        _rerun()

    # Wachtwoord vergeten tab
    with tabs[2]:
        with st.form("forgot_form", clear_on_submit=False):
            email_f = st.text_input("E-mail", key="forgot_email")
            submitted_f = st.form_submit_button("Verstuur herstel e-mail")
        if submitted_f:
            if not _email_valid(email_f):
                st.error("Ongeldig e-mailadres.")
            elif not _email_domain_allowed(email_f):
                st.error("Alleen e-mails van euroliquids.com en ttd.com zijn toegestaan.")
            else:
                rec = _users_find(email_f)
                if not rec:
                    st.info("Als het e-mailadres bestaat, is een e-mail verzonden.")
                else:
                    #_smtp_healthcheck()  # Optional: check SMTP before proceeding
                    token = _create_token()
                    rec["reset_token"] = token
                    rec["reset_expires"] = int((_utcnow() + timedelta(seconds=RESET_TOKEN_TTL_SECONDS)).timestamp())
                    data = _load_users()
                    _store_users(data, rec)
                    sent = _send_reset_email(email_f, token)
                    if sent:
                        st.success("Als het e-mailadres bestaat, is een e-mail met instructies voor wachtwoordherstel verzonden.")
                    else:
                        st.info("Een wachtwoordherstel is aangevraagd. Als het e-mailadres bestaat, ontvang je zo een e-mail met verdere instructies.")
                        reset_url = _build_app_url_with_query({"reset": token})
                        st.code(reset_url, language="text")
                        st.markdown(f"[Open link]({reset_url})")

    return None

def _rerun():
    rerun_fn = getattr(st, "experimental_rerun", None)
    if rerun_fn is None:
        rerun_fn = getattr(st, "rerun", None)
    if callable(rerun_fn):
        rerun_fn()

def current_user() -> Optional[User]:
    token = _get_cookie()
    if not token:
        # Fallback to a cached session token immediately after login
        # before the browser has persisted cookies.
        token = st.session_state.get("__auth_token__")
        if not token:
            return None
    return _verify_jwt(token)

def _consume_verification_token(token: str) -> Tuple[bool, str]:
    if not token:
        return False, "Ongeldige verificatielink."
    data = _load_users()
    found = None
    for u in data.get("local_users", []):
        if u.get("verification_token") == token:
            found = u
            break
    if not found:
        return False, "Verificatietoken niet gevonden."
    if int(found.get("verification_expires") or 0) < _now_ts():
        return False, "Verificatielink is verlopen."
    found["email_verified"] = True
    found["verification_token"] = ""
    found["verification_expires"] = 0
    _store_users(data, found)
    return True, "E-mailadres succesvol geverifieerd. Je kunt nu inloggen."

def _consume_reset_token_and_update_password(token: str, new_password: str) -> Tuple[bool, str]:
    if not token:
        return False, "Ongeldige resetlink."
    ok_pw, msg_pw = _password_valid(new_password)
    if not ok_pw:
        return False, msg_pw
    data = _load_users()
    found = None
    for u in data.get("local_users", []):
        if u.get("reset_token") == token:
            found = u
            break
    if not found:
        return False, "Reset token niet gevonden."
    if int(found.get("reset_expires") or 0) < _now_ts():
        return False, "Resetlink is verlopen."
    hashed = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    found["password_hash"] = hashed
    found["reset_token"] = ""
    found["reset_expires"] = 0
    _store_users(data, found)
    return True, "Wachtwoord succesvol bijgewerkt. Je kunt nu inloggen."

def _handle_magic_links() -> None:
    # Process verification and reset links from query params
    q = st.query_params
    verify_token = q.get("verify")
    reset_token = q.get("reset")
    if verify_token:
        ok, msg = _consume_verification_token(verify_token)
        # Store as flash and return to clean login screen
        st.session_state["_flash_notice"] = {"level": ("success" if ok else "error"), "msg": msg}
        try:
            del st.query_params["verify"]
        except Exception:
            pass
        _rerun()
        st.stop()
    if reset_token:
        # Show only the reset form; suppress the rest of the auth UI.
        st.markdown("## Nieuw wachtwoord instellen")
        with st.form("reset_form", clear_on_submit=False):
            pw1 = st.text_input("Nieuw wachtwoord", type="password", key="pw_new_1")
            pw2 = st.text_input("Herhaal nieuw wachtwoord", type="password", key="pw_new_2")
            s = st.form_submit_button("Opslaan")
        if s:
            if pw1 != pw2:
                st.error("Wachtwoorden komen niet overeen.")
            else:
                ok, msg = _consume_reset_token_and_update_password(reset_token, pw1)
                if ok:
                    # On success, flash and return to clean login screen
                    st.session_state["_flash_notice"] = {"level": "success", "msg": msg}
                    try:
                        del st.query_params["reset"]
                    except Exception:
                        pass
                    _rerun()
                    st.stop()
                else:
                    st.error(msg)
        # Provide a back-to-login action
        if st.button("Terug naar inloggen", key="back_to_login_from_reset"):
            try:
                del st.query_params["reset"]
            except Exception:
                pass
            _rerun()
            st.stop()
        # Stop here so the regular auth forms are not rendered beneath
        st.stop()

def login_gate() -> User:
    _inject_layout_css()
    _handle_magic_links()
    # Ensure cookie component is mounted and synced before reading auth
    _cookie_bootstrap_ready()

    user = current_user()
    if user:
        rec = _users_find(user.email)
        if not rec or not bool(rec.get("email_verified", False)):
            st.warning("Je e-mailadres is nog niet geverifieerd. Controleer je inbox.")
            if st.button("Verificatie e-mail opnieuw sturen", key="resend_verify_authgate"):
                if rec:
                    token = rec.get("verification_token") or _create_token()
                    rec["verification_token"] = token
                    rec["verification_expires"] = int((_utcnow() + timedelta(seconds=VERIFICATION_TOKEN_TTL_SECONDS)).timestamp())
                    data = _load_users()
                    _store_users(data, rec)
                    sent = _send_verification_email(user.email, token)
                    if sent:
                        st.info("Verificatie e-mail verzonden.")
                    else:
                        st.info("Verificatielink is gegenereerd. Controleer de console of e-mailconfiguratie.")
                        verify_url = _build_app_url_with_query({"verify": token})
                        st.code(verify_url, language="text")
                        st.markdown(f"[Open link]({verify_url})")
            st.stop()
        return user

    _auth_forms()
    st.stop()

def _inject_layout_css():
    st.markdown(
        """
        <style>
        .block-container {max-width: 1200px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

def _role_label(role: str) -> str:
    return {"admin": "Beheerder", "user": "Gebruiker"}.get(str(role).lower(), str(role))

def sidebar_user_pill() -> None:
    u = current_user()
    if not u:
        return
    role_label = _role_label(u.role)
    sidebar_css = """
    <style>
    section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {
      display: flex; flex-direction: column; height: 100%;
    }
    .sidebar-user-footer { margin-top: auto; padding-bottom: .5rem; }
    .sidebar-user-pill {
      background: rgba(240,242,246,0.9);
      color: #31333F;
      padding: .45rem .75rem;
      border-radius: 999px;
      font-size: 0.85rem;
      margin-bottom: 0.5rem;
      display: inline-block;
      max-width: 100%;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    </style>
    """
    st.sidebar.markdown(sidebar_css, unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-user-footer">', unsafe_allow_html=True)
    st.sidebar.markdown(
        f'<div class="sidebar-user-pill">Ingelogd als: <b>{u.email}</b> ({role_label})</div>',
        unsafe_allow_html=True,
    )
    if st.sidebar.button("Uitloggen", key="logout_btn_sidebar"):
        _clear_cookie()
        _rerun()
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

def logout_button(label: str = "Uitloggen") -> None:
    return None

# --- Optional toast in bottom-right after login ---
def show_login_toast() -> None:
    u = current_user()
    if not u:
        return
    toast = st.session_state.get("_login_toast")
    if not toast:
        return
    try:
        ts = float(toast.get("ts", 0))
    except Exception:
        ts = 0.0
    if time.time() - ts > 7.0:
        return
    role_label = {"admin": "Beheerder", "user": "Gebruiker"}.get((toast.get("role") or u.role).lower(), u.role)
    email = toast.get("email") or u.email
    css = """
    <style>
    @keyframes fadeOutSlow { 0% {opacity:1} 80% {opacity:1} 100% {opacity:0} }
    .login-toast { position: fixed; right: 1rem; bottom: 1rem; z-index: 1000;
      background: rgba(255,255,255,0.95); color:#31333F; padding:.5rem .75rem;
      border-radius:999px; box-shadow:0 1px 3px rgba(0,0,0,.15);
      animation: fadeOutSlow 5s ease-in forwards; }
    </style>
    """
    html = f'<div class="login-toast">Ingelogd als: <b>{email}</b> ({role_label})</div>'
    st.markdown(css + html, unsafe_allow_html=True)

# =========================
# Role guards (public helpers)
# =========================
def require_role(user: Optional[User], allowed: Tuple[str, ...] = ("admin", "user")) -> None:
    """Ensure the current user has one of the allowed roles.

    Usage in pages:
        require_role(user, allowed=("admin",))
    """
    if user is None:
        st.error("Niet ingelogd. Log in om verder te gaan.")
        st.stop()
    role = str(user.role or "").lower()
    allowed_norm = tuple(str(r).lower() for r in allowed)
    if role not in allowed_norm:
        st.error("Onvoldoende rechten voor deze handeling.")
        st.stop()

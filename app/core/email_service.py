# app/core/email_service.py
from __future__ import annotations

import os
import smtplib
from email.mime.text import MIMEText
from typing import Tuple

"""
Sales handoff mailer.

Usage:
- Prepare subject/body elsewhere (see sales API).
- Call send_manager_email(subject, body) to dispatch.
- If SMTP env vars are missing, we DO NOT fail; we return a DRY-RUN preview.
"""

MANAGER_EMAIL = os.getenv("MANAGER_EMAIL", "mdyasir3011@gmail.com")

# Optional SMTP configuration (set in .env if you want real sends)
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = os.getenv("SMTP_PORT")  # e.g., "465"
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER or "no-reply@local.invalid")


def send_manager_email(subject: str, body: str) -> Tuple[bool, str]:
    """
    Returns:
        (True, "sent") on success with SMTP configured.
        (False, preview_text) if SMTP not configured (DRY RUN; logged/returned for UI preview).
        (False, error_message) if SMTP attempt failed.
    """
    # DRY RUN if SMTP is not configured
    if not (SMTP_HOST and SMTP_PORT and SMTP_USER and SMTP_PASS):
        preview = (
            "=== EMAIL PREVIEW (DRY RUN) ===\n"
            f"To: {MANAGER_EMAIL}\n"
            f"From: {SMTP_FROM}\n"
            f"Subject: {subject}\n\n"
            f"{body}\n"
            "=== END PREVIEW ==="
        )
        return (False, preview)

    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = SMTP_FROM
        msg["To"] = MANAGER_EMAIL

        with smtplib.SMTP_SSL(SMTP_HOST, int(SMTP_PORT)) as smtp:
            smtp.login(SMTP_USER, SMTP_PASS)
            smtp.sendmail(SMTP_FROM, [MANAGER_EMAIL], msg.as_string())

        return (True, "sent")
    except Exception as e:
        return (False, f"send_error: {e}")


def build_lead_subject(product_ref: str, user_name: str) -> str:
    """Standardized subject line for manager handoff."""
    safe_product = (product_ref or "Product").strip()
    safe_user = (user_name or "Prospect").strip()
    return f"Lead: {safe_product} — Price/Discount Inquiry from {safe_user}"


def build_lead_body(
    user_name: str,
    user_email: str,
    user_phone: str,
    product_ref: str,
    summary: str,
    best_time: str | None = None,
    quoted_price: str | None = None,
) -> str:
    """
    Opinionated, manager-friendly plaintext template. Keep it succinct and actionable.
    """
    lines = [
        "Hello Team,",
        "",
        f"Customer: {user_name}",
        f"Email: {user_email}",
        f"Phone: {user_phone}",
        "",
        f"Interest: {product_ref}",
        f"Context Summary:\n{summary.strip() if summary else '-'}",
    ]
    if quoted_price:
        lines.append(f"Quoted Price (if applicable): {quoted_price}")
    lines.extend(
        [
            "",
            "Requested Action: Price confirmation / Discount options / Availability",
            f"Best Time to Reach: {best_time or '-'}",
            "",
            "Thanks,",
            "AI Sales Assistant",
        ]
    )
    return "\n".join(lines)

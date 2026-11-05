"""Minimal ``multipart/form-data`` reader for the operator photo upload.

The stdlib server has no multipart parser. ``email`` already understands the
RFC 2046 framing, headers and binary payloads, so the request body is wrapped
into a synthetic message instead of hand-splitting on the boundary. Only the
first part named ``file`` is returned; nothing is written to disk here.
"""
from __future__ import annotations

from email import policy
from email.parser import BytesParser


class MultipartError(ValueError):
    """Malformed or incomplete upload (HTTP 400)."""


def file_part(content_type, body, field="file"):
    """Return ``(filename, data)`` for the named file field of a form upload."""
    content_type = (content_type or "").strip()
    if not content_type.lower().startswith("multipart/form-data"):
        raise MultipartError("Ожидается multipart/form-data с полем file")
    if not body:
        raise MultipartError("Пустое тело загрузки")
    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + content_type.encode("latin-1", "replace")
        + b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
    if not message.is_multipart():
        raise MultipartError("Некорректный multipart-запрос")
    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        if part.get_param("name", header="content-disposition") != field:
            continue
        data = part.get_payload(decode=True)
        if not data:
            raise MultipartError("Файл не передан")
        return part.get_filename() or "receipt.jpg", data
    raise MultipartError(f"В форме нет файла в поле {field}")

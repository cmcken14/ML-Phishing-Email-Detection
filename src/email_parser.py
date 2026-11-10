
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
import re


class HTMLTextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("head", "style", "script", "title", "noscript"):
            self.skip += 1
        elif not self.skip:
            if tag == "img":
                alt = dict(attrs).get("alt", "")
                if alt:
                    self.parts.append(" " + alt + " ")
            elif tag in (
                "br", "p", "div", "li", "tr", "h1", "h2", "h3"
            ):
                self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("head", "style", "script", "title", "noscript"):
            self.skip = max(self.skip - 1, 0)
        elif not self.skip and tag in (
            "p", "div", "li", "tr", "h1", "h2", "h3"
        ):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_to_text(html):
    html = re.split(
        r"</html\s*>", html, maxsplit=1, flags=re.I
    )[0]

    parser = HTMLTextParser()
    parser.feed(html)

    return re.sub(
        r"\s+", " ", " ".join(parser.parts)
    ).strip()


def read_part(part):
    raw = part.get_payload(decode=False)

    if (
        part.get_content_type() == "text/html"
        and isinstance(raw, str)
        and re.search(
            r"^\s*(?:<!doctype\s+html|<html\b)",
            raw,
            re.I
        )
    ):
        return raw

    try:
        return part.get_content()
    except Exception:
        return ""


def parse_email(file):
    email_message = BytesParser(
        policy=policy.default
    ).parse(file)

    sender = str(email_message.get("From", ""))
    receiver = str(email_message.get("To", ""))
    subject = str(email_message.get("Subject", ""))

    body = ""

    # Handle multipart emails
    if email_message.is_multipart():
        for part in email_message.walk():
            if part.get_content_disposition() == "attachment":
                continue

            # Prefer the plain-text email body
            if part.get_content_type() == "text/plain":
                body += read_part(part)

        # Some emails only contain an HTML body
        if not body.strip():
            for part in email_message.walk():
                if part.get_content_disposition() == "attachment":
                    continue

                if part.get_content_type() == "text/html":
                    body += html_to_text(read_part(part))

    # Handle emails that are not multipart
    else:
        body = read_part(email_message)

        if email_message.get_content_type() == "text/html":
            body = html_to_text(body)

    # Match the preprocessing used during training
    text = (subject + " " + body).strip()

    return {
        "sender": sender,
        "receiver": receiver,
        "subject": subject,
        "body": body,
        "text": text
    }

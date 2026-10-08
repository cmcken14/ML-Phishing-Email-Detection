from email import policy
from email.parser import BytesParser


def parse_email(file):
    email_message = BytesParser(
        policy=policy.default
    ).parse(file)

    sender = email_message.get("From", "")
    receiver = email_message.get("To", "")
    subject = email_message.get("Subject", "")

    body = ""

    # Handle multipart emails
    if email_message.is_multipart():
        for part in email_message.walk():
            content_type = part.get_content_type()
            content_disposition = str(
                part.get("Content-Disposition", "")
            )

            # Ignore attachments
            if "attachment" in content_disposition:
                continue

            # Prefer the plain-text email body
            if content_type == "text/plain":
                try:
                    body += part.get_content()
                except Exception:
                    pass

        # Some emails only contain an HTML body
        if not body.strip():
            for part in email_message.walk():
                if part.get_content_type() == "text/html":
                    try:
                        body += part.get_content()
                    except Exception:
                        pass

    # Handle emails that are not multipart
    else:
        try:
            body = email_message.get_content()
        except Exception:
            body = ""

    # Match the preprocessing used during training
    text = (subject + " " + body).strip()

    return {
        "sender": sender,
        "receiver": receiver,
        "subject": subject,
        "body": body,
        "text": text
    }

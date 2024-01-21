import imaplib
import email
from dotenv import load_dotenv
import os

# IMAP server settings 
imap_host = "imap.gmail.com"
imap_port = 993

load_dotenv()

# Email account credentials
email_user = os.getenv('EMAIL_USER')  
email_pass = os.getenv('EMAIL_PASS')        

# Create an IMAP4_SSL class
mail = imaplib.IMAP4_SSL(imap_host, imap_port)

# Authenticate
if mail.login(email_user, email_pass)[0] == 'OK':
    print("Login successful")
else:
    print("Login failed")

# Select the mailbox you want to check 
mail.select('INBOX')



# Fetch the latest email
status, data = mail.search(None, 'ALL')
latest_email_id = data[0].split()[-1]
status, data = mail.fetch(latest_email_id, '(RFC822)')
raw_email = data[0][1]

# Decode the email content
email_message = email.message_from_bytes(raw_email)

# Function to save image attachments
def save_attachments(part):
    if part.get_content_maintype() == 'image':
        filename = part.get_filename()
        if filename:
            filepath = os.path.join('./attachments', filename)
            # Ensure the directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                f.write(part.get_payload(decode=True))

# Iterate over email parts
for part in email_message.walk():
    save_attachments(part)


# Get embedded images from email body
for part in email_message.walk():
    if part.get_content_maintype() == 'multipart':
        continue
    if part.get('Content-Disposition') is None:
        continue
    if part.get_content_type() == 'image/jpeg' or part.get_content_type() == 'image/png':
        filename = part.get_filename()
        if filename:
            filepath = os.path.join('./attachments', filename)
            with open(filepath, 'wb') as f:
                f.write(part.get_payload(decode=True))
            print(f"Saved embedded image: {filename}")

# Close the connection
mail.logout()


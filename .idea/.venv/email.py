import imaplib


# IMAP server settings (use SSL for security)
imap_host = 'imap.gmail.com'
imap_port = 993

# Email account credentials
email_user = 'your_email@gmail.com'  # Change to your Gmail email
email_pass = 'your_password'         # Change to your Gmail password

# Create an IMAP4_SSL class
mail = imaplib.IMAP4_SSL(imap_host, imap_port)

# Authenticate
mail.login(email_user, email_pass)

# Select the mailbox you want to check (INBOX is default)
mail.select('INBOX')

# Fetch the latest email
status, data = mail.search(None, 'ALL')
latest_email_id = data[0].split()[-1]
status, data = mail.fetch(latest_email_id, '(RFC822)')
raw_email = data[0][1]

# Decode the email content
email_message = email.message_from_bytes(raw_email)

# Process the email as needed
# ...

# Close the connection
mail.logout()


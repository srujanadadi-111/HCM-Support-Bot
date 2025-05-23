import os
import time
import datetime
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import boto3

# --- CONFIGURATION ---
# Get credentials from environment variables
aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
slack_token = os.environ.get('SLACK_BOT_TOKEN')

# Initialize Slack client
client = WebClient(token=slack_token)

# --- SLACK FUNCTIONS ---
def get_all_joined_channel_ids():
    channel_ids = []
    cursor = None
    while True:
        response = client.conversations_list(
            types="public_channel,private_channel",
            limit=1000,
            cursor=cursor
        )
        for channel in response['channels']:
            if channel.get('is_member'):
                channel_ids.append(channel['id'])
        cursor = response.get('response_metadata', {}).get('next_cursor')
        if not cursor:
            break
    return channel_ids

def fetch_hcmsupportbot_messages_all_channels(oldest, latest):
    all_messages = []
    channel_ids = get_all_joined_channel_ids()
    for channel_id in channel_ids:
        try:
            response = client.conversations_history(
                channel=channel_id,
                oldest=oldest,
                latest=latest,
                limit=1000
            )
            for msg in response['messages']:
                if '#hcmsupportbot' in msg.get('text', '').lower():
                    all_messages.append({
                        'channel': channel_id,
                        'user': msg.get('user', ''),
                        'text': msg.get('text', ''),
                        'ts': msg.get('ts')
                    })
        except Exception as e:
            print(f"Error fetching from {channel_id}: {e}")
    return all_messages

# --- PDF FORMATTING ---
styles = getSampleStyleSheet()
style_normal = ParagraphStyle(
    'Normal',
    parent=styles['Normal'],
    fontName='Courier',
    fontSize=10,
    leading=12
)
style_header = ParagraphStyle(
    'Header',
    parent=styles['Heading2'],
    alignment=1,
    spaceAfter=20
)

# --- PDF GENERATION ---
def save_to_pdf(messages, filename):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    flowables = []
    header_text = f"Daily #hcmsupportbot Messages\n{datetime.datetime.now().strftime('%Y-%m-%d')}"
    flowables.append(Paragraph(header_text, style_header))

    for msg in messages:
        # Use actual user/text from Slack message objects
        user = msg.get('user', 'unknown')
        text = msg.get('text', '')
        flowables.append(Paragraph(f"<b>{user}</b>: {text}", style_normal))
        flowables.append(Spacer(1, 12))

    doc.build(flowables)

def upload_to_s3(filename):
    # Create S3 client using environment variables
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    
    s3.upload_file(
        Filename=filename,
        Bucket='hcmbotknowledgesource',
        Key='slack_pdfs/' + filename
    )
    print(f"Uploaded {filename} to s3://hcmbotknowledgesource/slack_pdfs/{filename}")

# --- MAIN WORKFLOW ---
def main():
    # Calculate time range
    now = datetime.datetime.now()
    yesterday = now - datetime.timedelta(days=1)

    # Fetch messages
    messages = fetch_hcmsupportbot_messages_all_channels(
        oldest=int(yesterday.timestamp()),
        latest=int(now.timestamp())
    )

    if messages:
        # Generate PDF
        filename = f"hcmsupportbot_{now.strftime('%Y-%m-%d')}.pdf"
        save_to_pdf(messages, filename)

        # Upload to S3
        upload_to_s3(filename)
        
        print(f"Successfully processed {len(messages)} messages and created {filename}")
        
        # Return the filename for further processing
        return filename
    else:
        print("No messages found for today.")
        return None

if __name__ == "__main__":
    main()

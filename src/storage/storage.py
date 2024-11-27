import torch
import os
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
import boto3

class CloudStorage:
    """Handles cloud storage operations across Google Drive and AWS"""
    def __init__(self, config):
        self.config = config
        try:
            self.setup_google_drive()
            self.setup_aws()
        except Exception as e:
            print(f"Error initializing cloud storage: {e}")
            raise
        
    def setup_google_drive(self):
        """Initialize Google Drive connection"""
        # Load credentials from environment or file
        creds_file = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        self.drive_credentials = service_account.Credentials.from_service_account_file(
            creds_file,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        self.drive_service = build('drive', 'v3', credentials=self.drive_credentials)
        
    def setup_aws(self):
        """Initialize AWS connection"""
        self.s3 = boto3.client(
            's3',
            region_name=self.config.aws_region,
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
        )
    
    async def save_model_state(self, model, filename):
        """Save model state to both Google Drive and AWS"""
        try:
            # Save to temporary file
            temp_path = f"/tmp/{filename}"
            torch.save(model.state_dict(), temp_path)
            
            # Upload to Google Drive
            file_metadata = {
                'name': filename,
                'parents': [self.config.google_drive_folder]
            }
            media = MediaIoBaseUpload(
                io.BytesIO(open(temp_path, 'rb').read()),
                mimetype='application/octet-stream'
            )
            self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            # Upload to AWS
            self.s3.upload_file(
                temp_path,
                self.config.aws_bucket,
                f"models/{filename}"
            )
            
            os.remove(temp_path)
        except Exception as e:
            print(f"Error saving model state: {e}")
            raise
    
    async def load_model_state(self, filename):
        """Load model state from Google Drive"""
        # First try local cache
        cache_path = os.path.join(self.config.cache_dir, filename)
        if os.path.exists(cache_path):
            return torch.load(cache_path, map_location='cpu')
        
        # Download from Google Drive
        files = self.drive_service.files().list(
            q=f"name='{filename}' and parents in '{self.config.google_drive_folder}'",
            fields="files(id, name)"
        ).execute()
        
        if not files['files']:
            raise FileNotFoundError(f"Model state {filename} not found in Google Drive")
            
        file_id = files['files'][0]['id']
        request = self.drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            
        # Save to local cache
        fh.seek(0)
        torch.save(fh.read(), cache_path)
        return torch.load(cache_path, map_location='cpu')
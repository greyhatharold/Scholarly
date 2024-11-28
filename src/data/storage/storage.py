import torch
import os
import io
from pathlib import Path
from typing import Optional
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

class CloudStorage:
    """Handles cloud storage operations across Google Drive"""
    def __init__(self, config):
        self.config = config
        self.drive_service = None
        raw_folder = os.getenv('GOOGLE_DRIVE_FOLDER') or getattr(self.config, 'google_drive_folder', None)
        if raw_folder:
            self.drive_folder_id = raw_folder.split('#')[0].strip().strip('"\'')
        else:
            self.drive_folder_id = None
        
        print(f"Debug - Drive folder value: '{self.drive_folder_id}'")
        try:
            self.setup_google_drive()
            self._validate_drive_folder()
        except Exception as e:
            print(f"Error initializing cloud storage: {e}")
            raise
        
    def setup_google_drive(self):
        """Initialize Google Drive connection with fallback mechanisms"""
        creds_file = self._get_credentials_path()
        
        if not creds_file:
            raise FileNotFoundError(
                "Google credentials not found. Please ensure one of the following:\n"
                "1. Set GOOGLE_APPLICATION_CREDENTIALS environment variable\n"
                "2. Place credentials.json in the project's config directory\n"
                "3. Provide credentials path in the application config"
            )
            
        self.drive_credentials = service_account.Credentials.from_service_account_file(
            creds_file,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        self.drive_service = build('drive', 'v3', credentials=self.drive_credentials)
        
    def _get_credentials_path(self) -> Optional[str]:
        """Get credentials file path with multiple fallback locations"""
        paths_to_try = []
        
        # 1. Try environment variable first
        env_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if env_path:
            paths_to_try.append(Path(env_path))
        
        # 2. Try absolute path from config if available
        if hasattr(self.config, 'google_credentials_path'):
            paths_to_try.append(Path(self.config.google_credentials_path))
        
        # 3. Try project-relative paths
        project_root = Path(self.config.project_root)
        paths_to_try.extend([
            project_root / 'google-credentials.json',
            project_root / 'config' / 'credentials.json',
            project_root / 'config' / 'google-credentials.json',
            project_root / 'config' / 'service-account.json',
            Path.home() / '.config' / 'scholarly' / 'credentials.json'
        ])
        
        # Try each path, ensuring proper resolution and existence
        for path in paths_to_try:
            try:
                resolved_path = path.expanduser().resolve(strict=False)
                if resolved_path.is_file():
                    return str(resolved_path)
            except (OSError, RuntimeError):
                continue
            
        return None
        
    async def save_model_state(self, model, filename):
        """Save model state to Google Drive"""
        try:
            # Save to temporary file
            temp_path = f"/tmp/{filename}"
            torch.save(model.state_dict(), temp_path)
            
            # Upload to Google Drive
            file_metadata = {
                'name': filename,
                'parents': [self.drive_folder_id]
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
            
            os.remove(temp_path)
        except Exception as e:
            print(f"Error saving model state: {e}")
            raise
    
    async def load_model_state(self, filename: str) -> dict:
        """
        Load model state from Google Drive with fallback to local cache
        
        Args:
            filename: Name of the model state file
            
        Returns:
            dict: The model's state dictionary
            
        Raises:
            FileNotFoundError: If model state cannot be found
        """
        # First try local cache
        cache_path = os.path.join(self.config.cache_dir, filename)
        if os.path.exists(cache_path):
            return torch.load(cache_path, map_location='cpu')
        
        try:
            # Try Google Drive
            files = self.drive_service.files().list(
                q=f"name='{filename}' and '{self.drive_folder_id}' in parents",
                fields="files(id, name)",
                pageSize=1
            ).execute()
            
            if files.get('files'):
                file_id = files['files'][0]['id']
                request = self.drive_service.files().get_media(fileId=file_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                
                # Save to local cache
                fh.seek(0)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save(fh.read(), cache_path)
                return torch.load(cache_path, map_location='cpu')
                
            raise FileNotFoundError(
                f"Model state {filename} not found in Google Drive or local cache. "
                "If using Google Drive, ensure the API is enabled at: "
                "https://console.developers.google.com/apis/api/drive.googleapis.com/overview"
            )
                
        except Exception as e:
            print(f"Error loading model state: {str(e)}")
            raise
    
    def _validate_drive_folder(self) -> None:
        """
        Validate Google Drive folder exists and is accessible.
        If folder doesn't exist, creates it. If folder_id is a name, 
        attempts to find or create the corresponding folder.
        """
        if not self.drive_folder_id:
            raise ValueError("Google Drive folder ID not configured")
        
        try:
            print(f"Debug - Validating folder: '{self.drive_folder_id}'")
            # First try to find folder by name if it's not a typical ID format
            if not self._looks_like_drive_id(self.drive_folder_id):
                folder_name = self.drive_folder_id.split('#')[0].strip().strip('"\'')
                query = (
                    f"name='{folder_name}' and "
                    "mimeType='application/vnd.google-apps.folder' and "
                    "trashed=false"
                )
                print(f"Debug - Search query: {query}")
                results = self.drive_service.files().list(
                    q=query,
                    spaces='drive',
                    fields='files(id, name, mimeType)',
                    pageSize=1
                ).execute()
                
                if not results.get('files'):
                    # Create the folder if it doesn't exist
                    print(f"Debug - Creating folder: '{folder_name}'")
                    folder_metadata = {
                        'name': folder_name,
                        'mimeType': 'application/vnd.google-apps.folder'
                    }
                    folder = self.drive_service.files().create(
                        body=folder_metadata,
                        fields='id'
                    ).execute()
                    self.drive_folder_id = folder.get('id')
                    print(f"Debug - Created folder with ID: {self.drive_folder_id}")
                else:
                    self.drive_folder_id = results['files'][0]['id']
            
            # Validate the folder ID exists and is accessible
            folder = self.drive_service.files().get(
                fileId=self.drive_folder_id,
                fields='id,name,mimeType'
            ).execute()
            
            if folder.get('mimeType') != 'application/vnd.google-apps.folder':
                raise ValueError(f"Google Drive ID '{self.drive_folder_id}' is not a folder")
            
        except Exception as e:
            print(f"Warning: Could not validate Google Drive folder: {e}")
            raise

    def _looks_like_drive_id(self, value: str) -> bool:
        """Check if string matches typical Google Drive ID format"""
        return len(value) > 25 and all(c.isalnum() or c == '_' or c == '-' for c in value)
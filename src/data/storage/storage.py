import torch
import os
import io
import logging
from pathlib import Path
from typing import Optional
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

logger = logging.getLogger(__name__)

class CloudStorage:
    """Handles cloud storage operations across Google Drive"""
    def __init__(self, config):
        logger.debug("Initializing CloudStorage")
        self.config = config
        self.drive_service = None
        raw_folder = os.getenv('GOOGLE_DRIVE_FOLDER') or getattr(self.config, 'google_drive_folder', None)
        if raw_folder:
            self.drive_folder_id = raw_folder.split('#')[0].strip().strip('"\'')
            logger.debug(f"Using drive folder ID from environment/config: {self.drive_folder_id}")
        else:
            self.drive_folder_id = None
            logger.debug("No drive folder ID configured")
        
        logger.info(f"Drive folder value: '{self.drive_folder_id}'")
        try:
            self.setup_google_drive()
            self._validate_drive_folder()
        except Exception as e:
            logger.error(f"Error initializing cloud storage: {e}", exc_info=True)
            raise
        
    def setup_google_drive(self):
        """Initialize Google Drive connection with fallback mechanisms"""
        logger.debug("Setting up Google Drive connection")
        creds_file = self._get_credentials_path()
        
        if not creds_file:
            logger.error("Google credentials not found")
            raise FileNotFoundError(
                "Google credentials not found. Please ensure one of the following:\n"
                "1. Set GOOGLE_APPLICATION_CREDENTIALS environment variable\n"
                "2. Place credentials.json in the project's config directory\n"
                "3. Provide credentials path in the application config"
            )
            
        logger.debug(f"Using credentials file: {creds_file}")
        self.drive_credentials = service_account.Credentials.from_service_account_file(
            creds_file,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        self.drive_service = build('drive', 'v3', credentials=self.drive_credentials)
        logger.info("Successfully initialized Google Drive service")
        
    def _get_credentials_path(self) -> Optional[str]:
        """Get credentials file path with multiple fallback locations"""
        logger.debug("Searching for credentials file")
        paths_to_try = []
        
        # 1. Try environment variable first
        env_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if env_path:
            logger.debug(f"Found credentials path in environment: {env_path}")
            paths_to_try.append(Path(env_path))
        
        # 2. Try absolute path from config if available
        if hasattr(self.config, 'google_credentials_path'):
            logger.debug(f"Found credentials path in config: {self.config.google_credentials_path}")
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
                    logger.info(f"Found valid credentials at: {resolved_path}")
                    return str(resolved_path)
            except (OSError, RuntimeError) as e:
                logger.debug(f"Error checking path {path}: {e}")
                continue
            
        logger.warning("No valid credentials file found")
        return None
        
    async def save_model_state(self, state_dict, filename: str):
        """Save model state to Google Drive
        
        Args:
            state_dict: Model state to save (dict or model instance)
            filename: Name of file to save state to
        """
        logger.info(f"Saving model state to {filename}")
        try:
            # Create all necessary directories
            os.makedirs("/tmp", exist_ok=True)
            os.makedirs(os.path.dirname(os.path.join("/tmp", filename)), exist_ok=True)
            os.makedirs(self.config.cache_dir, exist_ok=True)
            
            # Save to temporary file
            temp_path = f"/tmp/{filename}"
            if isinstance(state_dict, (dict, bytes)):
                logger.debug("Saving raw state dict")
                torch.save(state_dict, temp_path)
            else:
                logger.debug("Saving model state dict")
                torch.save(state_dict.state_dict(), temp_path)
            
            # Upload to Google Drive
            file_metadata = {
                'name': filename,
                'parents': [self.drive_folder_id]
            }
            
            # Ensure cache directory exists
            cache_path = os.path.join(self.config.cache_dir, filename)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            # Copy to cache
            logger.debug(f"Copying to cache: {cache_path}")
            with open(temp_path, 'rb') as src, open(cache_path, 'wb') as dst:
                dst.write(src.read())
                
            # Upload to Drive
            try:
                # Check if file already exists
                logger.debug("Checking for existing file in Drive")
                files = self.drive_service.files().list(
                    q=f"name='{filename}' and '{self.drive_folder_id}' in parents",
                    fields="files(id)",
                    pageSize=1
                ).execute()
                
                if files.get('files'):
                    # Update existing file
                    file_id = files['files'][0]['id']
                    logger.info(f"Updating existing file {file_id}")
                    media = MediaIoBaseUpload(
                        io.BytesIO(open(temp_path, 'rb').read()),
                        mimetype='application/octet-stream',
                        resumable=True
                    )
                    self.drive_service.files().update(
                        fileId=file_id,
                        media_body=media
                    ).execute()
                else:
                    # Create new file
                    logger.info("Creating new file in Drive")
                    media = MediaIoBaseUpload(
                        io.BytesIO(open(temp_path, 'rb').read()),
                        mimetype='application/octet-stream',
                        resumable=True
                    )
                    self.drive_service.files().create(
                        body=file_metadata,
                        media_body=media,
                        fields='id'
                    ).execute()
            except Exception as e:
                logger.warning(f"Error uploading to Google Drive: {e}", exc_info=True)
                # Continue since we have local cache
            
            os.remove(temp_path)
            logger.debug("Removed temporary file")
        except Exception as e:
            logger.error(f"Error saving model state: {e}", exc_info=True)
            raise
    
    async def load_model_state(self, filename: str):
        """Load model state from storage with fallback mechanisms"""
        try:
            # Try local cache first
            cache_path = os.path.join(self.config.cache_dir, filename)
            if os.path.exists(cache_path):
                logger.debug(f"Loading from cache: {cache_path}")
                with torch.serialization.safe_globals(['numpy._core.multiarray._reconstruct']):
                    return torch.load(cache_path, weights_only=False)

            # Try Google Drive if available
            if self.drive_service:
                try:
                    logger.debug(f"Searching for file in Drive: {filename}")
                    files = self.drive_service.files().list(
                        q=f"name='{filename}' and '{self.drive_folder_id}' in parents",
                        fields="files(id)",
                        pageSize=1
                    ).execute()

                    if not files.get('files'):
                        raise FileNotFoundError(f"File {filename} not found in Drive")

                    file_id = files['files'][0]['id']
                    request = self.drive_service.files().get_media(fileId=file_id)
                    
                    fh = io.BytesIO()
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    
                    while not done:
                        _, done = downloader.next_chunk()
                    
                    # Save to local cache and return loaded state
                    fh.seek(0)
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with torch.serialization.safe_globals(['numpy._core.multiarray._reconstruct']):
                        state = torch.load(fh, weights_only=False)
                    torch.save(state, cache_path)
                    return state

                except Exception as drive_error:
                    logger.warning(f"Google Drive access failed: {drive_error}")
                    
            raise FileNotFoundError(f"Model state {filename} not found in storage")
                
        except Exception as e:
            logger.error(f"Error loading model state: {e}")
            raise
    
    def _validate_drive_folder(self) -> None:
        """
        Validate Google Drive folder exists and is accessible.
        If folder doesn't exist, creates it. If folder_id is a name, 
        attempts to find or create the corresponding folder.
        """
        if not self.drive_folder_id:
            # Create default folder if none specified
            self.drive_folder_id = 'scholarly-data'
            
        try:
            # First try to find folder by name
            folder_name = self.drive_folder_id.split('#')[0].strip().strip('"\'')
            query = (
                f"name='{folder_name}' and "
                "mimeType='application/vnd.google-apps.folder' and "
                "trashed=false"
            )
            
            results = self.drive_service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name, mimeType)',
                pageSize=1
            ).execute()
            
            if not results.get('files'):
                # Create the folder if it doesn't exist
                folder_metadata = {
                    'name': folder_name,
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                folder = self.drive_service.files().create(
                    body=folder_metadata,
                    fields='id'
                ).execute()
                self.drive_folder_id = folder.get('id')
                print(f"Created new storage folder: {folder_name} ({self.drive_folder_id})")
            else:
                self.drive_folder_id = results['files'][0]['id']
                print(f"Using existing storage folder: {folder_name} ({self.drive_folder_id})")
                
        except Exception as e:
            print(f"Warning: Error validating Google Drive folder: {e}")
            # Create local cache directory as fallback
            os.makedirs(self.config.cache_dir, exist_ok=True)
            print(f"Using local cache directory: {self.config.cache_dir}")

    def _looks_like_drive_id(self, value: str) -> bool:
        """Check if string matches typical Google Drive ID format"""
        logger.debug(f"Checking if '{value}' looks like a Drive ID")
        return len(value) > 25 and all(c.isalnum() or c == '_' or c == '-' for c in value)
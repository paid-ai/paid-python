import os
import tempfile
from pathlib import Path
from typing import Optional

import mutagen


def get_audio_duration(file_input) -> Optional[float]:
    """Extract audio duration in seconds from various file input types."""
    temp_file_path = None
    try:
        # Handle different file input types
        if isinstance(file_input, (str, Path)):
            # File path
            file_path = str(file_input)
        elif hasattr(file_input, "read"):
            # File-like object (IO[bytes])
            # Create a temporary file to analyze
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file_path = temp_file.name
                current_pos = 0
                if hasattr(file_input, "tell"):
                    current_pos = file_input.tell()
                if hasattr(file_input, "seek"):
                    file_input.seek(0)
                temp_file.write(file_input.read())
                if hasattr(file_input, "seek"):
                    file_input.seek(current_pos)
            file_path = temp_file_path
        elif isinstance(file_input, bytes):
            # Raw bytes
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(file_input)
            file_path = temp_file_path
        elif isinstance(file_input, tuple) and len(file_input) > 0:
            # Handle tuple formats: (filename, content), (filename, content, content_type), etc.
            content = file_input[1] if len(file_input) > 1 else file_input[0]
            return get_audio_duration(content)
        else:
            return None

        # Use mutagen to get duration
        audio_file = mutagen.File(file_path)  # type: ignore
        duration = audio_file.info.length if audio_file and hasattr(audio_file, "info") and audio_file.info else None

        # Clean up temporary file if created
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

        return duration

    except Exception:
        # If we can't determine duration, return None rather than failing
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass
        return None

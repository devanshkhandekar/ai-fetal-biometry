from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

@dataclass(frozen=True)
class ScanInput:
    image_path: str
    pixel_spacing_mm: Optional[float] = None
    roi_xyxy: Optional[Tuple[int, int, int, int]] = None

    @property
    def path(self) -> Path:
        return Path(self.image_path)

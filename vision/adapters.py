from typing import Protocol
from core.schemas import Detection

class Segmenter(Protocol):
    def detect(self, frame) -> list[Detection]: ...

class AnomalyDetector(Protocol):
    def score(self, frame) -> float: ...

class UnconfiguredModel:
    def detect(self, frame):
        raise RuntimeError('No trained model configured. Use explicitly labelled replay mode.')
    def score(self, frame):
        raise RuntimeError('No PatchCore memory bank configured.')

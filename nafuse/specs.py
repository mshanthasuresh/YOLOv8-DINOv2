"""Run specification and the prioritized experiment matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

FUSION_ARMS = ("nonorm", "ln", "gn", "lngn", "gated")
ALL_ARMS = ("baseline", *FUSION_ARMS)
DEFAULT_RES = 518
MAIN_EPOCHS = 15
LOW_DATA_EPOCHS = 30


@dataclass(frozen=True)
class RunSpec:
    arm: str
    seed: int
    group: str = "main"
    dataset: str = "voc"
    model: str = "yolov8n"
    dino_res: int = DEFAULT_RES
    epochs: int = MAIN_EPOCHS
    priority: int = 0
    train: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def name(self) -> str:
        res = f"-r{self.dino_res}" if self.arm != "baseline" and self.dino_res != DEFAULT_RES else ""
        return f"{self.dataset}-{self.model}-{self.arm}{res}-s{self.seed}"

    def fusion_cfg(self) -> dict | None:
        return None if self.arm == "baseline" else {"arm": self.arm, "resolution": self.dino_res}

    def to_dict(self) -> dict:
        return {**asdict(self), "name": self.name}


def experiment_matrix(
    epochs: int = MAIN_EPOCHS, low_data_epochs: int = LOW_DATA_EPOCHS, visdrone_epochs: int = MAIN_EPOCHS
) -> list[RunSpec]:
    """All runs in execution order; earlier priorities give a usable story if the queue is cut short."""
    seeds = (0, 1, 2)
    runs: list[RunSpec] = []
    # P0: the headline comparison, interleaved by seed so every arm has n>=1 as early as possible.
    for s in seeds:
        runs += [RunSpec(a, s, epochs=epochs, priority=0) for a in ("baseline", "lngn", "nonorm")]
    # P1: isolate each normalization and the historical gated design.
    for s in seeds:
        runs += [RunSpec(a, s, epochs=epochs, priority=1) for a in ("ln", "gn", "gated")]
    # P2: accuracy-latency curve over DINOv2 input resolution.
    runs += [RunSpec("lngn", 0, group="resolution", dino_res=r, epochs=epochs, priority=2) for r in (224, 336, 448)]
    # P3: low-label regime.
    for s in seeds:
        runs += [
            RunSpec(a, s, group="lowdata", dataset="voc10", epochs=low_data_epochs, priority=3)
            for a in ("baseline", "lngn")
        ]
    # P4: capacity control - a bigger CNN at similar cost.
    runs += [RunSpec("baseline", 0, group="capacity", model="yolov8s", epochs=epochs, priority=4)]
    # P5: second domain (aerial, small objects).
    for s in (0, 1):
        runs += [
            RunSpec(a, s, group="visdrone", dataset="visdrone", epochs=visdrone_epochs, priority=5)
            for a in ("baseline", "lngn")
        ]
    # P6: extra seeds where time allows.
    runs += [RunSpec("lngn", 1, group="resolution", dino_res=r, epochs=epochs, priority=6) for r in (224, 336, 448)]
    runs += [RunSpec("baseline", s, group="capacity", model="yolov8s", epochs=epochs, priority=6) for s in (1, 2)]
    return runs

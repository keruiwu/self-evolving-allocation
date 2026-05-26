"""Self-evolving allocation: greedy + online bandit orchestration."""

from .greedy import GreedyRunner, GreedyConfig
from .bandit import BANDITS, BanditState
from .orchestrator import run_online_bandit, OnlineBanditConfig
from .tasks import TASKS, TaskSpec
from .models import MODELS, ModelSpec

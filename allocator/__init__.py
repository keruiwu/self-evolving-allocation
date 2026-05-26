"""Self-evolving allocation: greedy + online bandit orchestration."""

from .greedy import GreedyRunner, GreedyConfig
from .bandit import BANDITS, BanditState
from .orchestrator import run_online_bandit, OnlineBanditConfig
from .tasks import TASKS, TaskSpec
from .models import MODELS, ModelSpec
from .flops import flops_call, flops_for_run, flops_summary, ALL_POLICIES

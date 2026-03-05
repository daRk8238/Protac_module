"""
AppState — Shared singleton for model + DB across all request handlers.
Loaded once at startup, reused across all requests.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from database.schema import init_db, get_session, Compound, TargetProtein, Bioactivity
from models.gnn import AffinityPredictor

logger = logging.getLogger(__name__)

DB_PATH = "protac_designer.db"
MODEL_PATH = "models/outputs/best_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AppState:
    """Shared application state — initialized once at startup."""

    model: AffinityPredictor = None
    model_loaded: bool = False
    db_engine = None
    compound_count: int = 0
    target_count: int = 0
    activity_count: int = 0

    @classmethod
    def initialize(cls):
        """Load model and connect to database."""
        cls._load_model()
        cls._connect_db()

    @classmethod
    def _load_model(cls):
        model_path = Path(MODEL_PATH)
        if not model_path.exists():
            logger.warning(f"[State] Model not found at {MODEL_PATH}. Predictions will be unavailable.")
            cls.model_loaded = False
            return

        try:
            cls.model = AffinityPredictor(
                node_features=9,
                hidden_dim=128,
                num_layers=3,
                dropout=0.2,
                fp_dim=2048,
            ).to(DEVICE)
            cls.model.load_state_dict(
                torch.load(model_path, map_location=DEVICE, weights_only=True)
            )
            cls.model.eval()
            cls.model_loaded = True
            logger.info(f"[State] Model loaded from {MODEL_PATH} on {DEVICE}")
        except Exception as e:
            logger.error(f"[State] Failed to load model: {e}")
            cls.model_loaded = False

    @classmethod
    def _connect_db(cls):
        try:
            cls.db_engine = init_db(DB_PATH)
            session = get_session(cls.db_engine)
            cls.compound_count = session.query(Compound).count()
            cls.target_count = session.query(TargetProtein).count()
            cls.activity_count = session.query(Bioactivity).count()
            session.close()
            logger.info(
                f"[State] DB connected: {cls.target_count} targets, "
                f"{cls.compound_count} compounds, {cls.activity_count} activities"
            )
        except Exception as e:
            logger.error(f"[State] DB connection failed: {e}")

    @classmethod
    def get_session(cls):
        """Get a new DB session."""
        if cls.db_engine is None:
            raise RuntimeError("Database not initialized")
        return get_session(cls.db_engine)

    @classmethod
    def refresh_counts(cls):
        """Refresh DB counts after pipeline runs."""
        try:
            session = cls.get_session()
            cls.compound_count = session.query(Compound).count()
            cls.target_count = session.query(TargetProtein).count()
            cls.activity_count = session.query(Bioactivity).count()
            session.close()
        except Exception:
            pass

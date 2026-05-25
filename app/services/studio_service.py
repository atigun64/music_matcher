from typing import List

from app.services.track_service import TrackService
from app.storage import STUDIO_STORE
from app.storage.studio_store import StudioStore

from app.models import (
    AlignmentSpec,
    QuerySpec,
    StudioSession,
)

from app.services.run_optimization import run_optimizer as compile_alignment

class StudioService:
    """
    Service for managing the music production studio, including tracks, annotations, and project state.
    """
    def __init__(self, studio_store: StudioStore = STUDIO_STORE):
        self.studio_store = studio_store
    
    def create_studio(self) -> str:
        """
        Create a new studio session and return its ID.
        """
        return self.studio_store.create_studio()
    
    def list_studio_ids(self) -> List[str]:
        """
        Get a list of all existing studio session IDs.
        """
        return self.studio_store.list_studio_ids()

    def get_studio_session(self, studio_id: str) -> StudioSession:
        """
        Load the full studio session, including meta, query, and alignment.
        """
        return self.studio_store.load_session(studio_id)
    
    def save_studio_session(self, session: StudioSession) -> None:
        """
        Save the full studio session, including meta, query, and alignment.
        """
        self.studio_store.save_session(session)

    def load_query(self, studio_id: str) -> QuerySpec:
        """
        Load the query spec for a studio session.
        """
        session = self.studio_store.load_session(studio_id)
        if session.query is None:
            raise ValueError("Query not found for this studio session.")
        return session.query
    def save_query(self, studio_id: str, query: QuerySpec) -> None:
        """
        Save the query spec for a studio session.
        """
        session = self.studio_store.load_session(studio_id)
        session.query = query
        self.studio_store.save_session(session)
    
    def load_alignment(self, studio_id: str) -> AlignmentSpec:
        """
        Load the alignment spec for a studio session.
        """
        session = self.studio_store.load_session(studio_id)
        if session.alignment is None:
            raise ValueError("Alignment not found for this studio session.")
        return session.alignment
    def save_alignment(self, studio_id: str, alignment: AlignmentSpec) -> None:
        """
        Save the alignment spec for a studio session.
        """
        session = self.studio_store.load_session(studio_id)
        session.alignment = alignment
        self.studio_store.save_session(session)

    def run_optimizer(self, studio_id: str) -> AlignmentSpec:
        """
        Run the optimizer for the given studio session and save the alignment result.
        """
        session = self.studio_store.load_session(studio_id)
        if session.query is None:
            raise ValueError("Cannot run optimizer: query is not set for this studio session.")
        alignment = compile_alignment(session.query, track_service=TrackService())
        session.alignment = alignment
        self.studio_store.save_session(session)
        return alignment
    

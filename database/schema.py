"""
Database schema for PROTAC Designer pipeline.
Uses SQLAlchemy ORM with SQLite backend (easy to swap to PostgreSQL later).
"""

from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    Text, DateTime, Boolean, ForeignKey, JSON
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime

Base = declarative_base()


class TargetProtein(Base):
    """Stores target protein metadata and structure info."""
    __tablename__ = "target_proteins"

    id = Column(Integer, primary_key=True)
    uniprot_id = Column(String(20), unique=True, nullable=False, index=True)
    pdb_id = Column(String(10), nullable=True)          # Best structure PDB code
    gene_name = Column(String(100), nullable=True)
    protein_name = Column(String(500), nullable=True)
    organism = Column(String(200), nullable=True)
    sequence = Column(Text, nullable=True)

    # Binding site info (populated after fpocket/structure analysis)
    binding_site_residues = Column(JSON, nullable=True)   # list of residue dicts
    pocket_volume = Column(Float, nullable=True)          # Angstrom^3
    pocket_druggability = Column(Float, nullable=True)    # 0-1 score

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    bioactivities = relationship("Bioactivity", back_populates="target")
    pdb_structures = relationship("PDBStructure", back_populates="target")

    def __repr__(self):
        return f"<TargetProtein {self.gene_name} ({self.uniprot_id})>"


class PDBStructure(Base):
    """Stores individual PDB structure metadata."""
    __tablename__ = "pdb_structures"

    id = Column(Integer, primary_key=True)
    pdb_id = Column(String(10), nullable=False, index=True)
    target_id = Column(Integer, ForeignKey("target_proteins.id"))
    resolution = Column(Float, nullable=True)             # Angstrom
    method = Column(String(100), nullable=True)           # X-ray, cryo-EM, etc.
    has_ligand = Column(Boolean, default=False)
    ligand_smiles = Column(Text, nullable=True)           # co-crystallized ligand
    file_path = Column(String(500), nullable=True)        # local .pdb file path
    created_at = Column(DateTime, default=datetime.utcnow)

    target = relationship("TargetProtein", back_populates="pdb_structures")

    def __repr__(self):
        return f"<PDBStructure {self.pdb_id} @ {self.resolution}Å>"


class Compound(Base):
    """
    Stores small molecule compounds from ChEMBL and ZINC.
    Central table — all compounds live here regardless of source.
    """
    __tablename__ = "compounds"

    id = Column(Integer, primary_key=True)
    source = Column(String(50), nullable=False)           # 'chembl', 'zinc', 'manual'
    source_id = Column(String(100), nullable=True, index=True)  # ChEMBL ID or ZINC ID

    # Structure
    smiles = Column(Text, nullable=False)
    inchi = Column(Text, nullable=True)
    inchikey = Column(String(50), unique=True, nullable=True, index=True)  # dedupe key

    # Physicochemical properties (computed by RDKit)
    mol_weight = Column(Float, nullable=True)
    logp = Column(Float, nullable=True)
    hbd = Column(Integer, nullable=True)                  # H-bond donors
    hba = Column(Integer, nullable=True)                  # H-bond acceptors
    tpsa = Column(Float, nullable=True)                   # Topological polar surface area
    rotatable_bonds = Column(Integer, nullable=True)
    aromatic_rings = Column(Integer, nullable=True)
    num_stereocenters = Column(Integer, nullable=True)

    # Flags
    passes_lipinski = Column(Boolean, nullable=True)
    passes_veber = Column(Boolean, nullable=True)         # oral bioavailability filter
    is_pains = Column(Boolean, nullable=True)             # Pan-assay interference
    is_aggregator = Column(Boolean, nullable=True)

    # For PROTAC components
    compound_role = Column(String(50), nullable=True)     # 'warhead', 'e3_ligand', 'linker', 'full_protac'

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    bioactivities = relationship("Bioactivity", back_populates="compound")

    def __repr__(self):
        return f"<Compound {self.source_id} ({self.source})>"


class Bioactivity(Base):
    """
    Stores bioactivity measurements linking compounds to targets.
    Sourced primarily from ChEMBL.
    """
    __tablename__ = "bioactivities"

    id = Column(Integer, primary_key=True)
    target_id = Column(Integer, ForeignKey("target_proteins.id"), index=True)
    compound_id = Column(Integer, ForeignKey("compounds.id"), index=True)

    # Activity data
    activity_type = Column(String(50), nullable=True)     # IC50, Ki, Kd, EC50
    activity_value = Column(Float, nullable=True)         # raw value
    activity_units = Column(String(20), nullable=True)    # nM, uM, etc.
    pchembl_value = Column(Float, nullable=True)          # -log10(activity in M), ChEMBL standard

    # Assay context
    assay_type = Column(String(50), nullable=True)        # B=binding, F=functional
    assay_description = Column(Text, nullable=True)
    chembl_assay_id = Column(String(50), nullable=True)
    confidence_score = Column(Integer, nullable=True)     # ChEMBL 0-9 confidence

    # Modulator type (we label this)
    modulator_type = Column(String(50), nullable=True)    # 'inhibitor', 'activator', 'degrader', 'unknown'

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    target = relationship("TargetProtein", back_populates="bioactivities")
    compound = relationship("Compound", back_populates="bioactivities")

    def __repr__(self):
        return f"<Bioactivity {self.activity_type}={self.activity_value} {self.activity_units}>"


class E3LigaseBinder(Base):
    """
    Warheads known to bind E3 ligases (for PROTAC design).
    Pre-curated from PROTAC-DB and literature.
    """
    __tablename__ = "e3_ligase_binders"

    id = Column(Integer, primary_key=True)
    e3_ligase = Column(String(100), nullable=False)       # CRBN, VHL, MDM2, IAP, etc.
    compound_id = Column(Integer, ForeignKey("compounds.id"))
    binding_affinity_nm = Column(Float, nullable=True)
    attachment_atom_idx = Column(Integer, nullable=True)  # atom index for linker attachment
    reference = Column(String(500), nullable=True)        # paper DOI or PROTAC-DB ID

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<E3LigaseBinder {self.e3_ligase}>"


def get_engine(db_path: str = "protac_designer.db"):
    """Create SQLite engine. Swap connection string for PostgreSQL in production."""
    return create_engine(f"sqlite:///{db_path}", echo=False)


def init_db(db_path: str = "protac_designer.db"):
    """Initialize database — creates all tables if they don't exist."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    print(f"[DB] Database initialized at: {db_path}")
    return engine


def get_session(engine):
    """Get a database session."""
    Session = sessionmaker(bind=engine)
    return Session()

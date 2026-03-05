"""
Quick sanity check tests — run these without needing API calls.
Tests mol_utils and DB schema in isolation.

Run: python tests/test_pipeline.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_mol_utils():
    """Test RDKit molecule utilities."""
    print("\n[Test] mol_utils...")
    from utils.mol_utils import (
        standardize_smiles, compute_properties,
        check_lipinski, check_pains, morgan_fingerprint,
        smiles_to_inchikey, process_compound
    )

    # Aspirin — should pass all basic checks
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    ibuprofen = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"

    # Test standardization
    canon = standardize_smiles(aspirin)
    assert canon is not None, "Standardization failed"
    print(f"  ✓ Canonical SMILES: {canon}")

    # Test properties
    props = compute_properties(aspirin)
    assert props is not None
    assert 100 < props["mol_weight"] < 300
    assert props["passes_lipinski"] is True
    print(f"  ✓ Properties: MW={props['mol_weight']}, logP={props['logp']}")

    # Test InChIKey (deduplication key)
    key = smiles_to_inchikey(aspirin)
    assert key is not None
    assert len(key) == 27  # InChIKeys are always 27 chars
    print(f"  ✓ InChIKey: {key}")

    # Test fingerprint
    fp = morgan_fingerprint(aspirin)
    assert fp is not None
    assert len(fp) == 2048
    print(f"  ✓ Morgan FP: {fp.sum()} bits set out of 2048")

    # Test PAINS check (aspirin should not be PAINS)
    is_pains = check_pains(aspirin)
    print(f"  ✓ PAINS check (aspirin): {is_pains} (expected False)")

    # Test full pipeline
    result = process_compound(aspirin, source="test", source_id="aspirin_001")
    assert result is not None
    assert result["smiles"] == canon
    print(f"  ✓ Full processing pipeline OK")

    # Test invalid SMILES
    invalid = standardize_smiles("NOT_A_MOLECULE_XYZ")
    assert invalid is None
    print(f"  ✓ Invalid SMILES handled correctly (returns None)")

    print("[Test] mol_utils: ALL PASSED ✓")


def test_database():
    """Test database schema initialization."""
    print("\n[Test] Database schema...")
    import tempfile, os
    from database.schema import init_db, get_session, TargetProtein, Compound, Bioactivity

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        engine = init_db(db_path)
        session = get_session(engine)

        # Create a test target
        target = TargetProtein(
            uniprot_id="P00533_TEST",
            gene_name="EGFR",
            protein_name="Epidermal growth factor receptor",
            organism="Homo sapiens"
        )
        session.add(target)
        session.commit()

        fetched = session.query(TargetProtein).filter_by(uniprot_id="P00533_TEST").first()
        assert fetched is not None
        assert fetched.gene_name == "EGFR"
        print(f"  ✓ Target created and fetched: {fetched}")

        # Create a test compound
        compound = Compound(
            source="test",
            source_id="TEST_001",
            smiles="CC(=O)Oc1ccccc1C(=O)O",
            inchikey="BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
            mol_weight=180.16,
            logp=1.19,
            hbd=1,
            hba=3,
            passes_lipinski=True,
        )
        session.add(compound)
        session.commit()
        print(f"  ✓ Compound created: {compound}")

        # Create a bioactivity link
        activity = Bioactivity(
            target_id=fetched.id,
            compound_id=compound.id,
            activity_type="IC50",
            activity_value=1000.0,
            activity_units="nM",
            pchembl_value=6.0,
            modulator_type="inhibitor",
        )
        session.add(activity)
        session.commit()
        print(f"  ✓ Bioactivity created: {activity}")

        # Test relationship traversal
        assert len(fetched.bioactivities) == 1
        assert fetched.bioactivities[0].compound.source_id == "TEST_001"
        print(f"  ✓ Relationships work correctly")

        session.close()
        print("[Test] Database: ALL PASSED ✓")

    finally:
        os.unlink(db_path)


def test_tanimoto():
    """Test similarity computation."""
    print("\n[Test] Tanimoto similarity...")
    from utils.mol_utils import tanimoto_similarity

    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    ibuprofen = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"

    # Self-similarity should be 1.0
    sim_self = tanimoto_similarity(aspirin, aspirin)
    assert abs(sim_self - 1.0) < 0.001, f"Self-similarity should be 1.0, got {sim_self}"
    print(f"  ✓ Self-similarity: {sim_self:.3f}")

    # Different molecules should have lower similarity
    sim_diff = tanimoto_similarity(aspirin, ibuprofen)
    assert 0.0 <= sim_diff < 1.0
    print(f"  ✓ Aspirin vs Ibuprofen similarity: {sim_diff:.3f}")

    print("[Test] Tanimoto: ALL PASSED ✓")


if __name__ == "__main__":
    print("=" * 50)
    print("PROTAC Designer — Pipeline Tests")
    print("=" * 50)

    try:
        test_mol_utils()
        test_database()
        test_tanimoto()
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED ✓")
        print("=" * 50)
        print("\nNext step: python run_pipeline.py --target EGFR")
    except ImportError as e:
        print(f"\n[ERROR] Missing dependency: {e}")
        print("Run: pip install rdkit sqlalchemy")
    except AssertionError as e:
        print(f"\n[FAILED] Assertion error: {e}")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()

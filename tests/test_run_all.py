from data_generation.run_all import PIPELINE
from data_generation.run_all import main


def test_pipeline_has_contract_order():
    assert PIPELINE.index("data_generation.make_company_master") < PIPELINE.index("data_generation.make_mock_data")
    assert PIPELINE.index("data_generation.make_mock_data") < PIPELINE.index("data_generation.gen_calendar")
    assert PIPELINE.index("data_generation.gen_esg") < PIPELINE.index("data_generation.build_model_dataset")
    assert PIPELINE.index("data_generation.build_model_dataset") < PIPELINE.index("data_generation.build_variants")


def test_run_all_skips_missing_generators_without_crashing(capsys):
    main()
    output = capsys.readouterr().out
    assert "Running SIRPS data pipeline" in output

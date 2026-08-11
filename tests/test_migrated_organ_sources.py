"""Regression tests owned by the public acquisition package."""

from congreso_open_data.organ_sources import (
    _canonical_export_groups_match,
    _canonical_export_rows,
    organ_data_resources,
    organ_inventory_resources,
)


def _inventory(*rows: tuple[str, str]) -> dict:
    return {
        "datosOrganos": [
            {
                "urlExport": (
                    "/organos/composicion-en-la-legislatura?p_p_id=organos&"
                    "_organos_statusOpenData=true&"
                    f"_organos_selectedOrganoSup={organ_sup}&"
                    f"_organos_selectedSuborgano={suborgan}&"
                    "_organos_selectedLegislatura=XV"
                ),
                "descOrgano": f"Órgano {suborgan}",
                "indexOrgano": index,
            }
            for index, (organ_sup, suborgan) in enumerate(rows)
        ]
    }


def test_organ_inventory_freezes_html_and_both_dynamic_post_queries() -> None:
    resources = organ_inventory_resources()

    assert len(resources) == 6
    assert sum(resource.format == "html" for resource in resources) == 4
    dynamic = [resource for resource in resources if resource.format == "json"]
    assert {resource.snapshot_token for resource in dynamic} == {
        "Leg15-type3-inventory",
        "Leg15-type4-inventory",
    }
    assert {resource.post_data["_opendata_tipoConsulta"] for resource in dynamic} == {
        "3",
        "4",
    }


def test_organ_data_plan_deduplicates_inventory_overlap_and_exports_all_formats() -> None:
    resources, descriptors = organ_data_resources(
        {
            "Leg15-type3-inventory": _inventory(("1", "301")),
            "Leg15-type4-inventory": _inventory(("1", "301"), ("358", "358201")),
        }
    )

    assert len(descriptors) == 5  # three superior organs plus two dynamic organs
    assert len(resources) == 20  # AJAX JSON plus CSV/JSON/XML for every organ
    exports = [resource for resource in resources if resource.dataset == "OrganCompositionExport"]
    assert len(exports) == 15
    by_token: dict[str, set[str]] = {}
    for resource in exports:
        by_token.setdefault(str(resource.snapshot_token), set()).add(resource.format)
        assert resource.post_data["_organos_fileType"] == resource.format
    assert all(formats == {"csv", "json", "xml"} for formats in by_token.values())


def test_organ_export_semantics_are_equal_across_csv_json_and_xml() -> None:
    json_payload = (
        b'[{"Cargo":"Presidenta","NombreOrgano":"Mesa",'
        b'"Nombre":"Persona Uno","Grupo":"G","FechaAlta":"01/01/2024",'
        b'"FechaBaja":""}]'
    )
    csv_payload = (
        b"Cargo;NombreOrgano;Nombre;Grupo;FechaAlta;FechaBaja\r\n"
        b"Presidenta;Mesa;Persona Uno;G;01/01/2024;\r\n"
    )
    xml_payload = b"""<?xml version="1.0" encoding="UTF-8"?>
    <organo><componente><Cargo>Presidenta</Cargo><NombreOrgano>Mesa</NombreOrgano>
    <Nombre>Persona Uno</Nombre><Grupo>G</Grupo><FechaAlta>01/01/2024</FechaAlta>
    <FechaBaja></FechaBaja></componente></organo>"""

    canonical = {
        _canonical_export_rows(json_payload, format_name="json")[0]["Nombre"],
        _canonical_export_rows(csv_payload, format_name="csv")[0]["Nombre"],
        _canonical_export_rows(xml_payload, format_name="xml")[0]["Nombre"],
    }

    assert canonical == {"Persona Uno"}
    json_rows = _canonical_export_rows(json_payload, format_name="json")
    csv_rows = _canonical_export_rows(csv_payload, format_name="csv")
    xml_rows = _canonical_export_rows(xml_payload, format_name="xml")
    assert json_rows == csv_rows == xml_rows


def test_organ_export_semantics_do_not_hide_whitespace_differences() -> None:
    json_payload = b'[{"Nombre":"Persona  Uno","Cargo":"Vocal"}]'
    csv_payload = b"Nombre;Cargo\r\nPersona Uno;Vocal\r\n"

    json_rows = _canonical_export_rows(json_payload, format_name="json")
    csv_rows = _canonical_export_rows(csv_payload, format_name="csv")

    assert json_rows != csv_rows


def test_organ_semantic_comparison_ignores_only_mapping_key_order() -> None:
    assert _canonical_export_groups_match(
        {
            "csv": [{"Cargo": "Presidenta", "Nombre": "Persona Uno"}],
            "json": [{"Nombre": "Persona Uno", "Cargo": "Presidenta"}],
            "xml": [{"Cargo": "Presidenta", "Nombre": "Persona Uno"}],
        }
    )
    assert not _canonical_export_groups_match(
        {
            "csv": [{"Cargo": "Presidenta", "Nombre": "Persona  Uno"}],
            "json": [{"Nombre": "Persona Uno", "Cargo": "Presidenta"}],
        }
    )

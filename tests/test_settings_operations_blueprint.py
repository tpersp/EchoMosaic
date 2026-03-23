from __future__ import annotations

from flask import Flask

from echomosaic_app.routes.settings_operations import create_settings_operations_blueprint
from echomosaic_app.services.operations import OperationsService


def test_settings_operations_blueprint_registers_expected_routes() -> None:
    app = Flask(__name__)
    app.register_blueprint(
        create_settings_operations_blueprint(
            settings={},
            load_config=lambda: {"SERVICE_NAME": "echomosaic.service", "logging": {}},
            default_ai_settings=lambda: {"model": "demo"},
            ai_fallback_defaults={"model": "fallback"},
            post_processors=[],
            max_loras=4,
            settings_export_payload=lambda: {},
            import_settings_handler=lambda: ("ok", 200),
            update_ai_defaults_handler=lambda: ("ok", 200),
            operations_service=OperationsService(
                load_config=lambda: {"SERVICE_NAME": "echomosaic.service", "logging": {}},
                backup_dirname="_backup",
                restore_point_dirname="restore_points",
                restore_point_metadata_file="metadata.json",
                max_restore_points=5,
                settings_file="settings.json",
                config_file="config.json",
                set_update_job_active=lambda active: True,
                run_update_job=lambda repo_path, update_script, service_name, socket_id: None,
                logger=type(
                    "SL",
                    (),
                    {
                        "warning": lambda *args, **kwargs: None,
                    },
                )(),
            ),
            logger=type(
                "L",
                (),
                {
                    "warning": lambda *args, **kwargs: None,
                    "exception": lambda *args, **kwargs: None,
                },
            )(),
        )
    )

    endpoints = {rule.rule for rule in app.url_map.iter_rules()}
    assert "/settings/export" in endpoints
    assert "/settings/import" in endpoints
    assert "/settings/ai-defaults" in endpoints
    assert "/settings" in endpoints
    assert "/restore_points" in endpoints
    assert "/restore_points/<point_id>" in endpoints
    assert "/restore_points/<point_id>/restore" in endpoints
    assert "/update_app" in endpoints
    assert "/update_info" in endpoints
    assert "/update_history" in endpoints
    assert "/rollback_app" in endpoints
    assert "/update" in endpoints

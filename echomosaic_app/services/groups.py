"""Group and mosaic layout service."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set


class GroupService:
    def __init__(
        self,
        *,
        settings,
        save_settings_debounced,
        safe_emit,
        maybe_int,
        clamp,
    ) -> None:
        self.settings = settings
        self.save_settings_debounced = save_settings_debounced
        self.safe_emit = safe_emit
        self.maybe_int = maybe_int
        self.clamp = clamp

    def streams_meta(self) -> Dict[str, Dict[str, Any]]:
        meta = {}
        for stream_id, conf in self.settings.items():
            if stream_id.startswith("_"):
                continue
            meta[stream_id] = {
                "label": conf.get("label", stream_id),
                "include_in_global": conf.get("include_in_global", True),
            }
        return meta

    def groups_payload(self) -> Dict[str, Any]:
        return self.settings.get("_groups", {})

    def create_group(self, data: Dict[str, Any]) -> Dict[str, Any]:
        name = (data.get("name") or "").strip()
        streams = data.get("streams") or []
        layout = data.get("layout") or None
        if not name:
            raise ValueError("Name required")
        if name.lower() == "default":
            raise RuntimeError("'default' is a reserved group name")
        self.settings.setdefault("_groups", {})
        for existing in list(self.settings["_groups"].keys()):
            if existing.lower() == name.lower() and existing != name:
                raise FileExistsError("Group name already exists (case-insensitive)")
        cleaned = [stream_id for stream_id in streams if stream_id in self.settings]
        if layout and isinstance(layout, dict):
            self.settings["_groups"][name] = {"streams": cleaned, "layout": layout}
        else:
            self.settings["_groups"][name] = cleaned
        self.save_settings_debounced()
        self.safe_emit("mosaic_refresh", {"group": name})
        return {"status": "ok", "group": {name: self.settings["_groups"][name]}}

    def delete_group(self, name: str) -> bool:
        if "_groups" in self.settings and name in self.settings["_groups"]:
            del self.settings["_groups"][name]
            self.save_settings_debounced()
            return True
        return False

    def normalize_group_layout(self, layout: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(layout, dict):
            return None
        allowed_layouts = {"grid", "focus", "pip"}
        layout_value_raw = layout.get("layout", "grid")
        layout_value = str(layout_value_raw).strip().lower() if isinstance(layout_value_raw, str) else "grid"
        if layout_value not in allowed_layouts:
            layout_value = "grid"

        def _bounded_int(value: Any, lower: int, upper: int) -> Optional[int]:
            maybe = self.maybe_int(value)
            if maybe is None:
                return None
            return int(self.clamp(maybe, lower, upper))

        sanitized: Dict[str, Any] = {"layout": layout_value}
        sanitized["cols"] = _bounded_int(layout.get("cols"), 1, 8)
        sanitized["rows"] = _bounded_int(layout.get("rows"), 1, 8)

        focus_mode = layout.get("focus_mode") if isinstance(layout.get("focus_mode"), str) else None
        if focus_mode not in {"1-2", "1-3", "1-5"}:
            focus_mode = "1-5"
        focus_pos = layout.get("focus_pos") if isinstance(layout.get("focus_pos"), str) else None
        allowed_focus_pos = {"left", "right", "top", "bottom", "top-left", "top-right", "bottom-left", "bottom-right"}
        if focus_pos not in allowed_focus_pos:
            focus_pos = "bottom-right" if focus_mode == "1-5" else ("right" if focus_mode == "1-2" else "bottom")
        focus_main = layout.get("focus_main") if isinstance(layout.get("focus_main"), str) else None
        sanitized["focus_mode"] = focus_mode
        sanitized["focus_pos"] = focus_pos
        sanitized["focus_main"] = focus_main

        pip_main = layout.get("pip_main") if isinstance(layout.get("pip_main"), str) else None
        pip_pip = layout.get("pip_pip") if isinstance(layout.get("pip_pip"), str) else None
        pip_corner = layout.get("pip_corner") if isinstance(layout.get("pip_corner"), str) else None
        if pip_corner not in {"top-left", "top-right", "bottom-left", "bottom-right"}:
            pip_corner = "bottom-right"
        pip_size = _bounded_int(layout.get("pip_size"), 10, 50) or 25
        sanitized["pip_main"] = pip_main
        sanitized["pip_pip"] = pip_pip
        sanitized["pip_corner"] = pip_corner
        sanitized["pip_size"] = pip_size
        return sanitized

    def build_group_view_model(self, name: str) -> Dict[str, Any]:
        groups = self.settings.get("_groups", {})
        group_def = groups.get(name)
        if not group_def and name.lower() == "default":
            group_def = [key for key in self.settings.keys() if not key.startswith("_")]
        if not group_def:
            raise KeyError(name)

        layout_conf: Optional[Dict[str, Any]] = None
        if isinstance(group_def, dict):
            members = group_def.get("streams", [])
            layout_conf = self.normalize_group_layout(group_def.get("layout"))
        else:
            members = list(group_def)

        member_ids = [member for member in members if member in self.settings]
        unique_ids: List[str] = []
        seen_ids: Set[str] = set()
        for stream_id in member_ids:
            if stream_id not in seen_ids:
                seen_ids.add(stream_id)
                unique_ids.append(stream_id)
        stream_lookup = {stream_id: self.settings[stream_id] for stream_id in unique_ids}
        stream_entries = [(stream_id, stream_lookup[stream_id]) for stream_id in member_ids]
        focus_order = list(member_ids)
        if layout_conf and layout_conf.get("layout") == "focus":
            focus_main_id = layout_conf.get("focus_main")
            if focus_main_id in stream_lookup:
                reordered: List[str] = [focus_main_id]
                skipped = False
                for stream_id in member_ids:
                    if stream_id == focus_main_id and not skipped:
                        skipped = True
                        continue
                    reordered.append(stream_id)
                focus_order = reordered
        return {
            "stream_settings": stream_lookup,
            "stream_entries": stream_entries,
            "stream_order": member_ids,
            "unique_stream_ids": unique_ids,
            "focus_order": focus_order,
            "mosaic_settings": layout_conf,
        }

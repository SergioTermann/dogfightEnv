# -*- coding: utf-8 -*-
"""Decision engines for the mission commander.

Both engines share one contract: decide(situation, last_assignments) returns
    {"assignments": [{"plane": str, "task": str, "target": str|None,
                      "heading": deg|None, "altitude": m|None, "speed": m/s|None}, ...],
     "reason": str}
or None when no usable decision could be produced (the commander then keeps the
previous assignments).

- RuleTactician: deterministic nearest-enemy pairing with retreat/patrol logic.
  Zero dependencies, used as the default demo engine and as the LLM fallback.
- LLMTactician: asks an OpenAI-compatible chat/completions endpoint (Zhipu GLM,
  OpenAI, DeepSeek, local vLLM/Ollama with an OpenAI adapter, ...) to do the
  allocation. Only uses urllib from the standard library so it also runs on the
  sandbox's embedded Python.
"""

import json
import math
import re
import urllib.request

TASK_ENGAGE = "engage"
TASK_PATROL = "patrol"
TASK_RETREAT = "retreat"
TASK_HOLD = "hold"

RETREAT_HEALTH = 0.35      # below this the plane disengages
PATROL_ALT_M = 1500.0
PATROL_SPEED_MS = 200.0
RETREAT_SPEED_MS = 260.0
RETREAT_ALT_M = 800.0


def bearing_deg(from_pos, to_pos):
    """Compass bearing (0=north/+Z, 90=east/+X) from one world position to another."""
    return math.degrees(math.atan2(to_pos[0] - from_pos[0], to_pos[2] - from_pos[2])) % 360.0


def distance_m(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def plane_alive(state):
    return (state is not None and not state.get("wreck") and not state.get("crashed")
            and not state.get("destroyed") and state.get("health_level", 0) > 0)


def build_situation(states, missiles, own_side, sim_time_s=0.0):
    """Compact battlefield summary handed to the decision engines.

    states: {plane_name: get_plane_state dict}; missiles: {plane_name: remaining count};
    own_side: "allies" or "ennemies".
    """
    planes = []
    for name in sorted(states):
        st = states[name]
        pos = st.get("position", [0, 0, 0])
        planes.append({
            "name": name,
            "side": "blue" if st.get("nationality") == 1 else "red",
            "alive": plane_alive(st),
            "health": round(float(st.get("health_level", 0.0)), 2),
            "missiles": int(missiles.get(name, 0)),
            "pos_km": [round(pos[0] / 1000.0, 1), round(pos[1] / 1000.0, 1), round(pos[2] / 1000.0, 1)],
            "altitude_m": round(float(st.get("altitude", 0.0))),
            "speed_ms": round(float(st.get("linear_speed", 0.0))),
            "heading_deg": round(float(st.get("heading", 0.0))) % 360,
            "target": st.get("target_id") or None,
            "locked": bool(st.get("target_locked")),
        })

    own_tag = "blue" if own_side == "allies" else "red"
    distances_km = {}
    for a in planes:
        for b in planes:
            if a["name"] < b["name"] and a["side"] != b["side"]:
                d = distance_m(states[a["name"]].get("position", [0, 0, 0]),
                               states[b["name"]].get("position", [0, 0, 0])) / 1000.0
                distances_km["%s<->%s" % (a["name"], b["name"])] = round(d, 1)

    return {"sim_time_s": round(sim_time_s, 1), "own_side": own_tag,
            "planes": planes, "distances_km": distances_km}


def _live_opponents(situation, own_plane):
    own_tag = own_plane["side"]
    return [p for p in situation["planes"] if p["side"] != own_tag and p["alive"]]


class RuleTactician:
    """Deterministic tactician: 1v1 nearest-enemy pairing with conflict spreading.

    - healthy plane with missiles -> engage the nearest live opponent; targets are
      spread (each opponent is claimed by at most one plane while unclaimed ones remain)
    - damaged (< RETREAT_HEALTH) or out of missiles -> retreat away from nearest opponent
    - no live opponent -> patrol toward the map center
    """

    name = "rule"

    def decide(self, situation, last_assignments=None):
        own = [p for p in situation["planes"] if p["side"] == situation["own_side"] and p["alive"]]
        opponents = [p for p in situation["planes"] if p["side"] != situation["own_side"] and p["alive"]]
        assignments = []

        # healthy fighters first so they get the unclaimed targets
        fighters = [p for p in own if p["health"] >= RETREAT_HEALTH and p["missiles"] > 0]

        claimed = set()
        for p in fighters:
            if not opponents:
                break
            unclaimed = [o for o in opponents if o["name"] not in claimed]
            if unclaimed:
                target = min(unclaimed, key=lambda o: self._pair_distance(situation, p, o))
                claimed.add(target["name"])
            else:  # outnumbered: concentrate on the nearest
                target = min(opponents, key=lambda o: self._pair_distance(situation, p, o))
            assignments.append({"plane": p["name"], "task": TASK_ENGAGE, "target": target["name"],
                                "heading": None, "altitude": None, "speed": None})

        for p in own:
            if any(a["plane"] == p["name"] for a in assignments):
                continue
            nearest = None
            if opponents:
                nearest = min(opponents, key=lambda o: self._pair_distance(situation, p, o))
            if nearest is not None:
                away = bearing_deg(nearest["pos_km"], p["pos_km"])  # heading moving away from the threat
                assignments.append({"plane": p["name"], "task": TASK_RETREAT, "target": None,
                                    "heading": round(away), "altitude": RETREAT_ALT_M,
                                    "speed": RETREAT_SPEED_MS})
            else:
                heading = bearing_deg(p["pos_km"], [0.0, 0.0, 0.0])
                assignments.append({"plane": p["name"], "task": TASK_PATROL, "target": None,
                                    "heading": round(heading), "altitude": PATROL_ALT_M,
                                    "speed": PATROL_SPEED_MS})

        n_engage = sum(1 for a in assignments if a["task"] == TASK_ENGAGE)
        return {"assignments": assignments,
                "reason": "rule: %d engage / %d retreat-or-patrol" % (n_engage, len(assignments) - n_engage)}

    @staticmethod
    def _pair_distance(situation, a, b):
        return situation["distances_km"].get("%s<->%s" % (a["name"], b["name"]),
                                             situation["distances_km"].get("%s<->%s" % (b["name"], a["name"]), 1e9))


SYSTEM_PROMPT = u"""你是空战任务指挥官，负责为己方每架战机分配任务。

可用任务（task 字段）：
- engage: 攻击指定敌机，必须给 target（存活敌机的名字）
- patrol: 沿指定航向巡逻待战，可给 heading（度，0=北 90=东）、altitude（米）、speed（米/秒）
- retreat: 朝指定航向高速脱离，可给 heading、altitude、speed
- hold: 保持现状，什么都不改

输出严格 JSON，不要任何其他文字：
{"assignments": [{"plane": "飞机名", "task": "engage", "target": "敌机名"}, ...], "reason": "一句话战术理由"}

战术要求：
1. 每架存活的己方飞机恰好分配一条任务，不要遗漏也不要编造飞机名
2. engage 的 target 必须是当前存活的敌机名；尽量分散火力避免全部堆在同一目标，除非能速歼
3. 导弹打完或血量低于0.35的飞机应 retreat 脱离保存战力
4. 敌机全灭后用 patrol 收拢队形（朝地图中心方向）
5. heading 取值 0-359（0=北，90=东），altitude 300-8000，speed 120-320"""


class LLMTactician:
    """OpenAI-compatible chat/completions tactician (Zhipu GLM by default)."""

    name = "llm"

    def __init__(self, api_base, api_key, model, temperature=0.2, timeout_s=25.0):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout_s = timeout_s

    def decide(self, situation, last_assignments=None):
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._user_content(situation, last_assignments)},
            ],
        }
        try:
            content = self._post(payload)
            return self._parse(content, situation)
        except Exception as exc:  # network / API / parse errors -> keep previous plan
            print("[llm] decision failed: %r" % (exc,))
            return None

    def _user_content(self, situation, last_assignments):
        msg = u"当前战况（JSON）：\n" + json.dumps(situation, ensure_ascii=False)
        if last_assignments:
            msg += u"\n上一轮分配（供参考，可以调整）：\n" + json.dumps(last_assignments, ensure_ascii=False)
        msg += u"\n请输出本轮任务分配 JSON。"
        return msg

    def _post(self, payload):
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = urllib.request.Request(self.api_base, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        if "error" in body:
            raise RuntimeError(body["error"])
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError("empty choices in LLM response")
        return choices[0].get("message", {}).get("content", "")

    @staticmethod
    def _parse(content, situation):
        """Extract the JSON object from a (possibly fenced / chatty) reply and
        drop every assignment that doesn't validate against the situation."""
        text = content.strip()
        fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
        if fence:
            text = fence.group(1).strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object in reply")
        obj = json.loads(text[start:end + 1])

        own = {p["name"] for p in situation["planes"]
               if p["side"] == situation["own_side"] and p["alive"]}
        foes = {p["name"] for p in situation["planes"]
                if p["side"] != situation["own_side"] and p["alive"]}

        assignments = []
        for a in obj.get("assignments", []):
            try:
                plane = str(a["plane"])
                task = str(a["task"])
            except (KeyError, TypeError):
                continue
            if plane not in own or task not in (TASK_ENGAGE, TASK_PATROL, TASK_RETREAT, TASK_HOLD):
                continue
            item = {"plane": plane, "task": task, "target": None,
                    "heading": None, "altitude": None, "speed": None}
            if task == TASK_ENGAGE:
                target = str(a.get("target") or "")
                if target not in foes:
                    continue  # hallucinated or dead target -> drop
                item["target"] = target
            if task in (TASK_PATROL, TASK_RETREAT):
                for key, lo, hi in (("heading", 0, 359), ("altitude", 300, 8000), ("speed", 120, 320)):
                    val = a.get(key)
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        item[key] = int(max(lo, min(hi, val)))
            assignments.append(item)

        if not assignments:
            raise ValueError("no valid assignments in LLM reply")
        # keep exactly one task per plane (last one wins)
        dedup = {a["plane"]: a for a in assignments}
        return {"assignments": [dedup[n] for n in sorted(dedup)],
                "reason": str(obj.get("reason", ""))[:200]}

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def _load_q2_module() -> Any:
    base_dir = Path(__file__).resolve().parent
    path = base_dir / "Q2-solution.py"
    spec = importlib.util.spec_from_file_location("_q2_solution", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 Q2-solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


q2 = _load_q2_module()


@dataclass(frozen=True)
class Event:
    time_min: int
    type: str
    payload: Dict[str, Any]


def _parse_time_to_min(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    s = str(value).strip()
    if not s:
        raise ValueError("时间为空")
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return int(s)
    if ":" in s:
        return int(_hhmm_to_minute(s))
    raise ValueError(f"无法解析时间: {value!r}")


def _coerce_payload_value(v: str) -> Any:
    s = v.strip()
    if not s:
        return ""
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    try:
        if any(ch in s for ch in (".", "e", "E")):
            return float(s)
        return int(s)
    except ValueError:
        return s


def parse_event_line(line: str) -> Event:
    raw = line.strip()
    if not raw:
        raise ValueError("空事件")
    if raw.startswith("{"):
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("JSON 事件必须是对象")
        if "time_min" in obj:
            time_min = _parse_time_to_min(obj["time_min"])
        elif "time" in obj:
            time_min = _parse_time_to_min(obj["time"])
        else:
            raise ValueError("JSON 事件缺少 time_min/time")
        etype = str(obj.get("type", "")).strip()
        if not etype:
            raise ValueError("JSON 事件缺少 type")
        payload = obj.get("payload", {})
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise ValueError("payload 必须是对象")
        return validate_event(Event(time_min=time_min, type=etype, payload=dict(payload)))

    parts = raw.split()
    if len(parts) < 2:
        raise ValueError("事件格式应为: <time> <type> [k=v ...]")
    time_min = _parse_time_to_min(parts[0])
    etype = parts[1].strip()

    rest = raw[len(parts[0]) :].lstrip()
    rest = rest[len(parts[1]) :].lstrip()
    if rest.startswith("{"):
        obj = json.loads(rest)
        if not isinstance(obj, dict):
            raise ValueError("行内 JSON payload 必须是对象")
        return validate_event(Event(time_min=time_min, type=etype, payload=dict(obj)))

    payload: Dict[str, Any] = {}
    positional: List[str] = []
    for token in parts[2:]:
        t = token.strip().rstrip(",")
        if not t:
            continue
        if "=" in t:
            k, v = t.split("=", 1)
            payload[k.strip()] = _coerce_payload_value(v)
        elif ":" in t:
            k, v = t.split(":", 1)
            payload[k.strip()] = _coerce_payload_value(v)
        else:
            positional.append(t)

    if not payload and positional:
        if etype == "order_cancel" and len(positional) >= 1:
            payload["order_id"] = _coerce_payload_value(positional[0])
        elif etype == "order_add" and len(positional) >= 4:
            payload["order_id"] = _coerce_payload_value(positional[0])
            payload["customer_id"] = _coerce_payload_value(positional[1])
            payload["weight_kg"] = _coerce_payload_value(positional[2])
            payload["volume_m3"] = _coerce_payload_value(positional[3])
        elif etype == "customer_address_change" and len(positional) >= 3:
            payload["customer_id"] = _coerce_payload_value(positional[0])
            payload["x_km"] = _coerce_payload_value(positional[1])
            payload["y_km"] = _coerce_payload_value(positional[2])
        elif etype == "customer_timewindow_change" and len(positional) >= 3:
            payload["customer_id"] = _coerce_payload_value(positional[0])
            payload["tw_start"] = positional[1]
            payload["tw_end"] = positional[2]

    return validate_event(Event(time_min=time_min, type=etype, payload=payload))


def parse_events_text(text: str) -> List[Event]:
    """解析事件文件。

    支持两种格式：
    1) JSONL / 普通行格式：每行一个事件；
    2) JSON 数组：[{"time":"10:30", "type":"order_add", ...}, ...]。

    以 # 开头的空白外注释行会被忽略。
    """
    raw = text.strip()
    if not raw:
        return []

    if raw.startswith("["):
        arr = json.loads(raw)
        if not isinstance(arr, list):
            raise ValueError("JSON 数组事件文件的顶层必须是 list")
        events: List[Event] = []
        for obj in arr:
            if not isinstance(obj, dict):
                raise ValueError("JSON 数组中的每个元素都必须是事件对象")
            events.append(parse_event_line(json.dumps(obj, ensure_ascii=False)))
        return events

    events = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        events.append(parse_event_line(s))
    return events


def group_events_for_replanning(events: Sequence[Event], mode: str = "batch") -> List[List[Event]]:
    """将事件划分为重规划批次。

    - batch：所有事件作为同一批，在 max(time_min) 时刻一次性修改订单池并只重规划一次。
      适用于“同时新增 5 个订单后统一排序、统一总成本”的场景。
    - by-time：按事件时间分组；同一时刻的事件批量重规划一次，不同时刻按时间先后分多次。
    - sequential：旧逻辑；每个事件单独重规划一次。
    """
    events = list(events)
    if not events:
        return []
    mode = str(mode or "batch").strip().lower()
    if mode == "batch":
        return [events]
    if mode == "sequential":
        return [[e] for e in events]
    if mode == "by-time":
        groups: Dict[int, List[Event]] = {}
        first_pos: Dict[int, int] = {}
        for pos, e in enumerate(events):
            t = int(e.time_min)
            groups.setdefault(t, []).append(e)
            first_pos.setdefault(t, pos)
        # 按事件时间排序；同一时间内保持文件原顺序。
        return [groups[t] for t in sorted(groups, key=lambda x: (x, first_pos[x]))]
    raise ValueError("event-mode 必须为 batch、by-time 或 sequential")


def _event_group_label(group: Sequence[Event]) -> str:
    if not group:
        return "空事件组"
    ts = sorted({_minute_to_hhmm(e.time_min) for e in group})
    types = "+".join(e.type for e in group)
    return f"{len(group)}个事件, time={','.join(ts)}, types={types}"


def validate_event(e: Event) -> Event:
    required: Dict[str, Tuple[str, ...]] = {
        "order_cancel": ("order_id",),
        "order_add": ("order_id", "customer_id", "weight_kg", "volume_m3"),
        "customer_address_change": ("customer_id", "x_km", "y_km"),
        "customer_timewindow_change": ("customer_id", "tw_start", "tw_end"),
    }
    if e.type not in required:
        raise ValueError(f"未知事件类型: {e.type}")
    miss = [k for k in required[e.type] if k not in e.payload]
    if miss:
        raise ValueError(f"事件 {e.type} 缺少字段: {miss}")
    return e


def _hhmm_to_minute(s: str) -> int:
    return q2.hhmm_to_minute(s)


def _minute_to_hhmm(x: float) -> str:
    return q2.minute_to_hhmm(x)


def load_base_data(base_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    orders = pd.read_excel(base_dir / q2.ORDER_FILE, sheet_name="完整订单数据").rename(
        columns={"订单编号": "order_id", "目标客户编号": "customer_id", "重量": "weight_kg", "体积": "volume_m3"}
    )
    orders["order_id"] = orders["order_id"].astype(int)
    orders["customer_id"] = orders["customer_id"].astype(int)
    orders = q2.complete_order_weight_volume(orders)

    dr = pd.read_excel(base_dir / q2.DIST_FILE)
    dr = dr.rename(columns={dr.columns[0]: "from_id"})
    dr.columns = ["from_id"] + [int(c) for c in dr.columns[1:]]
    dist_df = dr.set_index("from_id")
    dist_df.index = dist_df.index.astype(int)
    dist_df.columns = dist_df.columns.astype(int)

    coords_df = pd.read_excel(base_dir / q2.COORD_FILE).rename(
        columns={"ID": "customer_id", "X (km)": "x_km", "Y (km)": "y_km", "类型": "type"}
    )
    coords_df["customer_id"] = coords_df["customer_id"].astype(int)
    return orders, dist_df, coords_df


def load_time_windows(base_dir: Path) -> pd.DataFrame:
    tw = pd.read_excel(base_dir / q2.TW_FILE).rename(
        columns={"客户编号": "customer_id", "开始时间": "tw_start", "结束时间": "tw_end"}
    )
    tw["customer_id"] = tw["customer_id"].astype(int)
    tw["tw_start_min"] = tw["tw_start"].apply(q2.hhmm_to_minute)
    tw["tw_end_min"] = tw["tw_end"].apply(q2.hhmm_to_minute)
    return tw


def build_customer_info(coords_df: pd.DataFrame, tw_df: pd.DataFrame) -> pd.DataFrame:
    df = coords_df.merge(
        tw_df[["customer_id", "tw_start", "tw_end", "tw_start_min", "tw_end_min"]],
        on="customer_id",
        how="left",
    )
    depot_id = int(q2.DEPOT_ID)
    m_depot = df["customer_id"].astype(int) == depot_id
    df.loc[m_depot, "tw_start_min"] = int(q2.START_OF_DAY)
    df.loc[m_depot, "tw_end_min"] = int(q2.END_OF_DAY)
    df.loc[m_depot, "tw_start"] = _minute_to_hhmm(int(q2.START_OF_DAY))
    df.loc[m_depot, "tw_end"] = _minute_to_hhmm(int(q2.END_OF_DAY))

    miss_mask = df[["tw_start_min", "tw_end_min"]].isna().any(axis=1) & (~m_depot)
    if miss_mask.any():
        missing = df.loc[miss_mask, "customer_id"].tolist()
        raise ValueError(f"存在缺失时间窗的客户: {missing[:10]}")
    return df.set_index("customer_id", drop=False)


def _order_sort_key(row: pd.Series, w_cap: float, v_cap: float) -> float:
    return float(max(float(row["weight_kg"]) / w_cap, float(row["volume_m3"]) / v_cap))


def pack_orders_to_customer_batches(
    orders_df: pd.DataFrame,
    customer_info: pd.DataFrame,
    w_cap: float,
    v_cap: float,
) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, List[int]]]:
    rows: List[Dict[str, Any]] = []
    order_to_batch: Dict[int, int] = {}
    customer_to_batches: Dict[int, List[int]] = {}

    grouped = orders_df.groupby("customer_id", sort=True)
    for customer_id, g in grouped:
        customer_id = int(customer_id)
        if customer_id == int(q2.DEPOT_ID):
            continue
        g = g.copy()
        g = g[(g["weight_kg"] > 1e-9) & (g["volume_m3"] > 1e-12)]
        if g.empty:
            continue

        cust = customer_info.loc[customer_id]
        g["_k"] = g.apply(lambda r: _order_sort_key(r, w_cap, v_cap), axis=1)
        g = g.sort_values(["_k", "weight_kg", "volume_m3"], ascending=[False, False, False])

        bins: List[Dict[str, Any]] = []
        for _, r in g.iterrows():
            oid = int(r["order_id"])
            w = float(r["weight_kg"])
            v = float(r["volume_m3"])
            placed = False
            for b in bins:
                if b["w"] + w <= w_cap + 1e-9 and b["v"] + v <= v_cap + 1e-9:
                    b["w"] += w
                    b["v"] += v
                    b["orders"].append(oid)
                    placed = True
                    break
            if not placed:
                bins.append({"w": w, "v": v, "orders": [oid]})

        split_count = len(bins)
        for k, b in enumerate(bins, start=1):
            batch_id = len(rows) + 1
            customer_to_batches.setdefault(customer_id, []).append(batch_id)
            for oid in b["orders"]:
                order_to_batch[int(oid)] = int(batch_id)
            cap_frac = max(b["w"] / w_cap, b["v"] / v_cap)
            rows.append(
                {
                    "batch_id": batch_id,
                    "customer_id": customer_id,
                    "split_index": k,
                    "split_count": split_count,
                    "batch_kind": "packed",
                    "limiting_dimension": "weight" if (b["w"] / w_cap >= b["v"] / v_cap) else "volume",
                    "capacity_fraction": cap_frac,
                    "weight_kg": float(b["w"]),
                    "volume_m3": float(b["v"]),
                    "load_ratio_fuel": max(float(b["w"]) / float(q2.VEHICLE_WEIGHT_CAP), float(b["v"]) / float(q2.FUEL_VOLUME_CAP)),
                    "x_km": float(cust["x_km"]),
                    "y_km": float(cust["y_km"]),
                    "tw_start_min": int(cust["tw_start_min"]),
                    "tw_end_min": int(cust["tw_end_min"]),
                    "tw_start": str(cust["tw_start"]),
                    "tw_end": str(cust["tw_end"]),
                    "order_ids": tuple(int(x) for x in b["orders"]),
                }
            )

    batch_df = pd.DataFrame(rows)
    return batch_df, order_to_batch, customer_to_batches


def build_batch_lookup(batch_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    return batch_df.set_index("batch_id").to_dict("index")


def _route_customers(seq: Sequence[int], batch_lookup: Dict[int, Dict[str, Any]]) -> List[int]:
    return [int(batch_lookup[bid]["customer_id"]) for bid in seq]


def route_feasible(seq: Sequence[int], batch_lookup: Dict[int, Dict[str, Any]]) -> bool:
    if len(seq) > int(q2.MAX_ROUTE_STOPS):
        return False
    custs = _route_customers(seq, batch_lookup)
    if len(set(custs)) != len(custs):
        return False
    w, v = q2.route_load(seq, batch_lookup)
    if w > float(q2.VEHICLE_WEIGHT_CAP) + 1e-9:
        return False
    if v > float(q2.FUEL_VOLUME_CAP) + 1e-9:
        return False
    return True


def _penalty(eva: Dict[str, Any]) -> float:
    return float(q2.penalty_score(eva) + 5.0 * eva["late_min"])


def build_initial_routes_randomized(
    batch_df: pd.DataFrame,
    batch_lookup: Dict[int, Dict[str, Any]],
    dist_df: pd.DataFrame,
    green_zone_customers: Set[int],
    rng: np.random.Generator,
) -> List[List[int]]:
    df = batch_df.copy()
    noise = rng.normal(0.0, 12.0, size=len(df))
    df["_key"] = df["tw_start_min"].astype(float) + noise
    df["_is_gz"] = df["customer_id"].astype(int).isin(green_zone_customers).astype(int)
    df = df.sort_values(["_is_gz", "_key"], ascending=[False, True])
    order = df["batch_id"].astype(int).tolist()

    routes: List[List[int]] = []
    for bid in order:
        b = batch_lookup[int(bid)]
        cid = int(b["customer_id"])
        is_gz = cid in green_zone_customers
        best: Optional[Tuple[float, int, List[int]]] = None
        for ridx, seq in enumerate(routes):
            if cid in _route_customers(seq, batch_lookup):
                continue
            if not route_feasible(seq + [int(bid)], batch_lookup):
                continue
            seq_has_gz = any(int(batch_lookup[x]["customer_id"]) in green_zone_customers for x in seq)
            if is_gz and not seq_has_gz:
                gz_count = sum(
                    1
                    for s in routes
                    if any(int(batch_lookup[x]["customer_id"]) in green_zone_customers for x in s)
                )
                if gz_count < int(q2.EV_FLEET_LIMIT):
                    continue
            be = q2.evaluate_route(seq, batch_lookup, dist_df)
            bs = _penalty(be)
            for pos in range(len(seq) + 1):
                ns = seq[:pos] + [int(bid)] + seq[pos:]
                if not route_feasible(ns, batch_lookup):
                    continue
                ne = q2.evaluate_route(ns, batch_lookup, dist_df, start_minute=be["start_min"])
                sc = (float(_penalty(ne) - bs)) + 8.0 * float(ne["late_min"] - be["late_min"])
                if best is None or sc < best[0]:
                    best = (sc, ridx, ns)
        if best is None:
            routes.append([int(bid)])
        else:
            routes[best[1]] = best[2]
    return routes


def refine_routes(routes: List[List[int]], batch_lookup: Dict[int, Dict[str, Any]], dist_df: pd.DataFrame) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
    refined_routes: List[List[int]] = []
    refined_evals: List[Dict[str, Any]] = []
    for seq in routes:
        bs, _ = q2.refine_single_route(seq, batch_lookup, dist_df)
        refined_routes.append(bs)
        refined_evals.append(q2.evaluate_route(bs, batch_lookup, dist_df, final_search=True))
    return refined_routes, refined_evals


def _best_insertion_for_batch(
    routes: List[List[int]],
    bid: int,
    batch_lookup: Dict[int, Dict[str, Any]],
    dist_df: pd.DataFrame,
) -> Tuple[Optional[int], Optional[int], float, Optional[List[int]]]:
    cid = int(batch_lookup[bid]["customer_id"])
    best_ridx: Optional[int] = None
    best_pos: Optional[int] = None
    best_delta = float("inf")
    best_seq: Optional[List[int]] = None
    for ridx, seq in enumerate(routes):
        if cid in _route_customers(seq, batch_lookup):
            continue
        if not route_feasible(seq + [bid], batch_lookup):
            continue
        be = q2.evaluate_route(seq, batch_lookup, dist_df)
        bs = _penalty(be)
        for pos in range(len(seq) + 1):
            ns = seq[:pos] + [bid] + seq[pos:]
            if not route_feasible(ns, batch_lookup):
                continue
            ne = q2.evaluate_route(ns, batch_lookup, dist_df, start_minute=be["start_min"])
            delta = _penalty(ne) - bs
            if delta < best_delta:
                best_delta = float(delta)
                best_ridx = ridx
                best_pos = pos
                best_seq = ns
    return best_ridx, best_pos, best_delta, best_seq


def local_relocation_search_unique(
    routes: List[List[int]],
    batch_lookup: Dict[int, Dict[str, Any]],
    dist_df: pd.DataFrame,
    max_iter: int = 3,
) -> List[List[int]]:
    routes = [r[:] for r in routes if r]
    it = 0
    while it < int(max_iter):
        it += 1
        improved = False
        evs = [q2.evaluate_route(s, batch_lookup, dist_df) for s in routes]
        scs = [_penalty(e) for e in evs]
        best_move: Optional[Tuple[float, int, int, int, List[int], List[int]]] = None
        for i in range(len(routes)):
            for pi, bid in enumerate(routes[i]):
                if len(routes[i]) <= 1:
                    continue
                src_seq = routes[i][:pi] + routes[i][pi + 1 :]
                if not route_feasible(src_seq, batch_lookup):
                    continue
                src_e = q2.evaluate_route(src_seq, batch_lookup, dist_df)
                src_sc = _penalty(src_e)
                gained = scs[i] - src_sc

                cid = int(batch_lookup[int(bid)]["customer_id"])
                for j in range(len(routes)):
                    if j == i:
                        continue
                    if cid in _route_customers(routes[j], batch_lookup):
                        continue
                    if not route_feasible(routes[j] + [int(bid)], batch_lookup):
                        continue
                    base_e = evs[j]
                    base_sc = scs[j]
                    for pos in range(len(routes[j]) + 1):
                        cand = routes[j][:pos] + [int(bid)] + routes[j][pos:]
                        if not route_feasible(cand, batch_lookup):
                            continue
                        cand_e = q2.evaluate_route(cand, batch_lookup, dist_df, start_minute=base_e["start_min"])
                        cand_sc = _penalty(cand_e)
                        delta = gained - (cand_sc - base_sc)
                        if best_move is None or delta > best_move[0]:
                            best_move = (float(delta), i, j, pi, src_seq, cand)

        if best_move is not None and best_move[0] > 1e-6:
            _, i, j, _, src_seq, dst_seq = best_move
            routes[i] = src_seq
            routes[j] = dst_seq
            routes = [r for r in routes if r]
            improved = True

        if not improved:
            break
    return routes


def lns_improve(
    routes: List[List[int]],
    batch_lookup: Dict[int, Dict[str, Any]],
    dist_df: pd.DataFrame,
    rng: np.random.Generator,
    iters: int,
    remove_frac: float,
) -> List[List[int]]:
    all_bids = [bid for r in routes for bid in r]
    if not all_bids:
        return routes
    best_routes = [r[:] for r in routes]
    best_score = sum(_penalty(q2.evaluate_route(r, batch_lookup, dist_df)) for r in best_routes)

    for _ in range(int(iters)):
        cur = [r[:] for r in best_routes]
        bids = [bid for r in cur for bid in r]
        k = max(1, int(round(len(bids) * float(remove_frac))))
        removed = rng.choice(bids, size=min(k, len(bids)), replace=False).tolist()
        removed_set = set(int(x) for x in removed)
        cur = [[bid for bid in r if bid not in removed_set] for r in cur]
        cur = [r for r in cur if r]

        for bid in removed:
            bid = int(bid)
            ridx, _, _, seq = _best_insertion_for_batch(cur, bid, batch_lookup, dist_df)
            if ridx is None or seq is None:
                cur.append([bid])
            else:
                cur[ridx] = seq

        cur = local_relocation_search_unique(cur, batch_lookup, dist_df, max_iter=3)
        cur_score = sum(_penalty(q2.evaluate_route(r, batch_lookup, dist_df)) for r in cur)
        if cur_score < best_score - 1e-9:
            best_routes = [r[:] for r in cur]
            best_score = float(cur_score)
    return best_routes


def score_with_full_cost(
    routes: List[List[int]],
    route_evals: List[Dict[str, Any]],
    batch_lookup: Dict[int, Dict[str, Any]],
    dist_df: pd.DataFrame,
    green_zone_customers: Set[int],
) -> Tuple[float, pd.DataFrame, List[Dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    assign_df, route_evals2, _ = q2.assign_vehicle_types_q2(routes, route_evals, batch_lookup, dist_df, green_zone_customers)
    rsum_df, stop_df, leg_df = q2.build_result_tables(routes, route_evals2, assign_df, batch_lookup, green_zone_customers)
    chain_df, vcount = q2.assign_physical_vehicles(rsum_df)
    for vt, spec in q2.FLEET_SPECS.items():
        n = int(vcount.get(vt, 0))
        if n > int(spec["limit"]):
            raise RuntimeError(f"{vt} 实体车辆数 {n} 超过上限 {spec['limit']}")
    rsum_df = q2.add_vehicle_costs(rsum_df, chain_df)
    total_cost = float(rsum_df["total_cost"].sum())
    return total_cost, assign_df, route_evals2, rsum_df, stop_df, leg_df, chain_df, vcount


def optimize_static(
    orders_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    coords_df: pd.DataFrame,
    tw_df: pd.DataFrame,
    seed: int = 0,
    n_starts: int = 6,
    lns_iters: int = 40,
) -> Dict[str, Any]:
    customer_info = build_customer_info(coords_df, tw_df)
    green = q2.build_green_zone_customers(coords_df)

    batch_df, order_to_batch, customer_to_batches = pack_orders_to_customer_batches(
        orders_df, customer_info, float(q2.VEHICLE_WEIGHT_CAP), float(q2.FUEL_VOLUME_CAP)
    )
    batch_lookup = build_batch_lookup(batch_df)

    best: Optional[Dict[str, Any]] = None
    best_cost = float("inf")
    rng = np.random.default_rng(int(seed))

    for _ in range(int(n_starts)):
        routes = build_initial_routes_randomized(batch_df, batch_lookup, dist_df, green, rng)
        routes = local_relocation_search_unique(routes, batch_lookup, dist_df, max_iter=3)
        routes = lns_improve(routes, batch_lookup, dist_df, rng, iters=int(lns_iters), remove_frac=0.18)
        routes, route_evals = refine_routes(routes, batch_lookup, dist_df)
        try:
            total_cost, assign_df, route_evals2, rsum_df, stop_df, leg_df, chain_df, vcount = score_with_full_cost(
                routes, route_evals, batch_lookup, dist_df, green
            )
        except Exception:
            continue
        if total_cost < best_cost - 1e-9:
            best_cost = total_cost
            best = {
                "total_cost": total_cost,
                "routes": routes,
                "route_evals": route_evals2,
                "assign_df": assign_df,
                "route_summary_df": rsum_df,
                "stop_schedule_df": stop_df,
                "leg_df": leg_df,
                "vehicle_chain_df": chain_df,
                "vehicle_count": vcount,
                "batch_df": batch_df,
                "batch_lookup": batch_lookup,
                "green_zone_customers": green,
                "order_to_batch": order_to_batch,
                "customer_to_batches": customer_to_batches,
                "customer_info": customer_info,
            }

    if best is None:
        raise RuntimeError("未能生成可行解（可能是车队上限或时间窗过紧导致）。")
    return best


def order_status_from_solution(
    order_id: int,
    t: int,
    route_summary_df: pd.DataFrame,
    stop_df: pd.DataFrame,
    order_to_batch: Dict[int, int],
) -> str:
    order_id = int(order_id)
    if order_id not in order_to_batch:
        return "不存在"
    bid = int(order_to_batch[order_id])
    s = stop_df[stop_df["batch_id"].astype(int) == bid]
    if s.empty:
        return "不存在"
    leave = float(s["leave_min"].iloc[0])
    rid = int(s["route_id"].iloc[0])
    rs = route_summary_df[route_summary_df["route_id"].astype(int) == rid]
    start = float(rs["start_min"].iloc[0]) if not rs.empty else float(s["arrival_min"].iloc[0])
    if t < start - 1e-9:
        return "未配送"
    if t < leave - 1e-9:
        return "配送中"
    return "已配送"


def delivered_orders_at_time(t: int, stop_df: pd.DataFrame, batch_df: pd.DataFrame) -> Set[int]:
    t = int(t)
    delivered: Set[int] = set()
    if stop_df.empty:
        return delivered
    b2orders = batch_df.set_index("batch_id")["order_ids"].to_dict()
    for _, r in stop_df.iterrows():
        if float(r["leave_min"]) <= float(t) + 1e-9:
            bid = int(r["batch_id"])
            for oid in b2orders.get(bid, tuple()):
                delivered.add(int(oid))
    return delivered


def delivered_order_times_at_time(t: int, stop_df: pd.DataFrame, batch_df: pd.DataFrame) -> Dict[int, int]:
    t = int(t)
    out: Dict[int, int] = {}
    if stop_df.empty:
        return out
    b2orders = batch_df.set_index("batch_id")["order_ids"].to_dict()
    for _, r in stop_df.iterrows():
        leave = float(r["leave_min"])
        if leave <= float(t) + 1e-9:
            bid = int(r["batch_id"])
            for oid in b2orders.get(bid, tuple()):
                out[int(oid)] = int(round(leave))
    return out


class DynamicDispatcher:
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        seed: int = 0,
        n_starts: int = 6,
        lns_iters: int = 40,
    ):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent
        self.seed = int(seed)
        self.n_starts = int(n_starts)
        self.lns_iters = int(lns_iters)
        self.orders_df, self.dist_df, self.coords_df = load_base_data(self.base_dir)
        self.tw_df = load_time_windows(self.base_dir)
        self.solution: Optional[Dict[str, Any]] = None
        self.delivered_at: Dict[int, int] = {}
        self.canceled_orders: Set[int] = set()
        self.current_time_min: int = 0
        self.event_index: int = 0

    def plan_initial(self) -> Dict[str, Any]:
        self.solution = optimize_static(
            self.orders_df,
            self.dist_df,
            self.coords_df,
            self.tw_df,
            seed=self.seed,
            n_starts=self.n_starts,
            lns_iters=self.lns_iters,
        )
        self.current_time_min = 0
        self.event_index = 0
        return self.solution

    def _apply_event_updates(self, event: Event, t: int, delivered: Set[int], remaining: pd.DataFrame) -> pd.DataFrame:
        if event.type == "order_cancel":
            oid = int(event.payload["order_id"])
            if oid not in delivered:
                self.canceled_orders.add(int(oid))
            return remaining[remaining["order_id"].astype(int) != oid]
        if event.type == "order_add":
            row = {
                "order_id": int(event.payload["order_id"]),
                "customer_id": int(event.payload["customer_id"]),
                "weight_kg": float(event.payload["weight_kg"]),
                "volume_m3": float(event.payload["volume_m3"]),
            }
            if int(row["order_id"]) in delivered or int(row["order_id"]) in self.delivered_at:
                raise ValueError(f"order_add 的 order_id={row['order_id']} 已完成配送，不能新增")
            if int(row["customer_id"]) not in set(self.coords_df["customer_id"].astype(int).tolist()):
                raise ValueError(f"order_add 的 customer_id={row['customer_id']} 不存在于坐标表")
            if int(row["customer_id"]) not in set(self.tw_df["customer_id"].astype(int).tolist()) and int(row["customer_id"]) != int(q2.DEPOT_ID):
                raise ValueError(f"order_add 的 customer_id={row['customer_id']} 不存在于时间窗表")
            remaining = remaining[remaining["order_id"].astype(int) != int(row["order_id"])]
            remaining = pd.concat([remaining, pd.DataFrame([row])], ignore_index=True)
            return q2.complete_order_weight_volume(remaining)
        if event.type == "customer_address_change":
            cid = int(event.payload["customer_id"])
            x = float(event.payload["x_km"])
            y = float(event.payload["y_km"])
            if cid not in set(self.coords_df["customer_id"].astype(int).tolist()):
                raise ValueError(f"customer_id={cid} 不存在于坐标表")
            self.coords_df.loc[self.coords_df["customer_id"].astype(int) == cid, ["x_km", "y_km"]] = [x, y]
            coord = self.coords_df.set_index("customer_id")[["x_km", "y_km"]].to_dict("index")
            if cid in coord and int(q2.DEPOT_ID) in coord:
                for j in self.dist_df.index.astype(int).tolist():
                    if j not in coord or cid not in coord:
                        continue
                    dx = float(coord[cid]["x_km"]) - float(coord[j]["x_km"])
                    dy = float(coord[cid]["y_km"]) - float(coord[j]["y_km"])
                    d = math.hypot(dx, dy)
                    if cid in self.dist_df.index and j in self.dist_df.columns:
                        self.dist_df.loc[cid, j] = d
                    if j in self.dist_df.index and cid in self.dist_df.columns:
                        self.dist_df.loc[j, cid] = d
            return remaining
        if event.type == "customer_timewindow_change":
            cid = int(event.payload["customer_id"])
            tws = event.payload["tw_start"]
            twe = event.payload["tw_end"]
            tws_min = int(_hhmm_to_minute(tws))
            twe_min = int(_hhmm_to_minute(twe))
            if cid not in set(self.tw_df["customer_id"].astype(int).tolist()) and cid != int(q2.DEPOT_ID):
                raise ValueError(f"customer_id={cid} 不存在于时间窗表")
            self.tw_df.loc[self.tw_df["customer_id"].astype(int) == cid, ["tw_start", "tw_end", "tw_start_min", "tw_end_min"]] = [
                tws,
                twe,
                tws_min,
                twe_min,
            ]
            return remaining
        raise ValueError(f"未知事件类型: {event.type}")

    def apply_events(self, events: Sequence[Event]) -> Dict[str, Any]:
        if not events:
            if self.solution is None:
                return self.plan_initial()
            return self.solution
        if self.solution is None:
            self.plan_initial()
        assert self.solution is not None

        times = [int(e.time_min) for e in events]
        t = int(max(times))
        if t < int(self.current_time_min):
            raise ValueError(f"事件时间 {t} 早于当前时间 {self.current_time_min}")

        delivered_times = delivered_order_times_at_time(t, self.solution["stop_schedule_df"], self.solution["batch_df"])
        for oid, tm in delivered_times.items():
            self.delivered_at.setdefault(int(oid), int(tm))
        delivered = set(delivered_times.keys())
        remaining = self.orders_df[
            ~self.orders_df["order_id"].astype(int).isin(delivered)
            & ~self.orders_df["order_id"].astype(int).isin(self.canceled_orders)
        ].copy()

        for e in events:
            remaining = self._apply_event_updates(e, t=t, delivered=delivered, remaining=remaining)

        self.event_index += int(len(events))
        new_sol = optimize_static(
            remaining,
            self.dist_df,
            self.coords_df,
            self.tw_df,
            seed=self.seed + self.event_index,
            n_starts=self.n_starts,
            lns_iters=self.lns_iters,
        )
        self.solution = new_sol
        keep_delivered = self.orders_df[self.orders_df["order_id"].astype(int).isin(set(self.delivered_at.keys()))]
        self.orders_df = pd.concat([keep_delivered, remaining], ignore_index=True)
        self.current_time_min = max(int(self.current_time_min), int(t))
        return new_sol

    def apply_event(self, event: Event) -> Dict[str, Any]:
        if self.solution is None:
            self.plan_initial()
        assert self.solution is not None

        t = int(event.time_min)
        if t < int(self.current_time_min):
            raise ValueError(f"事件时间 {t} 早于当前时间 {self.current_time_min}")
        delivered_times = delivered_order_times_at_time(t, self.solution["stop_schedule_df"], self.solution["batch_df"])
        for oid, tm in delivered_times.items():
            self.delivered_at.setdefault(int(oid), int(tm))
        delivered = set(delivered_times.keys())
        remaining = self.orders_df[
            ~self.orders_df["order_id"].astype(int).isin(delivered)
            & ~self.orders_df["order_id"].astype(int).isin(self.canceled_orders)
        ].copy()

        remaining = self._apply_event_updates(event, t=t, delivered=delivered, remaining=remaining)

        self.event_index += 1
        new_sol = optimize_static(
            remaining,
            self.dist_df,
            self.coords_df,
            self.tw_df,
            seed=self.seed + self.event_index,
            n_starts=self.n_starts,
            lns_iters=self.lns_iters,
        )
        self.solution = new_sol
        keep_delivered = self.orders_df[self.orders_df["order_id"].astype(int).isin(set(self.delivered_at.keys()))]
        self.orders_df = pd.concat([keep_delivered, remaining], ignore_index=True)
        self.current_time_min = max(int(self.current_time_min), int(t))
        return new_sol

    def order_status(self, order_id: int, time_hhmm: str) -> str:
        if self.solution is None:
            self.plan_initial()
        assert self.solution is not None
        t = int(_hhmm_to_minute(time_hhmm))
        oid = int(order_id)
        if oid in self.delivered_at and t >= int(self.delivered_at[oid]) - 1e-9:
            return "已配送"
        if oid in self.canceled_orders and oid not in self.delivered_at:
            return "已取消"
        return order_status_from_solution(
            order_id,
            t,
            self.solution["route_summary_df"],
            self.solution["stop_schedule_df"],
            self.solution["order_to_batch"],
        )


def _route_id_list(sol: Dict[str, Any]) -> List[int]:
    routes = sol.get("routes", [])
    if not isinstance(routes, list):
        return []
    return list(range(1, len(routes) + 1))


def _route_seq_by_id(sol: Dict[str, Any], route_id: int) -> List[int]:
    routes = sol.get("routes", [])
    if not isinstance(routes, list):
        return []
    idx = int(route_id) - 1
    if idx < 0 or idx >= len(routes):
        return []
    return [int(x) for x in routes[idx]]


def _batch_orders_map(batch_df: pd.DataFrame) -> Dict[int, Tuple[int, ...]]:
    if batch_df is None or not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return {}
    out: Dict[int, Tuple[int, ...]] = {}
    for _, r in batch_df.iterrows():
        bid = int(r["batch_id"])
        oids = r.get("order_ids", tuple())
        if isinstance(oids, (list, tuple)):
            out[bid] = tuple(int(x) for x in oids)
        else:
            out[bid] = (int(oids),) if not pd.isna(oids) else tuple()
    return out


def _route_stop_signature(sol: Dict[str, Any], route_id: int) -> Tuple[Tuple[int, Tuple[int, ...]], ...]:
    seq = _route_seq_by_id(sol, route_id)
    batch_lookup = sol.get("batch_lookup", {})
    b2o = _batch_orders_map(sol.get("batch_df"))
    sig: List[Tuple[int, Tuple[int, ...]]] = []
    for bid in seq:
        info = batch_lookup.get(int(bid), {})
        cid = int(info.get("customer_id", -1))
        oids = tuple(sorted(b2o.get(int(bid), tuple())))
        sig.append((cid, oids))
    return tuple(sig)


def _match_routes_by_overlap(
    old_sol: Dict[str, Any],
    new_sol: Dict[str, Any],
) -> Dict[int, Optional[int]]:
    old_ids = _route_id_list(old_sol)
    new_ids = _route_id_list(new_sol)
    if not old_ids:
        return {}
    old_sets = {
        rid: set(cid for cid, _ in _route_stop_signature(old_sol, rid) if cid >= 0)
        for rid in old_ids
    }
    new_sets = {
        rid: set(cid for cid, _ in _route_stop_signature(new_sol, rid) if cid >= 0)
        for rid in new_ids
    }
    pairs: List[Tuple[int, int, float]] = []
    for oi in old_ids:
        for nj in new_ids:
            a = old_sets[oi]
            b = new_sets[nj]
            inter = len(a & b)
            if inter <= 0:
                continue
            union = len(a | b)
            score = float(inter) / float(union) if union else 0.0
            pairs.append((oi, nj, score))
    pairs.sort(key=lambda x: (x[2], len(old_sets[x[0]] & new_sets[x[1]])), reverse=True)
    assigned_old: Set[int] = set()
    assigned_new: Set[int] = set()
    out: Dict[int, Optional[int]] = {rid: None for rid in old_ids}
    for oi, nj, _ in pairs:
        if oi in assigned_old or nj in assigned_new:
            continue
        assigned_old.add(oi)
        assigned_new.add(nj)
        out[oi] = nj
    return out


def _changed_route_ids(
    old_sol: Dict[str, Any],
    new_sol: Dict[str, Any],
) -> Tuple[Set[int], Set[int], Dict[int, Optional[int]]]:
    match = _match_routes_by_overlap(old_sol, new_sol)
    old_changed: Set[int] = set()
    new_changed: Set[int] = set()
    used_new: Set[int] = set(n for n in match.values() if n is not None)
    for old_rid, new_rid in match.items():
        if new_rid is None:
            old_changed.add(int(old_rid))
            continue
        if _route_stop_signature(old_sol, old_rid) != _route_stop_signature(new_sol, new_rid):
            old_changed.add(int(old_rid))
            new_changed.add(int(new_rid))
    all_new = set(_route_id_list(new_sol))
    new_added = all_new - used_new
    new_changed |= set(int(x) for x in new_added)
    return old_changed, new_changed, match


def _route_detail_df(sol: Dict[str, Any], route_ids: Iterable[int]) -> pd.DataFrame:
    stop_df = sol.get("stop_schedule_df")
    if not isinstance(stop_df, pd.DataFrame) or stop_df.empty:
        return pd.DataFrame(
            columns=[
                "route_id",
                "stop_order",
                "customer_id",
                "order_ids",
                "arrival_time",
                "service_start_time",
                "leave_time",
                "tw_start",
                "tw_end",
                "timewindow_ok",
            ]
        )
    b2o = _batch_orders_map(sol.get("batch_df"))
    batch_lookup = sol.get("batch_lookup", {})

    rows: List[Dict[str, Any]] = []
    for rid in route_ids:
        rid = int(rid)
        sub = stop_df[stop_df["route_id"].astype(int) == rid].sort_values("stop_order")
        for _, r in sub.iterrows():
            bid = int(r["batch_id"])
            info = batch_lookup.get(bid, {})
            cid = int(r.get("customer_id", info.get("customer_id", -1)))
            tws = info.get("tw_start", "")
            twe = info.get("tw_end", "")
            tw_ok = float(r.get("late_min", 0.0)) <= 1e-9
            rows.append(
                {
                    "route_id": rid,
                    "stop_order": int(r["stop_order"]),
                    "customer_id": cid,
                    "order_ids": ",".join(str(x) for x in sorted(b2o.get(bid, tuple()))),
                    "arrival_time": str(r.get("arrival_time", "")),
                    "service_start_time": str(r.get("service_start_time", "")),
                    "leave_time": str(r.get("leave_time", "")),
                    "tw_start": str(tws),
                    "tw_end": str(twe),
                    "timewindow_ok": bool(tw_ok),
                }
            )
    return pd.DataFrame(rows)


def _format_route_text(sol: Dict[str, Any], route_id: int) -> str:
    route_id = int(route_id)
    stop_df = sol.get("stop_schedule_df")
    if not isinstance(stop_df, pd.DataFrame) or stop_df.empty:
        return ""
    batch_lookup = sol.get("batch_lookup", {})
    b2o = _batch_orders_map(sol.get("batch_df"))
    sub = stop_df[stop_df["route_id"].astype(int) == route_id].sort_values("stop_order")
    pieces: List[str] = ["0"]
    for _, r in sub.iterrows():
        bid = int(r["batch_id"])
        info = batch_lookup.get(bid, {})
        cid = int(r.get("customer_id", info.get("customer_id", -1)))
        oids = ",".join(str(x) for x in sorted(b2o.get(bid, tuple())))
        arr = str(r.get("arrival_time", ""))
        tw_ok = float(r.get("late_min", 0.0)) <= 1e-9
        tws = str(info.get("tw_start", ""))
        twe = str(info.get("tw_end", ""))
        pieces.append(f"{cid}[订单:{oids} 到达:{arr} TW:{tws}-{twe} {'OK' if tw_ok else 'FAIL'}]")
    pieces.append("0")
    return " -> ".join(pieces)


def _describe_event_adjustment(
    event: Event,
    old_sol: Dict[str, Any],
    new_sol: Dict[str, Any],
    match: Dict[int, Optional[int]],
    old_changed: Set[int],
    new_changed: Set[int],
) -> List[str]:
    t = _minute_to_hhmm(event.time_min)
    lines: List[str] = []

    if event.type == "order_cancel":
        oid = int(event.payload["order_id"])
        order_to_batch = old_sol.get("order_to_batch", {})
        bid = int(order_to_batch.get(oid, -1))
        old_stop_df = old_sol.get("stop_schedule_df")
        if bid >= 0 and isinstance(old_stop_df, pd.DataFrame):
            hit = old_stop_df[old_stop_df["batch_id"].astype(int) == bid]
            if not hit.empty:
                rid = int(hit["route_id"].iloc[0])
                cid = int(hit["customer_id"].iloc[0])
                lines.append(f"[{t}] 订单{oid}取消：原线路{rid}中客户{cid}的订单{oid}被删除")
            else:
                lines.append(f"[{t}] 订单{oid}取消：订单未出现在原计划停靠表中(可能已完成配送)")
        else:
            lines.append(f"[{t}] 订单{oid}取消：订单不在原计划中(可能已完成配送或不存在)")
    elif event.type == "order_add":
        oid = int(event.payload["order_id"])
        cid = int(event.payload["customer_id"])
        w = float(event.payload["weight_kg"])
        v = float(event.payload["volume_m3"])
        lines.append(f"[{t}] 新增订单{oid}：客户{cid}，重量{w:g}kg，体积{v:g}m3")
    elif event.type == "customer_address_change":
        cid = int(event.payload["customer_id"])
        x = float(event.payload["x_km"])
        y = float(event.payload["y_km"])
        lines.append(f"[{t}] 客户{cid}坐标变更：({x:g},{y:g})")
    elif event.type == "customer_timewindow_change":
        cid = int(event.payload["customer_id"])
        tws = str(event.payload["tw_start"])
        twe = str(event.payload["tw_end"])
        lines.append(f"[{t}] 客户{cid}时间窗变更：{tws}-{twe}")

    if old_changed or new_changed:
        for old_rid in sorted(old_changed):
            new_rid = match.get(int(old_rid))
            if new_rid is None:
                lines.append(f"线路{old_rid}被重构/移除：在新规划中未找到主要对应线路")
            else:
                lines.append(f"调整线路{old_rid} → 新规划线路{int(new_rid)}")
    else:
        lines.append("本次事件后规划线路未发生变化")

    return lines


def _describe_event_only(event: Event) -> str:
    t = _minute_to_hhmm(event.time_min)
    if event.type == "order_cancel":
        return f"[{t}] 订单{int(event.payload['order_id'])}取消"
    if event.type == "order_add":
        return (
            f"[{t}] 新增订单{int(event.payload['order_id'])}：客户{int(event.payload['customer_id'])}，"
            f"重量{float(event.payload['weight_kg']):g}kg，体积{float(event.payload['volume_m3']):g}m3"
        )
    if event.type == "customer_address_change":
        return f"[{t}] 客户{int(event.payload['customer_id'])}坐标变更"
    if event.type == "customer_timewindow_change":
        return f"[{t}] 客户{int(event.payload['customer_id'])}时间窗变更"
    return f"[{t}] {event.type}"


def _plot_routes_subset(
    sol: Dict[str, Any],
    route_ids: Sequence[int],
    out_path: Path,
    title: str,
    *,
    key_for_route_id: Optional[Dict[int, int]] = None,
    label_for_route_id: Optional[Dict[int, str]] = None,
) -> None:
    q2.set_matplotlib_font()
    coords_df = sol.get("coords_df")
    if not isinstance(coords_df, pd.DataFrame) or coords_df.empty:
        return
    coord = coords_df[["customer_id", "x_km", "y_km"]].set_index("customer_id").to_dict("index")
    depot_id = int(q2.DEPOT_ID)
    if depot_id not in coord:
        coord[depot_id] = {"x_km": 0.0, "y_km": 0.0}

    fig, ax = plt.subplots(figsize=(10.5, 10))
    circle = plt.Circle(
        (0, 0),
        float(getattr(q2, "GREEN_ZONE_RADIUS", 10.0)),
        fill=True,
        facecolor="#e8f5e9",
        edgecolor="#2ca02c",
        linewidth=1.2,
        linestyle="--",
        alpha=0.25,
    )
    ax.add_patch(circle)

    batch_lookup = sol.get("batch_lookup", {})

    all_route_ids = _route_id_list(sol)
    highlight_set = set(int(x) for x in route_ids)

    def seq_to_xy(seq: List[int]) -> Tuple[List[float], List[float], List[int]]:
        cids = [int(batch_lookup[int(bid)]["customer_id"]) for bid in seq if int(bid) in batch_lookup]
        path_ids = [depot_id] + cids + [depot_id]
        xs = [float(coord[c]["x_km"]) for c in path_ids]
        ys = [float(coord[c]["y_km"]) for c in path_ids]
        return xs, ys, cids

    for rid in all_route_ids:
        rid = int(rid)
        seq = _route_seq_by_id(sol, rid)
        if not seq:
            continue
        xs, ys, _ = seq_to_xy(seq)
        ax.plot(xs, ys, lw=0.9, color="#9e9e9e", alpha=0.35, zorder=1)

    keys: List[int] = []
    if key_for_route_id is not None:
        keys = sorted({int(key_for_route_id.get(int(rid), int(rid))) for rid in highlight_set})
    else:
        keys = sorted({int(rid) for rid in highlight_set})

    cmap = plt.get_cmap("tab20")
    key_color: Dict[int, Any] = {}
    for i, k in enumerate(keys):
        key_color[int(k)] = cmap(i % 20)

    for rid in sorted(highlight_set):
        rid = int(rid)
        seq = _route_seq_by_id(sol, rid)
        if not seq:
            continue
        xs, ys, cids = seq_to_xy(seq)
        key = int(key_for_route_id.get(rid, rid)) if key_for_route_id is not None else int(rid)
        color = key_color.get(key, cmap(key % 20))
        ax.plot(xs, ys, lw=2.8, color=color, alpha=0.95, zorder=5)
        ax.scatter(xs[1:-1], ys[1:-1], s=44, color=color, alpha=0.98, zorder=6)
        if cids:
            label = (
                label_for_route_id.get(rid, f"R{rid}")
                if label_for_route_id is not None
                else f"R{rid}"
            )
            ax.text(xs[1], ys[1], label, fontsize=9, color=color, zorder=7)

    ax.scatter(float(coord[depot_id]["x_km"]), float(coord[depot_id]["y_km"]), marker="*", s=320, color="red", zorder=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_events_with_report(
    base_dir: Path,
    seed: int,
    n_starts: int,
    lns_iters: int,
    events: Sequence[Event],
    report_excel: Path,
    plot_dir: Path,
) -> Dict[str, Any]:
    d = DynamicDispatcher(base_dir=base_dir, seed=seed, n_starts=n_starts, lns_iters=lns_iters)
    s0 = d.plan_initial()
    s0["coords_df"] = d.coords_df
    costs = [{"stage": "initial", "event_index": 0, "time": "", "type": "", "total_cost": float(s0["total_cost"])}]
    log_lines: List[Dict[str, Any]] = []
    before_rows: List[pd.DataFrame] = []
    after_rows: List[pd.DataFrame] = []
    route_text_rows: List[Dict[str, Any]] = []

    cur = s0
    for idx, ev in enumerate(events, start=1):
        old_sol = cur
        new_sol = d.apply_event(ev)
        new_sol["coords_df"] = d.coords_df
        old_changed, new_changed, match = _changed_route_ids(old_sol, new_sol)

        desc = _describe_event_adjustment(ev, old_sol, new_sol, match, old_changed, new_changed)
        for ln in desc:
            log_lines.append(
                {
                    "event_index": idx,
                    "event_time": _minute_to_hhmm(ev.time_min),
                    "event_type": ev.type,
                    "description": ln,
                }
            )

        changed_old_list = sorted(old_changed)
        changed_new_list = sorted(new_changed)
        bdf = _route_detail_df(old_sol, changed_old_list)
        adf = _route_detail_df(new_sol, changed_new_list)
        if not bdf.empty:
            bdf.insert(0, "event_index", idx)
            before_rows.append(bdf)
        if not adf.empty:
            adf.insert(0, "event_index", idx)
            after_rows.append(adf)

        for old_rid in changed_old_list:
            route_text_rows.append(
                {
                    "event_index": idx,
                    "side": "before",
                    "route_id": int(old_rid),
                    "matched_route_id": int(match.get(int(old_rid))) if match.get(int(old_rid)) is not None else "",
                    "route_text": _format_route_text(old_sol, int(old_rid)),
                }
            )
        for new_rid in changed_new_list:
            inv = {v: k for k, v in match.items() if v is not None}
            old_rid = inv.get(int(new_rid))
            route_text_rows.append(
                {
                    "event_index": idx,
                    "side": "after",
                    "route_id": int(new_rid),
                    "matched_route_id": int(old_rid) if old_rid is not None else "",
                    "route_text": _format_route_text(new_sol, int(new_rid)),
                }
            )

        if changed_old_list:
            _plot_routes_subset(
                old_sol,
                changed_old_list,
                plot_dir / f"event{idx:02d}_before.png",
                title=f"Event {idx:02d} Before (changed routes only)",
            )
        if changed_new_list:
            inv = {v: k for k, v in match.items() if v is not None}
            key_map_after: Dict[int, int] = {}
            label_map_after: Dict[int, str] = {}
            for nr in changed_new_list:
                nr = int(nr)
                if nr in inv:
                    key_map_after[nr] = int(inv[nr])
                    label_map_after[nr] = f"R{int(inv[nr])}"
                else:
                    key_map_after[nr] = int(1_000_000 + nr)
                    label_map_after[nr] = f"R{nr}"
            _plot_routes_subset(
                new_sol,
                changed_new_list,
                plot_dir / f"event{idx:02d}_after.png",
                title=f"Event {idx:02d} After (changed routes only)",
                key_for_route_id=key_map_after,
                label_for_route_id=label_map_after,
            )

        costs.append(
            {
                "stage": "after_event",
                "event_index": idx,
                "time": _minute_to_hhmm(ev.time_min),
                "type": ev.type,
                "total_cost": float(new_sol["total_cost"]),
            }
        )
        cur = new_sol

    cost_df = pd.DataFrame(costs)
    log_df = pd.DataFrame(log_lines)
    before_df = pd.concat(before_rows, ignore_index=True) if before_rows else pd.DataFrame()
    after_df = pd.concat(after_rows, ignore_index=True) if after_rows else pd.DataFrame()
    route_text_df = pd.DataFrame(route_text_rows)

    report_excel.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(report_excel, engine="openpyxl") as w:
        cost_df.to_excel(w, sheet_name="成本汇总", index=False)
        if not log_df.empty:
            log_df.to_excel(w, sheet_name="事件调整说明", index=False)
        if not before_df.empty:
            before_df.to_excel(w, sheet_name="更改前线路明细", index=False)
        if not after_df.empty:
            after_df.to_excel(w, sheet_name="更改后线路明细", index=False)
        if not route_text_df.empty:
            route_text_df.to_excel(w, sheet_name="线路调整文本", index=False)

    return {"initial": s0, "final": cur, "report_excel": str(report_excel), "plot_dir": str(plot_dir)}


def run_event_groups_with_report(
    base_dir: Path,
    seed: int,
    n_starts: int,
    lns_iters: int,
    event_groups: Sequence[Sequence[Event]],
    report_excel: Path,
    plot_dir: Path,
) -> Dict[str, Any]:
    d = DynamicDispatcher(base_dir=base_dir, seed=seed, n_starts=n_starts, lns_iters=lns_iters)
    s0 = d.plan_initial()
    s0["coords_df"] = d.coords_df
    costs = [{"stage": "initial", "group_index": 0, "time": "", "types": "", "total_cost": float(s0["total_cost"])}]
    log_lines: List[Dict[str, Any]] = []
    before_rows: List[pd.DataFrame] = []
    after_rows: List[pd.DataFrame] = []
    route_text_rows: List[Dict[str, Any]] = []

    cur = s0
    for gidx, group in enumerate(event_groups, start=1):
        if not group:
            continue
        old_sol = cur
        new_sol = d.apply_events(list(group))
        new_sol["coords_df"] = d.coords_df
        old_changed, new_changed, match = _changed_route_ids(old_sol, new_sol)

        ts = sorted({_minute_to_hhmm(e.time_min) for e in group})
        types = "+".join(str(e.type) for e in group)
        log_lines.append(
            {
                "group_index": gidx,
                "group_time": ",".join(ts),
                "group_types": types,
                "description": "批量事件：" + " | ".join(_describe_event_only(e) for e in group),
            }
        )
        head_lines = _describe_event_adjustment(group[0], old_sol, new_sol, match, old_changed, new_changed)
        for ln in head_lines[1:]:
            log_lines.append(
                {
                    "group_index": gidx,
                    "group_time": ",".join(ts),
                    "group_types": types,
                    "description": ln,
                }
            )

        changed_old_list = sorted(old_changed)
        changed_new_list = sorted(new_changed)
        bdf = _route_detail_df(old_sol, changed_old_list)
        adf = _route_detail_df(new_sol, changed_new_list)
        if not bdf.empty:
            bdf.insert(0, "group_index", gidx)
            before_rows.append(bdf)
        if not adf.empty:
            adf.insert(0, "group_index", gidx)
            after_rows.append(adf)

        for old_rid in changed_old_list:
            route_text_rows.append(
                {
                    "group_index": gidx,
                    "side": "before",
                    "route_id": int(old_rid),
                    "matched_route_id": int(match.get(int(old_rid))) if match.get(int(old_rid)) is not None else "",
                    "route_text": _format_route_text(old_sol, int(old_rid)),
                }
            )
        inv = {v: k for k, v in match.items() if v is not None}
        for new_rid in changed_new_list:
            new_rid = int(new_rid)
            old_rid = inv.get(new_rid)
            route_text_rows.append(
                {
                    "group_index": gidx,
                    "side": "after",
                    "route_id": int(new_rid),
                    "matched_route_id": int(old_rid) if old_rid is not None else "",
                    "route_text": _format_route_text(new_sol, int(new_rid)),
                }
            )

        if changed_old_list:
            _plot_routes_subset(
                old_sol,
                changed_old_list,
                plot_dir / f"group{gidx:02d}_before.png",
                title=f"Group {gidx:02d} Before (changed routes highlighted)",
            )
        if changed_new_list:
            key_map_after: Dict[int, int] = {}
            label_map_after: Dict[int, str] = {}
            for nr in changed_new_list:
                nr = int(nr)
                if nr in inv:
                    key_map_after[nr] = int(inv[nr])
                    label_map_after[nr] = f"R{int(inv[nr])}"
                else:
                    key_map_after[nr] = int(1_000_000 + nr)
                    label_map_after[nr] = f"R{nr}"
            _plot_routes_subset(
                new_sol,
                changed_new_list,
                plot_dir / f"group{gidx:02d}_after.png",
                title=f"Group {gidx:02d} After (changed routes highlighted)",
                key_for_route_id=key_map_after,
                label_for_route_id=label_map_after,
            )

        costs.append(
            {
                "stage": "after_group",
                "group_index": gidx,
                "time": ",".join(ts),
                "types": types,
                "total_cost": float(new_sol["total_cost"]),
            }
        )
        cur = new_sol

    cost_df = pd.DataFrame(costs)
    log_df = pd.DataFrame(log_lines)
    before_df = pd.concat(before_rows, ignore_index=True) if before_rows else pd.DataFrame()
    after_df = pd.concat(after_rows, ignore_index=True) if after_rows else pd.DataFrame()
    route_text_df = pd.DataFrame(route_text_rows)

    report_excel.parent.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(report_excel, engine="openpyxl") as w:
        cost_df.to_excel(w, sheet_name="成本汇总", index=False)
        if not log_df.empty:
            log_df.to_excel(w, sheet_name="事件调整说明", index=False)
        if not before_df.empty:
            before_df.to_excel(w, sheet_name="更改前线路明细", index=False)
        if not after_df.empty:
            after_df.to_excel(w, sheet_name="更改后线路明细", index=False)
        if not route_text_df.empty:
            route_text_df.to_excel(w, sheet_name="线路调整文本", index=False)

    return {"initial": s0, "final": cur, "report_excel": str(report_excel), "plot_dir": str(plot_dir)}


def demo() -> None:
    d = DynamicDispatcher(seed=42, n_starts=1, lns_iters=6)
    s0 = d.plan_initial()
    print(f"初始解总成本: {s0['total_cost']:.2f}, 线路数: {len(s0['routes'])}")

    e1 = Event(time_min=_hhmm_to_minute("10:30"), type="order_cancel", payload={"order_id": 1})
    s1 = d.apply_event(e1)
    print(f"取消订单后总成本: {s1['total_cost']:.2f}, 线路数: {len(s1['routes'])}")

    print("订单1@10:20状态:", d.order_status(1, "10:20"))


def _solution_summary_text(sol: Dict[str, Any]) -> str:
    total_cost = float(sol.get("total_cost", float("nan")))
    routes = sol.get("routes", [])
    route_count = len(routes) if isinstance(routes, list) else int(sol.get("route_count", 0))
    vehicle_count = sol.get("vehicle_count", None)
    pieces = [f"total_cost={total_cost:.2f}", f"routes={route_count}"]
    if vehicle_count is not None:
        if isinstance(vehicle_count, dict):
            try:
                pieces.append(f"vehicles={int(sum(int(v) for v in vehicle_count.values()))}")
            except Exception:
                pieces.append("vehicles=?")
        else:
            pieces.append(f"vehicles={int(vehicle_count)}")
    return ", ".join(pieces)


def _print_df_head(df: pd.DataFrame, n: int) -> None:
    if df is None or not isinstance(df, pd.DataFrame):
        print("(无表)")
        return
    if df.empty:
        print("(空表)")
        return
    n = max(1, int(n))
    text = df.head(n).to_string(index=False)
    print(text)


def _export_solution_csv(sol: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables: Dict[str, Any] = {
        "route_summary_df": sol.get("route_summary_df"),
        "stop_schedule_df": sol.get("stop_schedule_df"),
        "leg_df": sol.get("leg_df"),
        "vehicle_chain_df": sol.get("vehicle_chain_df"),
        "assign_df": sol.get("assign_df"),
        "batch_df": sol.get("batch_df"),
    }
    for name, obj in tables.items():
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")


def _run_interactive(
    d: DynamicDispatcher,
    report_excel: Optional[Path] = None,
    report_plot_dir: Optional[Path] = None,
) -> None:
    sol = d.plan_initial()
    print("初始规划:", _solution_summary_text(sol))
    print("输入事件(JSON 或: <time> <type> k=v ...)，输入 help 查看命令，输入 exit 退出。")

    events_so_far: List[Event] = []
    event_groups: List[List[Event]] = []
    batch_mode = False
    pending_batch: List[Event] = []

    def show_help() -> None:
        print("可用命令:")
        print("  <event>                         直接输入事件")
        print("  batch on|off                    开启/关闭批量模式(开启后事件先入队)")
        print("  apply                           应用已入队事件(一次重规划，输出一次结果)")
        print("  pending                         查看已入队事件")
        print("  clear                           清空已入队事件")
        print("  status <order_id> <HH:MM>       查询订单状态")
        print("  summary                         打印当前解摘要")
        print("  table <name> [n]                打印表头n行 (route_summary|stop_schedule|leg|vehicle_chain|assign|batch)")
        print("  export <dir>                    导出当前解为 CSV")
        if report_excel or report_plot_dir:
            print("  (已启用报告)                    每次事件后自动更新Excel与线路变更图")
        print("  exit                            退出")
        print("事件示例:")
        print('  {"time":"10:30","type":"order_cancel","payload":{"order_id":1}}')
        print('  10:30 order_add order_id=9001 customer_id=12 weight_kg=3.2 volume_m3=0.015')
        print('  600 customer_timewindow_change customer_id=12 tw_start=09:00 tw_end=12:00')

    while True:
        try:
            line = input("event> ")
        except EOFError:
            break
        cmd = line.strip()
        if not cmd:
            continue
        if cmd.lower() in {"exit", "quit", "q"}:
            break
        if cmd.lower() in {"help", "h", "?"}:
            show_help()
            continue
        if cmd.lower() == "pending":
            if not pending_batch:
                print("(暂无已入队事件)")
            else:
                for i, e in enumerate(pending_batch, start=1):
                    print(f"{i}. { _minute_to_hhmm(e.time_min) } {e.type} {e.payload}")
            continue
        if cmd.lower() == "clear":
            pending_batch = []
            print("已清空已入队事件")
            continue
        if cmd.lower() == "apply":
            if not pending_batch:
                print("(暂无已入队事件)")
                continue
            try:
                sol = d.apply_events(pending_batch)
                events_so_far.extend(pending_batch)
                event_groups.append(list(pending_batch))
                pending_batch = []
                print("已批量应用事件:", _solution_summary_text(sol))

                if report_excel or report_plot_dir:
                    out_excel = report_excel if report_excel is not None else (Path(__file__).resolve().parent / "q3_dynamic_report.xlsx")
                    out_plot = report_plot_dir if report_plot_dir is not None else (Path(__file__).resolve().parent / "q3_route_change_plots")
                    run_event_groups_with_report(
                        base_dir=d.base_dir,
                        seed=int(d.seed),
                        n_starts=int(d.n_starts),
                        lns_iters=int(d.lns_iters),
                        event_groups=event_groups,
                        report_excel=out_excel,
                        plot_dir=out_plot,
                    )
                    print("已更新动态调度报告:", str(out_excel))
                    print("线路变更图目录:", str(out_plot))
            except Exception as ex:
                print("错误:", ex)
            continue
        if cmd.lower().startswith("batch"):
            parts = cmd.split()
            if len(parts) == 1:
                batch_mode = not batch_mode
            elif len(parts) >= 2:
                if parts[1].lower() in {"on", "1", "true"}:
                    batch_mode = True
                elif parts[1].lower() in {"off", "0", "false"}:
                    batch_mode = False
                else:
                    print("用法: batch on|off")
                    continue
            print("批量模式:", "ON" if batch_mode else "OFF")
            continue
        if cmd.lower().startswith("status "):
            parts = cmd.split()
            if len(parts) != 3:
                print("用法: status <order_id> <HH:MM>")
                continue
            try:
                print(d.order_status(int(parts[1]), parts[2]))
            except Exception as ex:
                print("错误:", ex)
            continue
        if cmd.lower() == "summary":
            if d.solution is None:
                d.plan_initial()
            assert d.solution is not None
            print(_solution_summary_text(d.solution))
            continue
        if cmd.lower().startswith("table "):
            parts = cmd.split()
            name = parts[1] if len(parts) >= 2 else ""
            n = int(parts[2]) if len(parts) >= 3 else 10
            if d.solution is None:
                d.plan_initial()
            assert d.solution is not None
            mapping = {
                "route_summary": "route_summary_df",
                "stop_schedule": "stop_schedule_df",
                "leg": "leg_df",
                "vehicle_chain": "vehicle_chain_df",
                "assign": "assign_df",
                "batch": "batch_df",
            }
            key = mapping.get(name, name)
            obj = d.solution.get(key)
            if not isinstance(obj, pd.DataFrame):
                print(f"未知表名: {name}")
                continue
            _print_df_head(obj, n)
            continue
        if cmd.lower().startswith("export "):
            parts = cmd.split(maxsplit=1)
            if len(parts) != 2:
                print("用法: export <dir>")
                continue
            if d.solution is None:
                d.plan_initial()
            assert d.solution is not None
            try:
                _export_solution_csv(d.solution, Path(parts[1]).expanduser())
                print("已导出")
            except Exception as ex:
                print("导出失败:", ex)
            continue

        try:
            event = parse_event_line(cmd)
            if batch_mode:
                pending_batch.append(event)
                print(f"已入队事件({event.type}, t={_minute_to_hhmm(event.time_min)}), 当前队列={len(pending_batch)}")
            else:
                sol = d.apply_event(event)
                events_so_far.append(event)
                event_groups.append([event])
                print(f"应用事件({event.type}, t={_minute_to_hhmm(event.time_min)}):", _solution_summary_text(sol))

                if report_excel or report_plot_dir:
                    out_excel = report_excel if report_excel is not None else (Path(__file__).resolve().parent / "q3_dynamic_report.xlsx")
                    out_plot = report_plot_dir if report_plot_dir is not None else (Path(__file__).resolve().parent / "q3_route_change_plots")
                    run_event_groups_with_report(
                        base_dir=d.base_dir,
                        seed=int(d.seed),
                        n_starts=int(d.n_starts),
                        lns_iters=int(d.lns_iters),
                        event_groups=event_groups,
                        report_excel=out_excel,
                        plot_dir=out_plot,
                    )
                    print("已更新动态调度报告:", str(out_excel))
                    print("线路变更图目录:", str(out_plot))
        except Exception as ex:
            print("错误:", ex)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--base-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-starts", type=int, default=1)
    p.add_argument("--lns-iters", type=int, default=6)
    p.add_argument("--events", type=str, default=None)
    p.add_argument("--event-mode", choices=["batch", "by-time", "sequential"], default="batch",
                   help="事件文件的重规划方式：batch=全部事件一次性重规划；by-time=同一时刻批量；sequential=逐条重规划。")
    p.add_argument("--export-dir", type=str, default=None)
    p.add_argument("--report-excel", type=str, default="q3_dynamic_report.xlsx")
    p.add_argument("--report-plot-dir", type=str, default="q3_route_change_plots")
    p.add_argument("--no-interactive", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    d = DynamicDispatcher(
        base_dir=Path(args.base_dir) if args.base_dir else None,
        seed=int(args.seed),
        n_starts=int(args.n_starts),
        lns_iters=int(args.lns_iters),
    )

    sol = d.plan_initial()
    print("初始规划:", _solution_summary_text(sol))
    if args.export_dir:
        _export_solution_csv(sol, Path(args.export_dir).expanduser())

    if args.events:
        path = Path(args.events).expanduser()
        parsed_events = parse_events_text(path.read_text(encoding="utf-8"))
        event_groups = group_events_for_replanning(parsed_events, mode=str(args.event_mode))
        if not event_groups:
            print("事件文件为空，未触发动态重规划。")

        if args.report_excel or args.report_plot_dir:
            report_excel = (
                Path(args.report_excel).expanduser()
                if args.report_excel
                else (Path(args.export_dir).expanduser() if args.export_dir else Path(__file__).resolve().parent) / "q3_dynamic_report_batch.xlsx"
            )
            plot_dir = (
                Path(args.report_plot_dir).expanduser()
                if args.report_plot_dir
                else (Path(args.export_dir).expanduser() if args.export_dir else Path(__file__).resolve().parent) / "q3_route_change_plots_batch"
            )
            # 修复点：事件文件不再逐条调用 run_events_with_report/apply_event，
            # 而是按 event_groups 调用 apply_events，一组事件只重规划一次，只生成一个统一总成本。
            res = run_event_groups_with_report(
                base_dir=Path(args.base_dir).expanduser() if args.base_dir else Path(__file__).resolve().parent,
                seed=int(args.seed),
                n_starts=int(args.n_starts),
                lns_iters=int(args.lns_iters),
                event_groups=event_groups,
                report_excel=report_excel,
                plot_dir=plot_dir,
            )
            print("已生成动态调度报告:", res["report_excel"])
            print("线路变更图目录:", res["plot_dir"])
            print(f"事件重规划模式: {args.event_mode}; 重规划次数: {len(event_groups)}")
            sol = res["final"]
        else:
            for gidx, group in enumerate(event_groups, start=1):
                sol = d.apply_events(group)
                print(f"应用事件组{gidx}({_event_group_label(group)}):", _solution_summary_text(sol))
                if args.export_dir:
                    _export_solution_csv(sol, Path(args.export_dir).expanduser())

    if not args.no_interactive:
        _run_interactive(
            d,
            report_excel=Path(args.report_excel).expanduser() if args.report_excel else None,
            report_plot_dir=Path(args.report_plot_dir).expanduser() if args.report_plot_dir else None,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

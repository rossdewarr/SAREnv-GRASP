""""
GRASP-algorithm for path generation with reactive alpha selection
"""

import numpy as np
from multiprocessing import Pool
from shapely.geometry import LineString
from shapely.ops import substring

_shared_visibility: dict = {}

def _init_worker(vt: dict):
    global _shared_visibility
    _shared_visibility = vt


#helper functions from original paths.py
def restrict_path_length(line: LineString, max_length: float) -> LineString:
    if isinstance(line, list):
        return [restrict_path_length(path, max_length) for path in line]
    if line.is_empty or max_length is None or max_length <= 0 or line.length <= max_length:
        return line
    return substring(line, 0, max_length)


def _grid_path_length(path: list[tuple], dx: float, dy: float) -> float:
    return sum(
        np.hypot((path[k][1] - path[k - 1][1]) * dx,
                 (path[k][0] - path[k - 1][0]) * dy)
        for k in range(1, len(path))
    )


def evaluate_solution(paths, probability_map, visibility_table):
    seen = set()
    for path in paths:
        for pos in path:
            seen.update(visibility_table[pos])
    return sum(probability_map[r, c] for r, c in seen)


def precompute_visibility(
    height, width, x_offset, y_offset, dx, dy,
    detection_radius_cells_x, detection_radius_cells_y, detection_radius
) -> dict[tuple, frozenset]:
    table = {}
    for row in range(height):
        for col in range(width):
            world_x = x_offset + col * dx
            world_y = y_offset + row * dy
            visible = set()
            for r in range(max(0, row - detection_radius_cells_y),
                           min(height, row + detection_radius_cells_y + 1)):
                for c in range(max(0, col - detection_radius_cells_x),
                               min(width, col + detection_radius_cells_x + 1)):
                    if np.hypot((x_offset + c * dx) - world_x,
                                (y_offset + r * dy) - world_y) <= detection_radius:
                        visible.add((r, c))
            table[(row, col)] = frozenset(visible)
    return table


#phase 1: greedy randomised contruction
def construct_phase(
    center_x: float, center_y: float, max_radius: float,
    alpha: float, bounds: tuple, probability_map: np.ndarray,
    num_drones: int, fov_deg: float, altitude: float,
    visibility_table: dict, **kwargs
) -> list[LineString]:

    budget = kwargs.get('budget')
    height, width = probability_map.shape
    minx, miny, maxx, maxy = bounds
    rng = np.random.default_rng()

    if maxx <= minx or maxy <= miny:
        return [LineString() for _ in range(num_drones)]

    dx = (maxx - minx) / width
    dy = (maxy - miny) / height
    x_offset = minx + dx / 2
    y_offset = miny + dy / 2

    start_col = np.clip(int((center_x - minx) / dx), 0, width - 1)
    start_row = np.clip(int((center_y - miny) / dy), 0, height - 1)
    start_pos = (start_row, start_col)

    current_positions = [start_pos]
    for i in range(1, num_drones):
        angle = 2 * np.pi * i / num_drones
        offset_r = min(2, height // 10)
        offset_c = min(2, width // 10)
        new_r = np.clip(start_pos[0] + int(offset_r * np.sin(angle)), 0, height - 1)
        new_c = np.clip(start_pos[1] + int(offset_c * np.cos(angle)), 0, width - 1)
        current_positions.append((new_r, new_c))

    neighbor_offsets = [
        (dr, dc)
        for dr in [-1, 0, 1]
        for dc in [-1, 0, 1]
        if not (dr == 0 and dc == 0)
    ]
    max_radius_sq = max_radius * max_radius

    def get_candidates(current_pos, observed_cells, visited, allow_revisit=False):
        current_r, current_c = current_pos
        candidates = []
        for dr, dc in neighbor_offsets:
            nr, nc = current_r + dr, current_c + dc
            if nr < 0 or nr >= height or nc < 0 or nc >= width:
                continue
            world_x = x_offset + nc * dx
            world_y = y_offset + nr * dy
            if (world_x - center_x) ** 2 + (world_y - center_y) ** 2 >= max_radius_sq:
                continue
            candidate = (nr, nc)
            if not allow_revisit and candidate in visited:
                continue
            new_cells = visibility_table[candidate] - observed_cells
            score = sum(probability_map[r, c] for r, c in new_cells)
            candidates.append((candidate, score))
        return candidates

    paths = [[] for _ in range(num_drones)]
    globally_observed_cells = set()
    for i, pos in enumerate(current_positions):
        paths[i].append(pos)
        globally_observed_cells.update(visibility_table[pos])

    path_lengths = [0.0] * num_drones
    observed_cells_per_drone = [set() for _ in range(num_drones)]
    visited_cells_per_drone = [set() for _ in range(num_drones)]
    drone_with_budget = set(range(num_drones))

    while drone_with_budget:
        for i in list(drone_with_budget):
            if path_lengths[i] >= budget:
                drone_with_budget.discard(i)
                continue
            candidates = get_candidates(
                current_positions[i], observed_cells_per_drone[i],
                visited_cells_per_drone[i], allow_revisit=False
            )
            if not candidates:
                candidates = get_candidates(
                    current_positions[i], observed_cells_per_drone[i],
                    visited_cells_per_drone[i], allow_revisit=True
                )
            if not candidates:
                drone_with_budget.discard(i)
                continue

            scores = [sc for _, sc in candidates]
            s_max, s_min = max(scores), min(scores)
            threshold = s_max - alpha * (s_max - s_min)
            rcl = [(pos, sc) for pos, sc in candidates if sc >= threshold]
            selected_pos, _ = rcl[rng.integers(0, len(rcl))]

            dist = np.hypot(
                (selected_pos[1] - current_positions[i][1]) * dx,
                (selected_pos[0] - current_positions[i][0]) * dy
            )
            if path_lengths[i] + dist > budget:
                drone_with_budget.discard(i)
                continue

            current_positions[i] = selected_pos
            paths[i].append(selected_pos)
            path_lengths[i] += dist
            observed_cells_per_drone[i].update(visibility_table[selected_pos])
            visited_cells_per_drone[i].add(selected_pos)

    line_paths = []
    for drone_path_indices in paths:
        if len(drone_path_indices) > 1:
            line_paths.append(LineString([
                (x_offset + c * dx, y_offset + r * dy)
                for r, c in drone_path_indices
            ]))
        else:
            line_paths.append(LineString())

    return line_paths


#phase 2: local search refinement
def local_search(
    grid_paths, probability_map, bounds,
    visibility_table, budgets, max_iterations=200
):
    height, width = probability_map.shape
    minx, miny, maxx, maxy = bounds
    dx = (maxx - minx) / width
    dy = (maxy - miny) / height
    local_search_radius = 4

    def segment_length(p1, p2):
        return np.hypot((p2[1] - p1[1]) * dx, (p2[0] - p1[0]) * dy)

    coverage_count = np.zeros((height, width), dtype=np.int32)
    for path in grid_paths:
        for pos in path:
            for (r, c) in visibility_table[pos]:
                coverage_count[r, c] += 1

    paths = [p.copy() for p in grid_paths]
    score_before = sum(
        probability_map[r, c]
        for r in range(height)
        for c in range(width)
        if coverage_count[r, c] > 0
    )

    for iteration in range(max_iterations):
        any_improved = False
        for drone_idx, path in enumerate(paths):
            if len(path) < 3:
                continue
            budget = budgets[drone_idx]
            current_length = sum(segment_length(path[k - 1], path[k])
                                 for k in range(1, len(path)))
            path_set = set(path)
            best_delta = 0.0
            best_node = None
            best_substitution = None

            for k in range(1, len(path) - 1):
                node = path[k]
                prev_pos = path[k - 1]
                next_pos = path[k + 1]
                old_seg_len = segment_length(prev_pos, node) + segment_length(node, next_pos)
                node_visibility = visibility_table[node]

                for (r, c) in node_visibility:
                    coverage_count[r, c] -= 1
                unique_loss = sum(
                    probability_map[r, c]
                    for (r, c) in node_visibility
                    if coverage_count[r, c] == 0
                )

                for dr in range(-local_search_radius, local_search_radius + 1):
                    for dc in range(-local_search_radius, local_search_radius + 1):
                        if dr == 0 and dc == 0:
                            continue
                        new_row, new_col = node[0] + dr, node[1] + dc
                        if not (0 <= new_row < height and 0 <= new_col < width):
                            continue
                        potential_sub = (new_row, new_col)
                        if potential_sub in path_set:
                            continue
                        new_seg_len = (segment_length(prev_pos, potential_sub)
                                       + segment_length(potential_sub, next_pos))
                        if current_length - old_seg_len + new_seg_len > budget:
                            continue
                        gain = sum(
                            probability_map[r, c]
                            for (r, c) in visibility_table[potential_sub]
                            if coverage_count[r, c] == 0
                        )
                        delta = gain - unique_loss
                        if delta > best_delta:
                            best_delta = delta
                            best_node = k
                            best_substitution = potential_sub

                for (r, c) in node_visibility:
                    coverage_count[r, c] += 1

            if best_substitution is not None:
                k = best_node
                old = path[k]
                for (r, c) in visibility_table[old]:
                    coverage_count[r, c] -= 1
                path[k] = best_substitution
                path_set.discard(old)
                path_set.add(best_substitution)
                for (r, c) in visibility_table[best_substitution]:
                    coverage_count[r, c] += 1
                prev_pos = path[k - 1]
                next_pos = path[k + 1]
                old_seg = segment_length(prev_pos, old) + segment_length(old, next_pos)
                new_seg = (segment_length(prev_pos, best_substitution)
                           + segment_length(best_substitution, next_pos))
                current_length = current_length - old_seg + new_seg
                any_improved = True

        if not any_improved:
            print(f"  Local search converged at iteration {iteration}")
            break

    score_after = sum(
        probability_map[r, c]
        for r in range(height)
        for c in range(width)
        if coverage_count[r, c] > 0
    )
    print(f"  Local search: {score_before:.4f} -> {score_after:.4f} "
          f"(+{score_after - score_before:.6f})")

    final_paths = []
    for drone_idx, path in enumerate(paths):
        if not path:
            continue
        cleaned = [path[0]]
        for pos in path[1:]:
            if pos != cleaned[-1]:
                cleaned.append(pos)
        path_len = 0.0
        final = [cleaned[0]]
        for k in range(1, len(cleaned)):
            seg = segment_length(final[-1], cleaned[k])
            if path_len + seg > budgets[drone_idx]:
                break
            path_len += seg
            final.append(cleaned[k])
        final_paths.append(final)

    return final_paths

#single iteration to be run in parallel
def _single_iteration(args):
    (i, center_x, center_y, num_drones, probability_map, bounds,
     max_radius, alpha, kwargs) = args

    print(f"GRASP iteration {i + 1} (alpha={alpha:.2f})")

    line_paths = construct_phase(
        center_x=center_x, center_y=center_y, num_drones=num_drones,
        probability_map=probability_map, bounds=bounds,
        max_radius=max_radius, alpha=alpha,
        visibility_table=_shared_visibility, **kwargs
    )

    height, width = probability_map.shape
    minx, miny, maxx, maxy = bounds
    dx       = (maxx - minx) / width
    dy       = (maxy - miny) / height
    x_offset = minx + dx / 2
    y_offset = miny + dy / 2

    grid_paths = []
    for lp in line_paths:
        if lp.is_empty:
            grid_paths.append([])
        else:
            grid_paths.append([
                (int((y - y_offset) / dy), int((x - x_offset) / dx))
                for x, y in lp.coords
            ])

    construction_score = evaluate_solution(grid_paths, probability_map, _shared_visibility)
    print(f"  Iteration {i + 1}: construction score {construction_score:.4f}")
    return construction_score, grid_paths

#reactive grasp alpha update
def _update_alpha_probs(
    alpha_set: list[float],
    alpha_scores: dict[float, list[float]],
    incumbent: float,
    gamma: float = 5.0,
) -> np.ndarray:
    q = np.ones(len(alpha_set))

    for idx, a in enumerate(alpha_set):
        scores = alpha_scores.get(a, [])
        if scores:
            A_i = np.mean(scores)
            ratio = A_i / incumbent if incumbent > 0 else 1.0
            ratio = max(ratio, 1e-12)
            q[idx] = ratio ** gamma

    probs = q / q.sum()
    return probs

#full grasp path loop
def generate_grasp_path(
    center_x: float, center_y: float, num_drones: int,
    probability_map: np.ndarray, bounds: tuple[float, float, float, float],
    max_radius: float, alpha: float = 0.15, **kwargs
) -> list[LineString]:
    num_grasp_iterations = kwargs.get('grasp_iterations', 275)
    batch_size           = kwargs.get('batch_size', 8)
    use_reactive_alpha   = kwargs.get('use_reactive_alpha', True)
    alpha_min            = kwargs.get('alpha_min', 0.1)
    alpha_max            = kwargs.get('alpha_max', 0.4)
    alpha_steps          = kwargs.get('alpha_steps', 8)
    alpha_set            = list(np.linspace(alpha_min, alpha_max, alpha_steps)) if use_reactive_alpha else [alpha]
    budget               = kwargs.get('budget')
    fov_deg              = kwargs.get('fov_deg', 45.0)
    altitude             = kwargs.get('altitude', 80.0)

    height, width   = probability_map.shape
    minx, miny, maxx, maxy = bounds
    dx       = (maxx - minx) / width
    dy       = (maxy - miny) / height
    x_offset = minx + dx / 2
    y_offset = miny + dy / 2

    detection_radius         = altitude * np.tan(np.radians(fov_deg / 2))
    detection_radius_cells_x = int(np.ceil(detection_radius / dx))
    detection_radius_cells_y = int(np.ceil(detection_radius / dy))

    print("Precomputing visibility table...")
    visibility_table = precompute_visibility(
        height, width, x_offset, y_offset, dx, dy,
        detection_radius_cells_x, detection_radius_cells_y, detection_radius
    )
    print(f"Visibility table ready ({len(visibility_table)} cells).")

    m             = len(alpha_set)
    probs         = np.ones(m) / m
    alpha_scores  = {a: [] for a in alpha_set}
    rng           = np.random.default_rng()

    best_score  = -np.inf
    best_paths  = None
    best_alpha  = None

    if use_reactive_alpha:
        iteration = 0
        batch_num = 0

        with Pool(initializer=_init_worker, initargs=(visibility_table,)) as pool:
            while iteration < num_grasp_iterations:
                this_batch = min(batch_size, num_grasp_iterations - iteration)
                batch_num += 1

                sampled_alphas = rng.choice(alpha_set, size=this_batch, p=probs)

                print(f"\nBatch {batch_num} ({this_batch} iterations) — "
                      f"best alpha so far: {best_alpha} — "
                      f"alpha probs: " +
                      ", ".join(f"{a:.2f}:{p:.3f}" for a, p in zip(alpha_set, probs)))

                args_list = [
                    (iteration + k, center_x, center_y, num_drones, probability_map,
                     bounds, max_radius, float(sampled_alphas[k]), kwargs)
                    for k in range(this_batch)
                ]

                results = pool.map(_single_iteration, args_list)

                for k, (construction_score, grid_paths) in enumerate(results):
                    a = float(sampled_alphas[k])
                    alpha_scores[a].append(construction_score)

                    if construction_score > best_score:
                        best_score = construction_score
                        best_paths = [p.copy() for p in grid_paths]
                        best_alpha = a
                        print(f"  New best: {construction_score:.4f} (alpha={a:.2f})")
                    else:
                        print(f"  Score: {construction_score:.4f}  (best: {best_score:.4f}, alpha={a:.2f})")

                probs = _update_alpha_probs(alpha_set, alpha_scores, best_score)
                iteration += this_batch

        print(f"\nGRASP construction complete — best: {best_score:.4f} (best alpha={best_alpha:.2f})")
        print("Final alpha probabilities:")
        for a, p in zip(alpha_set, probs):
            scores = alpha_scores[a]
            mean_s = np.mean(scores) if scores else float('nan')
            print(f"  alpha={a:.2f}  p={p:.4f}  n={len(scores)}  mean={mean_s:.4f}")

    else:
        _init_worker(visibility_table)
        print(f"\nRunning {num_grasp_iterations} iterations (fixed alpha={alpha})")
        for i in range(num_grasp_iterations):
            construction_score, grid_paths = _single_iteration((
                i, center_x, center_y, num_drones, probability_map,
                bounds, max_radius, alpha, kwargs
            ))
            if construction_score > best_score:
                best_score = construction_score
                best_paths = [p.copy() for p in grid_paths]
                best_alpha = alpha
                print(f"  New best: {construction_score:.4f}")
            else:
                print(f"  Score: {construction_score:.4f}  (best: {best_score:.4f})")

        print(f"\nGRASP construction complete — best: {best_score:.4f} (alpha={best_alpha:.2f})")

    for idx, path in enumerate(best_paths):
        print(f"  Path {idx}: {len(path)} nodes")

    if kwargs.get('enable_local_search', True) and best_paths and any(len(p) > 0 for p in best_paths):
        print("\nRunning local search on best construction solution...")
        best_paths = local_search(
            grid_paths=best_paths,
            probability_map=probability_map,
            bounds=bounds,
            visibility_table=visibility_table,
            budgets=[budget] * num_drones,
        )

    line_paths = []
    for path in best_paths:
        if len(path) > 1:
            line_paths.append(LineString([
                (x_offset + c * dx, y_offset + r * dy)
                for r, c in path
            ]))
        else:
            line_paths.append(LineString())

    return line_paths
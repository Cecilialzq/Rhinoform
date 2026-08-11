from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy import sparse

from rhinoform.data import load_rows
from rhinoform.repro import atomic_write_json, sha256_file


EPS = 1e-12
Z_DAME = math.sqrt(math.log(100.0 / math.pi)) / math.pi

# Primary formula sources. The implementations below are explicitly adapted to
# the cropped, same-connectivity nasal ROI and are not reference implementations.
FMPD_PAPER = "https://www.gipsa-lab.grenoble-inp.fr/~kai.wang/papers/CG12.pdf"
MSDM2_PAPER = "https://perso.liris.cnrs.fr/guillaume.lavoue/revue/SGP2011.pdf"
DAME_FORMULA = "https://arxiv.org/html/2402.10365v1"


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def completed_prefix(
    path: Path, pairs: list[tuple[str, str]]
) -> list[dict[str, float | str]]:
    if not path.is_file():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as error:
        raise ValueError(f"Cannot safely resume malformed progress file {path}: {error}") from error
    required = {
        "pair_index", "source_id", "target_id", "dame_area_proxy",
        "fmpd_roi_proxy", "msdm2_style_proxy", "normal_flip_face_count",
        "normal_flip_pct",
    }
    if len(rows) > len(pairs) or (rows and not required.issubset(rows[0])):
        raise ValueError(f"Cannot safely resume incompatible progress file: {path}")
    for index, row in enumerate(rows):
        if int(float(row["pair_index"])) != index:
            raise ValueError(f"Non-contiguous pair index at row {index} in {path}")
        if (str(row["source_id"]), str(row["target_id"])) != pairs[index]:
            raise ValueError(f"Pair-order mismatch at row {index} in {path}")
    return rows


def topology_data(faces: np.ndarray, n_vertices: int) -> dict[str, np.ndarray | sparse.csr_matrix]:
    faces = np.asarray(faces, dtype=np.int64)
    edge_faces: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for face_id, (a, b, c) in enumerate(faces):
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (min(u, v), max(u, v))
            edge_faces.setdefault(key, []).append((face_id, u, v))

    interior = [value for value in edge_faces.values() if len(value) == 2]
    boundary_vertices = sorted({v for edge, value in edge_faces.items() if len(value) == 1 for v in edge})
    directed_edges = np.asarray([(value[0][1], value[0][2]) for value in interior], dtype=np.int64)
    adjacent_faces = np.asarray([(value[0][0], value[1][0]) for value in interior], dtype=np.int64)

    undirected_edges = np.asarray(sorted(edge_faces), dtype=np.int64)
    rows = np.concatenate([undirected_edges[:, 0], undirected_edges[:, 1], np.arange(n_vertices)])
    cols = np.concatenate([undirected_edges[:, 1], undirected_edges[:, 0], np.arange(n_vertices)])
    values = np.ones(len(rows), dtype=np.float64)
    adjacency = sparse.csr_matrix((values, (rows, cols)), shape=(n_vertices, n_vertices))
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    diffusion = sparse.diags(1.0 / np.maximum(degree, 1.0)) @ adjacency

    return {
        "directed_edges": directed_edges,
        "adjacent_faces": adjacent_faces,
        "boundary_vertices": np.asarray(boundary_vertices, dtype=np.int64),
        "interior_vertex_mask": ~np.isin(np.arange(n_vertices), np.asarray(boundary_vertices, dtype=np.int64)),
        "diffusion": diffusion.tocsr(),
    }


def face_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tri = vertices[:, faces]
    cross = np.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0])
    twice_area = np.linalg.norm(cross, axis=-1)
    normals = cross / np.maximum(twice_area[..., None], EPS)
    return normals, 0.5 * twice_area


def oriented_dihedral(
    vertices: np.ndarray,
    faces: np.ndarray,
    directed_edges: np.ndarray,
    adjacent_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    normals, areas = face_geometry(vertices, faces)
    n0 = normals[:, adjacent_faces[:, 0]]
    n1 = normals[:, adjacent_faces[:, 1]]
    edge = vertices[:, directed_edges[:, 1]] - vertices[:, directed_edges[:, 0]]
    edge /= np.maximum(np.linalg.norm(edge, axis=-1, keepdims=True), EPS)
    sine = np.sum(edge * np.cross(n0, n1), axis=-1)
    cosine = np.clip(np.sum(n0 * n1, axis=-1), -1.0, 1.0)
    angle = np.arctan2(sine, cosine)
    adjacent_area = 0.5 * (areas[:, adjacent_faces[:, 0]] + areas[:, adjacent_faces[:, 1]])
    return angle, adjacent_area


def corner_angles_and_cotangents(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tri = vertices[:, faces]
    angles = []
    cotangents = []
    for corner in range(3):
        a = tri[:, :, (corner + 1) % 3] - tri[:, :, corner]
        b = tri[:, :, (corner + 2) % 3] - tri[:, :, corner]
        cross_norm = np.linalg.norm(np.cross(a, b), axis=-1)
        dot = np.sum(a * b, axis=-1)
        angles.append(np.arctan2(cross_norm, dot))
        cotangents.append(dot / np.maximum(cross_norm, EPS))
    _, areas = face_geometry(vertices, faces)
    return np.stack(angles, axis=-1), np.stack(cotangents, axis=-1), areas


def curvature_features(
    vertices: np.ndarray,
    faces: np.ndarray,
    boundary_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch, n_vertices, _ = vertices.shape
    angles, cot, face_areas = corner_angles_and_cotangents(vertices, faces)
    angle_sum = np.zeros((batch, n_vertices), dtype=np.float64)
    vertex_area = np.zeros((batch, n_vertices), dtype=np.float64)
    batch_index = np.arange(batch)[:, None]
    for corner in range(3):
        ids = faces[:, corner][None, :]
        np.add.at(angle_sum, (batch_index, ids), angles[:, :, corner])
        np.add.at(vertex_area, (batch_index, ids), face_areas / 3.0)

    base = np.full((batch, n_vertices), 2.0 * math.pi, dtype=np.float64)
    base[:, boundary_vertices] = math.pi
    gaussian = base - angle_sum

    weighted_curvature = np.zeros_like(gaussian)
    weight_sum = np.zeros_like(gaussian)
    laplace_position = np.zeros_like(vertices)
    # Cotangent at a corner weights the edge opposite that corner.
    for corner in range(3):
        i = faces[:, (corner + 1) % 3]
        j = faces[:, (corner + 2) % 3]
        weight = 0.5 * cot[:, :, corner]
        np.add.at(weighted_curvature, (batch_index, i[None, :]), weight * gaussian[:, j])
        np.add.at(weighted_curvature, (batch_index, j[None, :]), weight * gaussian[:, i])
        np.add.at(weight_sum, (batch_index, i[None, :]), weight)
        np.add.at(weight_sum, (batch_index, j[None, :]), weight)
        delta = vertices[:, j] - vertices[:, i]
        np.add.at(laplace_position, (batch_index, i[None, :], slice(None)), weight[..., None] * delta)
        np.add.at(laplace_position, (batch_index, j[None, :], slice(None)), -weight[..., None] * delta)

    safe_weight = np.where(np.abs(weight_sum) > EPS, weight_sum, 1.0)
    neighbour_gaussian = weighted_curvature / safe_weight
    local_roughness = np.abs(gaussian - neighbour_gaussian)
    mean_curvature = 0.25 * np.linalg.norm(laplace_position, axis=-1) / np.maximum(vertex_area, EPS)
    bbox_diag = np.linalg.norm(vertices.max(axis=1) - vertices.min(axis=1), axis=1)
    mean_curvature *= bbox_diag[:, None]
    return local_roughness, mean_curvature, vertex_area


def fmpd_global_roughness(
    local_roughness: np.ndarray,
    vertex_area: np.ndarray,
    pair_threshold: np.ndarray,
    low: float = 5e-4,
    high: float = 0.2,
    alpha: float = 0.15,
    beta: float = 0.5,
) -> np.ndarray:
    clipped = np.clip(local_roughness, low, high)
    modulated = np.power(clipped, alpha) - low**alpha
    threshold = pair_threshold[:, None]
    final = np.where(modulated > threshold, threshold + beta * (modulated - threshold), modulated)
    return np.sum(final * vertex_area, axis=1) / np.maximum(np.sum(vertex_area, axis=1), EPS)


def fmpd_pair(
    reference_lr: np.ndarray,
    reference_area: np.ndarray,
    distorted_lr: np.ndarray,
    distorted_area: np.ndarray,
) -> np.ndarray:
    def weighted_average(values: np.ndarray, areas: np.ndarray) -> np.ndarray:
        return np.sum(values * areas, axis=1) / np.maximum(np.sum(areas, axis=1), EPS)

    ref_mean = weighted_average(reference_lr, reference_area)
    dist_mean = weighted_average(distorted_lr, distorted_area)
    low, high, alpha = 5e-4, 0.2, 0.15
    ref_mod_mean = np.power(np.clip(ref_mean, low, high), alpha) - low**alpha
    dist_mod_mean = np.power(np.clip(dist_mean, low, high), alpha) - low**alpha
    threshold = np.minimum(ref_mod_mean, dist_mod_mean)
    ref_global = fmpd_global_roughness(reference_lr, reference_area, threshold)
    dist_global = fmpd_global_roughness(distorted_lr, distorted_area, threshold)
    return np.minimum(1.0, 8.0 * np.abs(ref_global - dist_global))


def diffuse(diffusion: sparse.csr_matrix, values: np.ndarray) -> np.ndarray:
    return np.asarray((diffusion @ values.T).T)


def msdm2_style_proxy(
    reference_curvature: np.ndarray,
    distorted_curvature: np.ndarray,
    diffusion: sparse.csr_matrix,
    interior_mask: np.ndarray,
) -> np.ndarray:
    ref = reference_curvature.copy()
    dist = distorted_curvature.copy()
    local_maps = []
    k = 1e-8
    for _ in range(3):
        ref = diffuse(diffusion, ref)
        dist = diffuse(diffusion, dist)
        mu_ref = diffuse(diffusion, ref)
        mu_dist = diffuse(diffusion, dist)
        var_ref = np.maximum(diffuse(diffusion, ref * ref) - mu_ref * mu_ref, 0.0)
        var_dist = np.maximum(diffuse(diffusion, dist * dist) - mu_dist * mu_dist, 0.0)
        std_ref = np.sqrt(var_ref)
        std_dist = np.sqrt(var_dist)
        covariance = diffuse(diffusion, ref * dist) - mu_ref * mu_dist
        luminance = np.abs(mu_ref - mu_dist) / (np.maximum(mu_ref, mu_dist) + k)
        contrast = np.abs(std_ref - std_dist) / (np.maximum(std_ref, std_dist) + k)
        structure = np.abs(std_ref * std_dist - covariance) / (std_ref * std_dist + k)
        local_maps.append(np.clip((luminance + contrast + 0.5 * structure) / 2.5, 0.0, 1.0))
    multiscale = np.mean(local_maps, axis=0)[:, interior_mask]
    return np.power(np.mean(np.power(multiscale, 3.0), axis=1), 1.0 / 3.0)


def flip_counts(source: np.ndarray, distorted: np.ndarray, faces: np.ndarray) -> np.ndarray:
    source_normals, _ = face_geometry(source, faces)
    distorted_normals, _ = face_geometry(distorted, faces)
    return np.sum(np.sum(source_normals * distorted_normals, axis=-1) < 0.0, axis=1).astype(np.int64)


def load_prediction(path: Path, key: str) -> np.ndarray:
    package = np.load(path, allow_pickle=True)
    if key not in package.files:
        raise KeyError(f"{key!r} not found in {path}; available={package.files}")
    return np.asarray(package[key])


def summarize(rows: list[dict[str, float | str]], n_faces: int) -> dict[str, float | int]:
    output: dict[str, float | int] = {"n_pairs": len(rows), "n_roi_faces": int(n_faces)}
    for metric in ("dame_area_proxy", "fmpd_roi_proxy", "msdm2_style_proxy"):
        values = np.asarray([float(row[metric]) for row in rows])
        output[f"{metric}_mean"] = float(values.mean())
        output[f"{metric}_median"] = float(np.median(values))
        output[f"{metric}_p95"] = float(np.percentile(values, 95))
    counts = np.asarray([int(row["normal_flip_face_count"]) for row in rows])
    output.update(
        {
            "normal_flip_face_count_total": int(counts.sum()),
            "normal_flip_face_count_mean": float(counts.mean()),
            "normal_flip_face_count_median": float(np.median(counts)),
            "normal_flip_face_count_p95": float(np.percentile(counts, 95)),
            "normal_flip_face_count_max": int(counts.max()),
            "pairs_with_any_flip": int(np.sum(counts > 0)),
            "pairs_ge_1pct_faces_flipped": int(np.sum(counts >= math.ceil(0.01 * n_faces))),
            "pairs_ge_5pct_faces_flipped": int(np.sum(counts >= math.ceil(0.05 * n_faces))),
            "pairs_ge_10pct_faces_flipped": int(np.sum(counts >= math.ceil(0.10 * n_faces))),
        }
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute ROI-adapted perceptual mesh proxies and absolute flip counts.")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--pairs-key", default="test_pairs")
    parser.add_argument("--method", action="append", nargs=3, metavar=("NAME", "NPZ", "KEY"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    _, by_id = load_rows(args.repo)
    pair_package = np.load(args.predictions, allow_pickle=True)
    pairs = [(str(a), str(b)) for a, b in pair_package[args.pairs_key].tolist()]
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
    template = next(iter(by_id.values()))
    faces = np.asarray(template["faces"], dtype=np.int64)
    n_vertices = len(template["vertices"])
    topo = topology_data(faces, n_vertices)
    boundary = np.asarray(topo["boundary_vertices"])
    interior_mask = np.asarray(topo["interior_vertex_mask"])
    diffusion = topo["diffusion"]
    directed_edges = np.asarray(topo["directed_edges"])
    adjacent_faces = np.asarray(topo["adjacent_faces"])

    unique_targets = sorted({target for _, target in pairs}, key=lambda value: int(value))
    target_cache: dict[str, dict[str, np.ndarray]] = {}
    for target_index, target_id in enumerate(unique_targets, start=1):
        target = np.asarray(by_id[target_id]["vertices"], dtype=np.float64)[None]
        dihedral, dihedral_area = oriented_dihedral(target, faces, directed_edges, adjacent_faces)
        roughness, mean_curvature, vertex_area = curvature_features(target, faces, boundary)
        target_cache[target_id] = {
            "vertices": target[0],
            "dihedral": dihedral[0],
            "dihedral_area": dihedral_area[0],
            "roughness": roughness[0],
            "mean_curvature": mean_curvature[0],
            "vertex_area": vertex_area[0],
        }
        if target_index % 10 == 0 or target_index == len(unique_targets):
            print(
                f"PERCEPTUAL target-cache live {target_index}/{len(unique_targets)}",
                flush=True,
            )

    all_reports: dict[str, dict] = {}
    for method_name, prediction_path_raw, prediction_key in args.method:
        prediction_path = Path(prediction_path_raw)
        pair_path = args.out / f"pair_metrics_{method_name}.csv"
        progress_path = args.out / f"progress_{method_name}.json"
        rows = completed_prefix(pair_path, pairs)
        start_pair = len(rows)
        if start_pair:
            print(
                f"PERCEPTUAL {method_name} resume from Drive {start_pair}/{len(pairs)}",
                flush=True,
            )
        predictions = load_prediction(prediction_path, prediction_key)
        if len(predictions) < len(pairs):
            raise ValueError(f"{method_name}: predictions have {len(predictions)} rows, expected at least {len(pairs)}")
        print(f"PERCEPTUAL {method_name} evaluating {len(pairs)} pairs", flush=True)
        for start in range(start_pair, len(pairs), args.batch_size):
            stop = min(len(pairs), start + args.batch_size)
            batch_pairs = pairs[start:stop]
            source = np.stack([by_id[source_id]["vertices"] for source_id, _ in batch_pairs]).astype(np.float64)
            distorted = source + np.asarray(predictions[start:stop], dtype=np.float64).reshape(stop - start, n_vertices, 3)
            reference = np.stack([target_cache[target_id]["vertices"] for _, target_id in batch_pairs])
            reference_dihedral = np.stack([target_cache[target_id]["dihedral"] for _, target_id in batch_pairs])
            reference_dihedral_area = np.stack([target_cache[target_id]["dihedral_area"] for _, target_id in batch_pairs])
            reference_roughness = np.stack([target_cache[target_id]["roughness"] for _, target_id in batch_pairs])
            reference_curvature = np.stack([target_cache[target_id]["mean_curvature"] for _, target_id in batch_pairs])
            reference_area = np.stack([target_cache[target_id]["vertex_area"] for _, target_id in batch_pairs])

            distorted_dihedral, _ = oriented_dihedral(distorted, faces, directed_edges, adjacent_faces)
            angle_difference = np.abs(np.arctan2(np.sin(distorted_dihedral - reference_dihedral), np.cos(distorted_dihedral - reference_dihedral)))
            masking = np.exp(np.square(Z_DAME * reference_dihedral))
            dame = np.sum(angle_difference * masking * reference_dihedral_area, axis=1) / np.maximum(
                np.sum(reference_dihedral_area, axis=1), EPS
            )
            distorted_roughness, distorted_curvature, distorted_area = curvature_features(distorted, faces, boundary)
            fmpd = fmpd_pair(reference_roughness, reference_area, distorted_roughness, distorted_area)
            msdm2 = msdm2_style_proxy(reference_curvature, distorted_curvature, diffusion, interior_mask)
            flips = flip_counts(source, distorted, faces)

            for local, (source_id, target_id) in enumerate(batch_pairs):
                rows.append(
                    {
                        "pair_index": start + local,
                        "source_id": source_id,
                        "target_id": target_id,
                        "dame_area_proxy": float(dame[local]),
                        "fmpd_roi_proxy": float(fmpd[local]),
                        "msdm2_style_proxy": float(msdm2[local]),
                        "normal_flip_face_count": int(flips[local]),
                        "normal_flip_pct": float(100.0 * flips[local] / len(faces)),
                    }
                )
            batch_number = (stop + args.batch_size - 1) // args.batch_size
            total_batches = (len(pairs) + args.batch_size - 1) // args.batch_size
            if batch_number % 10 == 0 or stop == len(pairs):
                print(
                    f"PERCEPTUAL {method_name} live {stop}/{len(pairs)} "
                    f"({100.0 * stop / len(pairs):.1f}%) batch={batch_number}/{total_batches}",
                    flush=True,
                )
            if stop == len(pairs) or stop // 500 > start // 500:
                write_csv(pair_path, rows)
                atomic_write_json(progress_path, {
                    "status": "COMPLETE" if stop == len(pairs) else "IN_PROGRESS",
                    "method": method_name,
                    "completed_pairs": stop,
                    "total_pairs": len(pairs),
                    "pair_metrics": str(pair_path),
                    "pair_metrics_sha256": sha256_file(pair_path),
                })
                print(
                    f"PERCEPTUAL {method_name} persisted to Drive {stop}/{len(pairs)} "
                    f"({100.0 * stop / len(pairs):.1f}%): {pair_path.name}",
                    flush=True,
                )

        if not rows:
            raise ValueError(f"No perceptual rows were produced for {method_name}")
        if start_pair == len(pairs):
            print(f"PERCEPTUAL {method_name} already complete on Drive", flush=True)
        all_reports[method_name] = {
            "prediction_artifact": prediction_path.name,
            "prediction_key": prediction_key,
            "prediction_sha256": sha256_file(prediction_path),
            "pair_metrics": pair_path.name,
            "summary": summarize(rows, len(faces)),
        }
        del predictions

    report = {
        "status": "roi_adapted_perceptual_proxies_not_official_reference_implementations",
        "protocol": {
            "reference": "target identity ROI mesh",
            "distorted": "source ROI mesh plus predicted dense deformation",
            "same_connectivity": True,
            "n_vertices": n_vertices,
            "n_faces": len(faces),
            "n_interior_edges": len(directed_edges),
            "n_boundary_vertices": len(boundary),
            "pairs_key": args.pairs_key,
            "n_pairs": len(pairs),
        },
        "definitions": {
            "dame_area_proxy": "Oriented interior-edge dihedral difference with reference-angle masking and adjacent target-face area weighting.",
            "fmpd_roi_proxy": "FMPD equations using angle-defect Gaussian curvature, cotangent roughness, saturation/masking modulation and area integration; cropped ROI boundaries use pi angle defect.",
            "msdm2_style_proxy": "Same-connectivity three-scale diffusion approximation to MSDM2 curvature-statistics pooling; not the official projection/geodesic-radius implementation.",
            "normal_flip_face_count": "Exact count over all ROI faces whose source-to-prediction normal dot product is negative.",
        },
        "formula_sources": {
            "dame": DAME_FORMULA,
            "fmpd": FMPD_PAPER,
            "msdm2": MSDM2_PAPER,
        },
        "limitations": [
            "These values are auxiliary within-dataset proxies and have not been calibrated against human judgements for nasal ROIs.",
            "The cropped open ROI requires boundary handling not present in the original closed-mesh benchmarks.",
            "MSDM2-style values must not be labelled as official MSDM2 results.",
        ],
        "methods": all_reports,
    }
    atomic_write_json(args.out / "perceptual_mesh_proxy_summary.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

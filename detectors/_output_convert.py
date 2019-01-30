import numpy as np

from scipy.ndimage import maximum_filter

def non_max_suppression(plain, window_size, threshold):
    under_threshold_indices = plain <= threshold
    plain[under_threshold_indices] = 0
    return plain * (plain == maximum_filter(plain, footprint=np.ones((window_size, window_size))))


def estimate(heat_mat, threshold = 0.5, adaptive_threshold=False):
    if adaptive_threshold:
        _threshold = max(np.min(np.mean(heat_mat, axis=(0, 1)) * 4.0), threshold)
        _threshold = min(_threshold, 0.3)
    else:
        _threshold = threshold

    # extract interesting coordinates using NMS.
    coords = []     # [[coords in plane1], [....], ...]
    for i in range(heat_mat.shape[-1]):
        plain = heat_mat[..., i]
        peaks_mask = (plain - np.mean(plain)) > 0
        if int(np.sum(peaks_mask)) > 0:
            nms = non_max_suppression(plain, 3, _threshold)
            y, x = np.where(nms > _threshold)
            coords.append(list(zip(list(x), list(y))))
        else:
            coords.append([])
        
    return coords

def get_score(x1, y1, x2, y2, paf_mat_x, paf_mat_y):
    import math
    
    Local_PAF_Threshold = 0.2
    __num_inter = 10
    __num_inter_f = float(__num_inter)
    dx, dy = x2 - x1, y2 - y1
    normVec = math.sqrt(dx ** 2 + dy ** 2)

    if normVec < 1e-4:
        return 0.0, 0

    vx, vy = dx / normVec, dy / normVec

    xs = np.arange(x1, x2, dx / __num_inter_f) if x1 != x2 else np.full((__num_inter,), x1)
    ys = np.arange(y1, y2, dy / __num_inter_f) if y1 != y2 else np.full((__num_inter,), y1)
    xs = (xs + 0.5).astype(np.int8)
    ys = (ys + 0.5).astype(np.int8)

    # without vectorization
    pafXs = np.zeros(__num_inter)
    pafYs = np.zeros(__num_inter)
    for idx, (mx, my) in enumerate(zip(xs, ys)):
        pafXs[idx] = paf_mat_x[my][mx]
        pafYs[idx] = paf_mat_y[my][mx]

    # vectorization slow?
    # pafXs = pafMatX[ys, xs]
    # pafYs = pafMatY[ys, xs]

    local_scores = pafXs * vx + pafYs * vy
    thidxs = local_scores > Local_PAF_Threshold

    return sum(local_scores * thidxs), sum(thidxs)

def score_pairs(e, i, j, points, L, S, rescale=(1.0, 1.0)):
    part_idx1 = i
    part_idx2 = j
    coord_list1= points[i]
    coord_list2 = points[j] 
    paf_mat_x = L[..., e, 0]
    paf_mat_y = L[..., e, 1]
    heatmap = S
    connection_temp = []
    PAF_Count_Threshold = 1
    cnt = 0
    for idx1, (x1, y1) in enumerate(coord_list1):
        for idx2, (x2, y2) in enumerate(coord_list2):
            score, count = get_score(x1, y1, x2, y2, paf_mat_x, paf_mat_y)
            cnt += 1
            if count < PAF_Count_Threshold or score <= 0.0:
                continue
            connection_temp.append(dict(
                score=score,
                joint_start=part_idx1, joint_end=part_idx2,
                point_ind_start=idx1, point_ind_end=idx2,
                coord1=(x1 * rescale[0], y1 * rescale[1]),
                coord2=(x2 * rescale[0], y2 * rescale[1]),
                score1=heatmap[y1, x1, part_idx1],
                score2=heatmap[y2, x2, part_idx2],
            ))

    connection = []
    used_idx1, used_idx2 = set(), set()
    for candidate in sorted(connection_temp, key=lambda x: x['score'], reverse=True):
        # check not connected
        if candidate['point_ind_start'] in used_idx1 or candidate['point_ind_end'] in used_idx2:
            continue
        connection.append(candidate)
        used_idx1.add(candidate['point_ind_start'])
        used_idx2.add(candidate['point_ind_end'])

    return connection

def get_merges_list(humans, PARTS_LIST):
    merges_list = []
    for h1 in range(len(humans)-1):
        for h2 in range(h1+1, len(humans)):
            for i in range(len(PARTS_LIST)):
                if humans[h1][i] == humans[h2][i] and humans[h1][i] is not None and humans[h2][i] is not None:
                    merges_list.append((i, (h1, h2)))
    return merges_list

def get_humans(S, L, BODY_EDGES, PARTS_LIST):
    points = estimate(S)
    edges = []

    for e, (i,j) in enumerate(BODY_EDGES):
        edges.append(score_pairs(e, i, j, points, L, S))
    
    humans = []
    for e, edge in enumerate(edges):
        used_edge_idxs = []
        for i, human_edge in enumerate(edge):
            for j, human in enumerate(humans):
                if human[human_edge['joint_start']] == human_edge['point_ind_start']:
                    human[human_edge['joint_end']] = human_edge['point_ind_end']

                    used_edge_idxs.append(i)
                    break

        for i, human_edge in enumerate(edge):
            if i in used_edge_idxs:
                continue
            else:
                humans.append([None]*len(PARTS_LIST))
                humans[-1][human_edge['joint_start']] = human_edge['point_ind_start']
                humans[-1][human_edge['joint_end']] = human_edge['point_ind_end']
    
    merges_list = get_merges_list(humans, PARTS_LIST)
    while merges_list:
        new_humans = []
        for i, (h1, h2) in merges_list:
            tmp = humans[h1]
            humans[h1] = []
            h1 = tmp

            tmp = humans[h2]
            humans[h2] = []
            h2 = tmp

            if not h1 or not h2:
                if h1:
                    new_humans.append(h1)
                if h2:
                    new_humans.append(h2)
                continue

            r_list = []
            
            for j in range(len(PARTS_LIST)):
                if h1[j] is None or j == i: 
                    r_list.append(h2[j])
                    
                elif h2[j] is None:
                    r_list.append(h1[j])
                else:
                    print(j)
                    print("h1:", h1)
                    print("h2:", h2)
                    raise ValueError()
            new_humans.append(r_list)

        humans = new_humans

        merges_list = get_merges_list(humans, PARTS_LIST)
    
    humans = [[(points[i][j] if j is not None else None) for i, j in enumerate(person)] for person in humans]
    return humans
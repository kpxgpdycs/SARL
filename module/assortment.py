import numpy as np

def assortment(products, capacity):
    """
    Based on the given product list and capacity, select an optimal product to maximize profits.
    
    parameters:
    products (numpy.ndarray): feature vectore.
    capacity (int): the capacity of the assortment.
    
    returns:
    tuple: 
        - selected (list): the selected products.
        - opt (float): optimal profit.
        - p (list): the probability of each product being selected.
    """

    # add a no choice with zero profit and weight
    products = np.vstack([products, [0, 0]])
    n = products.shape[0]

    # calculate the intersection of each pair of products
    intersections = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            vi, wi = products[i, 0], products[i, 1]
            vj, wj = products[j, 0], products[j, 1]
            if vi == vj:
                continue
            x = (vi * wi - vj * wj) / (vi - vj)
            y = vi * (wi - x)
            intersections.append((i, j, x, y))

    # sort steps
    intersections.sort(key=lambda x: x[2])
    rank = [{
        'order': np.argsort(-products[:, 0]).tolist(),
        'x_start': -np.inf,
        'x_end': np.inf
    }]
    current_order = rank[0]['order'].copy()
    for intersect in intersections:
        i, j, x_intersect, _ = intersect
        vi = products[i, 0]
        vj = products[j, 0]
        wi = products[i, 1]
        wj = products[j, 1]
        x_test = x_intersect + 1e-9 
        term_i = vi * (wi - x_test)
        term_j = vj * (wj - x_test)
        
        # exchange condition
        try:
            idx_i = current_order.index(i)
            idx_j = current_order.index(j)
        except ValueError:
            continue
        if term_i > term_j and idx_j < idx_i:
            new_order = current_order.copy()
            new_order[idx_j], new_order[idx_i] = new_order[idx_i], new_order[idx_j]
            rank[-1]['x_end'] = x_intersect
            rank.append({
                'order': new_order,
                'x_start': x_intersect,
                'x_end': np.inf
            })
            current_order = new_order

    # search for the best solution
    max_reward = -np.inf
    best_selected = None
    best_p = None
    for interval in rank:
        order = interval['order']
        x_start = interval['x_start']
        x_end = interval['x_end']
        for k in range(0, min(capacity, len(order)) + 1):
            if k == 0:
                x_candidate = 0  
            else:
                selected = order[:k]
                sum_v = products[selected, 0].sum()
                sum_vw = (products[selected, 0] * products[selected, 1]).sum()
                if sum_v == 0:
                    continue
                x_candidate = sum_vw / (sum_v + 1)
            if not (x_start <= x_candidate < x_end):
                continue
            valid = True
            selected_indices = []
            if k > 0:
                for idx in order[:k]:
                    term = products[idx, 0] * (products[idx, 1] - x_candidate)
                    if term <= 0:
                        valid = False
                        break
                selected_indices = order[:k]
                if valid and k < len(order):
                    for idx in order[k:]:
                        term = products[idx, 0] * (products[idx, 1] - x_candidate)
                        if term > 0:
                            valid = False
                            break
                            
            if valid:
                # calculate the profit
                sum_v_sel = products[selected_indices, 0].sum()
                sum_v_all = products[:, 0].sum()
                
                p = np.zeros(n)
                for idx in range(n):
                    if idx in selected_indices:
                        p[idx] = products[idx, 0] / (1 + sum_v_sel)
                    else:
                        p[idx] = (products[idx, 0] / (1 + sum_v_all)) * (1 / (1 + sum_v_sel))
                p[-1] = 1 - p[:-1].sum()  
                
                current_reward = np.dot(p, products[:, 1])
                
                # update the best solution
                if current_reward > max_reward:
                    max_reward = current_reward
                    best_selected = np.zeros(n, dtype=int)
                    best_selected[selected_indices] = 1
                    best_p = p
    best_selected = best_selected.tolist()
    best_selected_index = [ i for i, x in enumerate(best_selected) if x == 1]


    return best_selected_index, max_reward, best_p.tolist()


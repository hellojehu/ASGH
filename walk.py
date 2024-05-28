import numpy as np
import networkx as nx
import random
import math

class RWGraph():
    def __init__(self, dic_G):
        self.G = dic_G

    def find_min_indices(self, lst):
        sorted_lst = sorted(enumerate(lst), key=lambda x: x[1])
        return [x[0] for x in sorted_lst[:5]]
        
    def walk_new(self, walk_length, start, alpha, target_type, type_att, vectors):
        G = self.G
        length = 0
        change_probability = 0
        
        type = list()
        type.append(start)
        walk = [start]
        while len(walk) < walk_length:
            cur = walk[-1]
            candidates = []
            if cur[0] == target_type:
                change_probability = 1 - math.pow(alpha, length)
                r = random.uniform(0, 1)
                order = list(type_att[cur])
                rr = random.uniform(0, 1)
                if r > change_probability:
                    if order[0] > rr:
                        candidates.extend([e for e in G[cur] if (e[0] == 'd')])
                        if not candidates:
                            candidates.extend([e for e in G[cur] if (e[0] == 'a')])
                    else:
                        candidates.extend([e for e in G[cur] if (e[0] == 'a')])
                        if not candidates:
                            candidates.extend([e for e in G[cur] if (e[0] == 'd')])

                else:
                    if order[0] > rr:
                        if 'd' not in type:
                            candidates.extend([e for e in G[cur] if (e[0] == 'd')])
                        if not candidates:
                            if 'a' not in type:
                                candidates.extend([e for e in G[cur] if (e[0] == 'a')])
                    else:
                        if 'a' not in type:
                            candidates.extend([e for e in G[cur] if (e[0] == 'a')])
                        if not candidates:
                            if 'd' not in type:
                                candidates.extend([e for e in G[cur] if (e[0] == 'd')])
                if candidates: 
                    now_v = walk[-1]
                    vec_now_v = np.array(vectors[str(now_v)])
                    top = []
                    vec_v= [vectors[str(v)] for v in candidates]

                    vec_v_array = np.array(vec_v)
                    two_vec_v = [] 
                    for point in candidates:
                        two_v = list(G[point].keys())
                        length = len(two_v)
                        two_v_vec = np.array(0)
                        v_vectors = [vectors[str(x)] for x in two_v]
                        vectors_array = np.array(v_vectors)
                        for i in vectors_array:
                            two_v_vec = two_v_vec + i
                        every_two_v = np.divide(two_v_vec, length)
                        two_vec_v.append(list(every_two_v))
                    vectors_new = np.array(two_vec_v) * 0.1 + vec_v_array * 0.9
                    distances = []
                    for i in range(vectors_new.shape[0]):
                        cos_sim = 1 - (np.dot(vec_now_v, vectors_new[i]) / (np.linalg.norm(vec_now_v) * np.linalg.norm(vectors_new[i])))
                        distances.append(cos_sim)
                    min_indices = self.find_min_indices(distances)
                    for index in min_indices:
                        top.append(str(candidates[index]))
                    next = random.choice(top)
                    if next[0] == type[-1]:
                        length = length + 1
                    else:
                        type[:1] = []
                        type.append(next[0])
                        length = 0
                    walk.append(next)
                else:
                    break
            else:
                candidates.extend([e for e in G[cur]])
                if candidates:
                    next = random.choice(candidates)
                    walk.append(next)
                else:
                    break
        return walk
    
    
    def walk(self, walk_length, start, alpha, target_type, type_att):
        # Simulate a random walk starting from start node.
        G = self.G
        length = 0
        change_probability = 0
        
        type = list() # 游走序列
        type.append(start)
        walk = [start]
        while len(walk) < walk_length:
            cur = walk[-1]
            candidates = []
            if cur[0] == target_type:
                change_probability = 1 - math.pow(alpha, length)
                r = random.uniform(0, 1)
                order = list(type_att[cur])
                rr = random.uniform(0, 1)
                if r > change_probability:
                    if order[0] > rr:
                        candidates.extend([e for e in G[cur] if (e[0] == 'd')])
                        if not candidates:
                            candidates.extend([e for e in G[cur] if (e[0] == 'a')])
                    else:
                        candidates.extend([e for e in G[cur] if (e[0] == 'a')])
                        if not candidates:
                            candidates.extend([e for e in G[cur] if (e[0] == 'd')])

                else:
                    if order[0] > rr:
                        if 'd' not in type:
                            candidates.extend([e for e in G[cur] if (e[0] == 'd')])
                        if not candidates:
                            if 'a' not in type:
                                candidates.extend([e for e in G[cur] if (e[0] == 'a')])
                    else:
                        if 'a' not in type:
                            candidates.extend([e for e in G[cur] if (e[0] == 'a')])
                        if not candidates:
                            if 'd' not in type:
                                candidates.extend([e for e in G[cur] if (e[0] == 'd')])
                if candidates:
                    weights = []
                    next_list = []
                    for candidate in candidates:
                        weights.append((candidate,G.degree(candidate)))
                    weights = sorted(weights,key=lambda x:x[1],reverse = True)
                    for i in range(int(len(weights)/3)+1):
                        next_list.extend([weights[i][0]])
                    next = random.choice(next_list)
                    if next[0] == type[-1]:
                        length = length + 1
                    else:
                        type[:1] = []
                        type.append(next[0])
                        length = 0
                    walk.append(next)
                else:
                    break
            else:
                candidates.extend([e for e in G[cur]])
                if candidates:
                    next = random.choice(candidates)
                    walk.append(next)
                else:
                    break
        return walk

    
    def simulate_walks_new(self, num_walks, walk_length, alpha, type, type_att, vectors):
        G = self.G
        walks = []
        paths = []
        all_walks = []
        nodes = list(n for n in G.nodes())
        for walk_iter in range(num_walks):
            print('---walk num:', walk_iter)
            random.shuffle(nodes)
            for node in nodes:
                walk = self.walk_new(walk_length=walk_length, start=node, alpha = alpha, target_type = type, type_att = type_att,vectors = vectors)
                all_walks.append([str(n) for n in walk])
                walkn = []
                pathn = []
                for n in walk:
                    if n[0] == type:
                        walkn.append(n)
                    else:
                        pathn.append(n[0])
                walks.append([str(n) for n in walkn])
                paths.append([str(t) for t in pathn])
        return all_walks, walks, paths
    
    def simulate_walks(self, num_walks, walk_length, alpha, type, type_att):
        G = self.G
        walks = []
        paths = []
        all_walks = []
        nodes = list(n for n in G.nodes())
        for walk_iter in range(num_walks):
            print('---walk num:', walk_iter)
            random.shuffle(nodes)
            for node in nodes:
                walk = self.walk(walk_length=walk_length, start=node, alpha = alpha, target_type = type, type_att = type_att)
                all_walks.append([str(n) for n in walk])
                walkn = []
                pathn = []
                for n in walk:
                    if n[0] == type:
                        walkn.append(n)
                    else:
                        pathn.append(n[0])
                walks.append([str(n) for n in walkn])
                paths.append([str(t) for t in pathn])

        return all_walks, walks, paths



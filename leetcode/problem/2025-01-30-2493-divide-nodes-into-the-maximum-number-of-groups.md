---
layout: leetcode-entry
title: "2493. Divide Nodes Into the Maximum Number of Groups"
permalink: "/leetcode/problem/2025-01-30-2493-divide-nodes-into-the-maximum-number-of-groups/"
leetcode_ui: true
entry_slug: "2025-01-30-2493-divide-nodes-into-the-maximum-number-of-groups"
---

[2493. Divide Nodes Into the Maximum Number of Groups](https://leetcode.com/problems/divide-nodes-into-the-maximum-number-of-groups/description/) hard
[blog post](https://leetcode.com/problems/divide-nodes-into-the-maximum-number-of-groups/solutions/6347637/kotlin-rust-by-samoylenkodmitry-r9j8/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/30012025-2493-divide-nodes-into-the?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/2w-oQUz4w6E)
![1.webp](/assets/leetcode_daily_images/8064c763.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/880

#### Problem TLDR

Max count of bipartitions in a graph #hard #bfs #graph

#### Intuition

Didn't solve without the hints.
Hints:
* know how to bipartite: assign colors in BFS, check if no siblings match
* the n <= 500, check every possible start node to find the longest path

#### Approach

* don't forget disconnected nodes
* we can skip using Union-Find: just increase the groups counter

#### Complexity

- Time complexity:
$$O(V(V + E))$$, (V + E) for BFS, n = V times

- Space complexity:
$$O(V)$$

#### Code

```kotlin

    fun magnificentSets(n: Int, edges: Array<IntArray>): Int {
        val g = Array(n + 1) { ArrayList<Int>() }; for ((a, b) in edges) { g[a] += b; g[b] += a }
         val group = IntArray(n + 1); val gs = arrayListOf(0)
        for (start in 1..n) {
            if (group[start] < 1) gs += 0; val color = IntArray(n + 1);
            val q = ArrayDeque<Int>(listOf(start)); color[start] = 1; var lvl = 0
            while (q.size > 0 && ++lvl > 0) { repeat(q.size) {
                val u = q.removeFirst(); if (group[u] < 1) group[u] = gs.lastIndex
                for (v in g[u]) if (color[v] < 1) { color[v] = 3 - color[u]; q += v }
                    else if (color[v] == color[u]) return -1
            }}
            gs[group[start]] = max(gs[group[start]], lvl)
        }
        return gs.sum()
    }

```
```rust

    pub fn magnificent_sets(n: i32, edges: Vec<Vec<i32>>) -> i32 {
        let n = (n + 1) as usize; let mut g = vec![vec![]; n];
        for e in edges { let (a, b) = (e[0] as usize, e[1] as usize); g[a].push(b); g[b].push(a)}
        let mut group = vec![0; n]; let mut gs = vec![0];
        for start in 1..n {
            if group[start] < 1 { gs.push(0) }; let mut color = vec![0; n];
            let mut q = VecDeque::from([start]); color[start] = 1; let mut lvl = 0;
            while q.len() > 0 { for _ in 0..q.len() {
                let u = q.pop_front().unwrap(); if group[u] < 1 { group[u] = gs.len() - 1 }
                for &v in &g[u] { if color[v] < 1 { color[v] = 3 - color[u]; q.push_back(v) }
                    else if color[v] == color[u] { return -1 }}
            }; lvl += 1; }
            gs[group[start]] = lvl.max(gs[group[start]])
        }
        gs.iter().sum::<usize>() as i32
    }

```
```c++

    int magnificentSets(int n, vector<vector<int>>& edges) {
        int group[501] = {}, gs[501] = {}, q[501], res = 0; vector<int> g[501];
        for (auto& e: edges) g[e[0]].push_back(e[1]), g[e[1]].push_back(e[0]);
        for (int start = 1; start <= n; ++start) {
            if (!group[start]) gs[++gs[0]] = 0;
            int color[501] = {}, l = 0, r = 0, lvl = 0;
            q[r++] = start, color[start] = 1;
            while (l < r && ++lvl) for (int k = r - l; k--;) {
                int u = q[l++];
                if (!group[u]) group[u] = gs[0];
                for (int v: g[u])
                    if (!color[v]) color[v] = 3 - color[u], q[r++] = v;
                    else if (color[v] == color[u]) return -1;
            }
            if (lvl > gs[group[start]]) res += lvl - exchange(gs[group[start]], lvl);
        }
        return res;
    }

```


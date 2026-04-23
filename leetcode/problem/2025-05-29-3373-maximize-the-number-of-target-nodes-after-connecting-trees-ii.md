---
layout: leetcode-entry
title: "3373. Maximize the Number of Target Nodes After Connecting Trees II"
permalink: "/leetcode/problem/2025-05-29-3373-maximize-the-number-of-target-nodes-after-connecting-trees-ii/"
leetcode_ui: true
entry_slug: "2025-05-29-3373-maximize-the-number-of-target-nodes-after-connecting-trees-ii"
---

[3373. Maximize the Number of Target Nodes After Connecting Trees II](https://leetcode.com/problems/maximize-the-number-of-target-nodes-after-connecting-trees-ii/description) hard
[blog post](https://leetcode.com/problems/maximize-the-number-of-target-nodes-after-connecting-trees-ii/solutions/6792086/kotlin-rust-by-samoylenkodmitry-l1y8/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29052025-3373-maximize-the-number?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/PblcK9Ah8J0)
![1.webp](/assets/leetcode_daily_images/b0e7d550.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1003

#### Problem TLDR

Max even-edged siblings after merging trees #hard #graph

#### Intuition

![2.png](/assets/leetcode_daily_images/5cfa0fe6.webp)

    1. node either in the odd or even set
    2. mark nodes, calculate count_odd, count_even

#### Approach

* track parent or just check mark[y] == 0
* use DFS or BFS

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 110ms
    fun maxTargetNodes(e1: Array<IntArray>, e2: Array<IntArray>): IntArray {
        val (cm1, cm2) = listOf(e1, e2).map { e ->
            val g = Array(e.size + 1) { ArrayList<Int>() }
            for ((a, b) in e) { g[a] += b; g[b] += a }
            val m = IntArray(g.size); val c = IntArray(3)
            fun dfs(x: Int, o: Int) {
                m[x] = o; c[o]++; for (y in g[x]) if (m[y] < 1) dfs(y, 3 - o) }
            dfs(0, 1); c to m
        }
        val (c1, m1) = cm1; val (c2, m2) = cm2; val cmax = c2.max()
        return IntArray(m1.size) { c1[m1[it]] + cmax }
    }

```
```rust

// 63ms
    pub fn max_target_nodes(e1: Vec<Vec<i32>>, e2: Vec<Vec<i32>>) -> Vec<i32> {
        let [(c1, m1), (c2, m2)] = [e1, e2].map(|e| {
            let (mut g, mut m, mut c, mut q, mut q1, mut o) =
                (vec![vec![]; e.len() + 1], vec![0; e.len() + 1], [0; 3], vec![0], vec![], 1);
            for e in e { let (a, b) = (e[0] as usize, e[1] as usize);
                g[a].push(b); g[b].push(a) }
            while q.len() > 0 { for &x in &q {
                m[x] = o; c[o] += 1; for &y in &g[x] { if m[y] < 1 { q1.push(y) }}}
                (q, q1) = (q1, q); o = 3 - o; q1.clear()
            }; (c, m)
        });
        let cmax = c2[1].max(c2[2]); m1.into_iter().map(|m1| c1[m1] + cmax).collect()
    }

```
```c++

// 277ms
    vector<int> maxTargetNodes(vector<vector<int>>& e1, vector<vector<int>>& e2) {
        auto f = [&](auto& e){
            vector<vector<int>> g(size(e) + 1);
            for (auto& p: e) g[p[0]].push_back(p[1]), g[p[1]].push_back(p[0]);
            vector<int> m(size(g)), c(3); queue<int> q; q.push(0); m[0] = 1; c[1]++;
            while (size(q)) {
                int u = q.front(); q.pop();
                for(int v: g[u]) if(!m[v]) m[v] = 3 - m[u], c[m[v]]++, q.push(v); }
            return pair {c, m};
        };
        auto [c1, m1] = f(e1); auto [c2, m2] = f(e2);
        int cm = max(c2[1], c2[2]); vector<int> r(size(m1));
        for (int i = 0; i < size(m1); ++i) r[i] = c1[m1[i]] + cm;
        return r;
    }

```


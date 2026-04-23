---
layout: leetcode-entry
title: "3372. Maximize the Number of Target Nodes After Connecting Trees I"
permalink: "/leetcode/problem/2025-05-28-3372-maximize-the-number-of-target-nodes-after-connecting-trees-i/"
leetcode_ui: true
entry_slug: "2025-05-28-3372-maximize-the-number-of-target-nodes-after-connecting-trees-i"
---

[3372. Maximize the Number of Target Nodes After Connecting Trees I](https://leetcode.com/problems/maximize-the-number-of-target-nodes-after-connecting-trees-i/description/) medium
[blog post](https://leetcode.com/problems/maximize-the-number-of-target-nodes-after-connecting-trees-i/solutions/6788847/kotlin-rust-by-samoylenkodmitry-bl4z/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28052025-3372-maximize-the-number?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/_USSRr0B6SU)
![1.webp](/assets/leetcode_daily_images/b4518794.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1002

#### Problem TLDR

Max k-reachable nodes of merged trees #medium #dfs

#### Intuition

The brute-force DFS is accepted.

Chain-of-thoughts:

```j

    // for k - 1
    // find the most optimal spot on edges2
    // how many nodes can be (k-1) reached from each node
    // solve same problem for edges1(k) and edges2(k-1)
    // 1000 edges, brute-force bfs from each 1000*k, n^2
    // lets write brute-force, no good ideas (23 minutes)

```

#### Approach

* sometimes it is better to start with brute-force, then spending too much time thinking about a better algorithm

#### Complexity

- Time complexity:
$$O(n^2)$$, n = 1000

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 147ms
    fun maxTargetNodes(e1: Array<IntArray>, e2: Array<IntArray>, k: Int): IntArray {
        val (g1, g2) = listOf(e1, e2).map { e ->
            Array(e.size + 1) { ArrayList<Int>() }.also { g ->
                for ((a, b) in e) { g[a] += b; g[b] += a }}}
        fun dfs(x: Int, g: Array<ArrayList<Int>>, p: Int, k: Int): Int =
            if (k < 0) 0 else 1 + g[x].sumOf { if (it == p) 0 else dfs(it, g, x, k - 1) }
        val cnt2 = g2.indices.maxOf { dfs(it, g2, -1, k - 1) }
        return IntArray(g1.size) { cnt2 + dfs(it, g1, -1, k) }
    }

```
```rust

// 67ms
    pub fn max_target_nodes(e1: Vec<Vec<i32>>, e2: Vec<Vec<i32>>, k: i32) -> Vec<i32> {
        let [g1, g2] = [e1, e2].map(|e| { let mut g = vec![vec![]; e.len() + 1];
            for e in e { let (a, b) = (e[0] as usize, e[1] as usize);
            g[a].push(b); g[b].push(a) }; g });
        fn dfs(x: usize, g: &[Vec<usize>], p: usize, k: i32) -> i32 {
            if k < 0 { 0 } else {
                1 + g[x].iter().map(|&s| if s == p { 0 } else { dfs(s, g, x, k - 1)}).sum::<i32>() }}
        let cnt2 = (0..g2.len()).map(|x| dfs(x, &g2, 1001, k - 1)).max().unwrap();
        (0..g1.len()).map(|x| cnt2 + dfs(x, &g1, 1001, k)).collect()
    }

```
```c++

// 95ms
    vector<int> maxTargetNodes(vector<vector<int>>& e1, vector<vector<int>>& e2, int k) {
        int n = size(e1) + 1, m = size(e2) + 1, c2 = 0; vector<vector<int>> g1(n), g2(m);
        for (auto& e: e1) g1[e[0]].push_back(e[1]), g1[e[1]].push_back(e[0]);
        for (auto& e: e2) g2[e[0]].push_back(e[1]), g2[e[1]].push_back(e[0]);
        auto dfs = [&](this const auto& dfs, int x, vector<vector<int>>& g, int p, int k) -> int {
            if (k < 0) return 0; int cnt = 1; for (int s: g[x]) if (s != p) cnt += dfs(s, g, x, k - 1);
            return cnt;
        }; vector<int> r(n);
        for (int i = 0; i < m; ++i) c2 = max(c2, dfs(i, g2, -1, k - 1));
        for (int i = 0; i < n; ++i) r[i] = c2 + dfs(i, g1, -1, k); return r;
    }

```


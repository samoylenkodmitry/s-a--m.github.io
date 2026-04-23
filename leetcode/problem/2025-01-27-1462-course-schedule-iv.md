---
layout: leetcode-entry
title: "1462. Course Schedule IV"
permalink: "/leetcode/problem/2025-01-27-1462-course-schedule-iv/"
leetcode_ui: true
entry_slug: "2025-01-27-1462-course-schedule-iv"
---

[1462. Course Schedule IV](https://leetcode.com/problems/course-schedule-iv/description/) medium
[blog post](https://leetcode.com/problems/course-schedule-iv/solutions/6334635/kotlin-rust-by-samoylenkodmitry-by2j/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27012025-1462-course-schedule-iv?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Vzksn6304Fw)
![1.webp](/assets/leetcode_daily_images/8b08113d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/877

#### Problem TLDR

All innodes for each query in graph #medium #dfs #toposort #floyd_warshall

#### Intuition

For each node, we should know all the incoming nodes.
Several ways:
* Depth-First Search and cache the results (Kotlin)
* Floyd-Warshall: if i->k and j->k then i->j (Rust)
* Topological Sorting: put zero-incoming nodes into queue, connect siblings (c++)

#### Approach

* the hardest to grasp is the toposort one: if node i->q then i->q.sibling

#### Complexity

- Time complexity:
$$O(n^2 + q + p)$$, O(n^3 + q + p) for Floyd-Warshall

- Space complexity:
$$O(n^2 + q)$$

#### Code

```kotlin

    fun checkIfPrerequisite(n: Int, pre: Array<IntArray>, q: Array<IntArray>) = {
        val g = pre.groupBy { it[1] }; val dp = HashMap<Int, Set<Int>>()
        fun dfs(i: Int): Set<Int> = dp.getOrPut(i) {
            ((g[i]?.map { dfs(it[0]) }?.flatten() ?: setOf()) + i).toSet()
        }
        q.map { (a, b) -> a in dfs(b) }
    }()

```
```rust

    pub fn check_if_prerequisite(n: i32, pre: Vec<Vec<i32>>, q: Vec<Vec<i32>>) -> Vec<bool> {
        let n = n as usize; let mut p = vec![vec![0; n]; n];
        for e in pre { p[e[1] as usize][e[0] as usize] = 1 }
        for k in 0..n { for i in 0..n { for j in 0..n {
          if p[i][k] * p[k][j] > 0 { p[i][j] = 1 }}}}
        q.iter().map(|e| p[e[1] as usize][e[0] as usize] > 0).collect()
    }

```
```c++

    vector<bool> checkIfPrerequisite(int n, vector<vector<int>>& pre, vector<vector<int>>& qw) {
        vector<vector<int>> g(n); vector<int> ind(n); bool dp[100][100] = {}; queue<int> q;
        for (auto& p: pre) g[p[0]].push_back(p[1]), ind[p[1]]++, dp[p[0]][p[1]] = 1;
        for (int i = 0; i < n; ++i) if (!ind[i]) q.push(i);
        while (size(q)) { for (int v: g[q.front()]) {
            for (int i = 0; i < n; ++i) if (dp[i][q.front()]) dp[i][v] = 1;
            if (!--ind[v]) q.push(v);
        } q.pop(); }
        vector<bool> r; for (auto& x: qw) r.push_back(dp[x[0]][x[1]]); return r;
    }

```


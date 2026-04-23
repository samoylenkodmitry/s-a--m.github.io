---
layout: leetcode-entry
title: "2872. Maximum Number of K-Divisible Components"
permalink: "/leetcode/problem/2024-12-21-2872-maximum-number-of-k-divisible-components/"
leetcode_ui: true
entry_slug: "2024-12-21-2872-maximum-number-of-k-divisible-components"
---

[2872. Maximum Number of K-Divisible Components](https://leetcode.com/problems/maximum-number-of-k-divisible-components/description/) hard
[blog post](https://leetcode.com/problems/maximum-number-of-k-divisible-components/solutions/6170131/kotlin-rust-by-samoylenkodmitry-adt0/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21122024-2872-maximum-number-of-k?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/7Dz_YsHUtaI)
[deep-dive](https://notebooklm.google.com/notebook/006a7ebe-578c-4121-a202-88e83dccbdb3/audio)
![1.webp](/assets/leetcode_daily_images/923a42ed.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/839

#### Problem TLDR

Max connected components divisible by `k` in graph #hard #toposort

#### Intuition

Can't solve without hints.
The hints: walk from any node, merge values if sum is not divisible by `k`.

If we go from each leaf up to the parent, we can compute the sum of this parent.

#### Approach

* we can walk with DFS
* we can walk with BFS, doing the Topological Sorting algorithm: decrease in-degrees, add in-degrees of `1`
* we can use `values` as a sum results holder

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun maxKDivisibleComponents(n: Int, edges: Array<IntArray>, values: IntArray, k: Int): Int {
        val g = Array(n) { ArrayList<Int>() }; for ((a, b) in edges) { g[a] += b; g[b] += a }
        fun dfs(crr: Int, frm: Int): Int =
            g[crr].sumOf { nxt ->
                if (nxt == frm) 0 else dfs(nxt, crr).also { values[crr] += values[nxt] % k }
            } + if (values[crr] % k > 0) 0 else 1
        return dfs(0, 0)
    }

```
```rust

    pub fn max_k_divisible_components(n: i32, edges: Vec<Vec<i32>>, mut values: Vec<i32>, k: i32) -> i32 {
        let (mut cnt, mut g, mut deg) = (0, vec![vec![]; n as usize], vec![0; n as usize]);
        for e in edges {  let (u, v) = (e[0] as usize, e[1] as usize);
            deg[u] += 1; deg[v] += 1; g[u].push(v); g[v].push(u) }
        let mut q = VecDeque::from_iter((0..n as usize).filter(|&u| deg[u] < 2));
        while let Some(u) = q.pop_front() {
            deg[u] -= 1; if values[u] % k == 0 { cnt += 1 }
            for &v in &g[u] {
                if deg[v] < 1 { continue }
                deg[v] -= 1; values[v] += values[u] % k;
                if deg[v] == 1 { q.push_back(v); }
            }
        }; cnt
    }

```
```c++

    int maxKDivisibleComponents(int n, vector<vector<int>>& edges, vector<int>& values, int k) {
        vector<vector<int>> g(n); vector<int> deg(n); int res = 0; queue<int> q;
        for (auto e: edges) g[e[0]].push_back(e[1]), g[e[1]].push_back(e[0]);
        for (int i = 0; i < n; ++i) if ((deg[i] = g[i].size()) < 2) q.push(i);
        while (q.size()) {
            int u = q.front(); q.pop(); --deg[u];
            res += values[u] % k == 0;
            for (int v: g[u]) if (deg[v]) {
                values[v] += values[u] % k;
                if (--deg[v] == 1) q.push(v);
            }
        } return res;
    }

```


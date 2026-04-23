---
layout: leetcode-entry
title: "2467. Most Profitable Path in a Tree"
permalink: "/leetcode/problem/2025-02-24-2467-most-profitable-path-in-a-tree/"
leetcode_ui: true
entry_slug: "2025-02-24-2467-most-profitable-path-in-a-tree"
---

[2467. Most Profitable Path in a Tree](https://leetcode.com/problems/most-profitable-path-in-a-tree/description/) medium
[blog post](https://leetcode.com/problems/most-profitable-path-in-a-tree/solutions/6461874/kotlin-rust-by-samoylenkodmitry-xfcd/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24022025-2467-most-profitable-path?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/p8ysMxon3NI)
![1.webp](/assets/leetcode_daily_images/f8585860.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/905

#### Problem TLDR

Max Alice path down, Bob up in tree #medium #dfs

#### Intuition

Build a graph, then traverse. Bob only goes up, so we can use parents[] vector. Alice goes down, we should choose the best overall path.

I personally tried BFS and failed with some corner case (still unknown). The DFS worked.

#### Approach

* Bob time can be tracked in the same DFS but `post-order` (clever trick, not mine)
* default reward is Int.MIN_VALUE, not zero
* I still think the simulation BFS with marked nodes possible to run both Alice and Bob together

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun mostProfitablePath(edges: Array<IntArray>, bob: Int, amount: IntArray): Int {
        val g = Array(amount.size) { ArrayList<Int>() }; val bt = IntArray(g.size) { g.size }
        for ((a, b) in edges) { g[a] += b; g[b] += a }; bt[bob] = 0
        fun dfs(c: Int, p: Int, t: Int): Int =  (g[c].filter { it != p }
            .maxOfOrNull { e -> dfs(e, c, t + 1).also { bt[c] = min(bt[c], 1 + bt[e]) }} ?:0) +
            (if (t < bt[c]) amount[c] else if (t == bt[c]) amount[c] / 2 else 0)
        return dfs(0, -1, 0)
    }

```
```rust

    pub fn most_profitable_path(edges: Vec<Vec<i32>>, bob: i32, amount: Vec<i32>) -> i32 {
        let n = amount.len(); let (mut g, mut bt) = (vec![vec![]; n], vec![n as i32; n]);
        for e in edges.iter() { let (a, b) = (e[0] as usize, e[1] as usize); g[a].push(b); g[b].push(a)}
        fn dfs(c: usize, p: i32, t: i32, g: &Vec<Vec<usize>>, bt: &mut Vec<i32>, amount: &Vec<i32>) -> i32 {
            (g[c].iter().filter(|&&e| e != p as usize).map(|&e| { let x = dfs(e, c as i32, t + 1, g, bt, amount);
            bt[c] = bt[c].min(1 + bt[e]); x}).max().unwrap_or(0))
                + if t < bt[c] { amount[c] } else if t == bt[c] { amount[c] / 2 } else { 0 }}
        bt[bob as usize] = 0; dfs(0, -1, 0, &g, &mut bt, &amount)
    }

```
```c++

    int mostProfitablePath(vector<vector<int>>& edges, int bob, vector<int>& amount) {
        int n = amount.size(); vector<vector<int>> g(n); vector<int> bt(n, n);
        for (auto& e : edges) g[e[0]].push_back(e[1]), g[e[1]].push_back(e[0]);
        auto d = [&](this const auto d, int c, int p, int t) -> int {
            int mx = INT_MIN;
            for (int e : g[c]) if (e != p) { mx = max(mx, d(e, c, t + 1)); bt[c] = min(bt[c], bt[e] + 1);}
            return (mx == INT_MIN ? 0 : mx) + (t < bt[c] ? amount[c] : (t == bt[c] ? amount[c] / 2 : 0)); };
        bt[bob] = 0; return d(0, -1, 0);
    }

```


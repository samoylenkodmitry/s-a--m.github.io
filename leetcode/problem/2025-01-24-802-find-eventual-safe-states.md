---
layout: leetcode-entry
title: "802. Find Eventual Safe States"
permalink: "/leetcode/problem/2025-01-24-802-find-eventual-safe-states/"
leetcode_ui: true
entry_slug: "2025-01-24-802-find-eventual-safe-states"
---

[802. Find Eventual Safe States](https://leetcode.com/problems/find-eventual-safe-states/description/) medium
[blog post](https://leetcode.com/problems/find-eventual-safe-states/solutions/6323313/kotlin-rust-by-samoylenkodmitry-b0k7/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24012025-802-find-eventual-safe-states?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/EIMorpk32wc)
![1.webp](/assets/leetcode_daily_images/3577d67c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/874

#### Problem TLDR

Nodes without cycles #medium #dfs #toposort

#### Intuition

The problem description was misleading. The actual task is to filter out cycles.

Simple DFS with memoization works.

Why does the Topological Sort work? Example:

```j

// [2, 2] [0] [3] []  <-- not valid input, [2,2],
//                        graph [i] must be strictly increasing
// [1] [0] [3] []
// 0 -> 1
// 1 -> 0
// 2 -> 3
// 3 -> .        reverse: 3 -> [2], 2 -> [], 1 -> [0], 0 -> [1]
//        0 1 2 3
// deg:   1 1 1 0
// take 3->[2]
// deg:   1 1 0 0
// take 2->[] end

```

As we can see, `in-degrees for cycles are always > 0`.

#### Approach

* let's implement both DFS and Toposort.

#### Complexity

- Time complexity:
$$O(EV)$$

- Space complexity:
$$O(E + V)$$

#### Code

```kotlin

    fun eventualSafeNodes(graph: Array<IntArray>): List<Int> {
        val safe = HashMap<Int, Boolean>()
        fun dfs(i: Int, vis: HashSet<Int>): Boolean = safe.getOrPut(i) {
            vis.add(i) && graph[i].all { dfs(it, vis) }
        }
        return graph.indices.filter { dfs(it, HashSet()) }
    }

```
```rust

    pub fn eventual_safe_nodes(graph: Vec<Vec<i32>>) -> Vec<i32> {
        let mut g = vec![vec![]; graph.len()];
        let (mut deg, mut safe) = (vec![0; g.len()], vec![false; g.len()]);
        for i in 0..g.len() {
            for &s in &graph[i] { let s = s as usize; g[s].push(i); deg[i] += 1 }
        }
        let mut q = VecDeque::from_iter((0..g.len()).filter(|&i| deg[i] == 0));
        while let Some(i) = q.pop_front() {
            safe[i] = true;
            for &s in &g[i] { deg[s] -= 1; if deg[s] == 0 { q.push_back(s) } }
        }
        (0..g.len()).filter(|&i| safe[i]).map(|i| i as i32).collect()
    }

```
```c++

    vector<int> eventualSafeNodes(vector<vector<int>>& g) {
        vector<int> s(size(g)), r;
        function<bool(int)> dfs = [&](int i) {
            if (s[i]) return s[i] == 2; s[i] = 1;
            for (int j: g[i]) if (!dfs(j)) return false;
            s[i] = 2; return true;
        };
        for (int i = 0; i < size(g); ++i) if (dfs(i)) r.push_back(i);
        return r;
    }

```


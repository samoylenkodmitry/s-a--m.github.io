---
layout: leetcode-entry
title: "2192. All Ancestors of a Node in a Directed Acyclic Graph"
permalink: "/leetcode/problem/2024-06-29-2192-all-ancestors-of-a-node-in-a-directed-acyclic-graph/"
leetcode_ui: true
entry_slug: "2024-06-29-2192-all-ancestors-of-a-node-in-a-directed-acyclic-graph"
---

[2192. All Ancestors of a Node in a Directed Acyclic Graph](https://leetcode.com/problems/all-ancestors-of-a-node-in-a-directed-acyclic-graph/description/) medium
[blog post](https://leetcode.com/problems/all-ancestors-of-a-node-in-a-directed-acyclic-graph/solutions/5385624/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29062024-2192-all-ancestors-of-a?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/OwYS_oe6DWQ)
![2024-06-29_08-11_1.webp](/assets/leetcode_daily_images/d5177c89.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/654

#### Problem TLDR

List of ancestors in a DAG #medium #dfs #toposort

#### Intuition

![2024-06-29_08-14.webp](/assets/leetcode_daily_images/a6750a0e.webp)
We can use Depth-First Search for each node, caching the result to not execute twice, but we should walk backwards from child to parent.

Another solution is to walk from parents in a Topological Sort order and appending the results.

#### Approach

Let's implement both approaches.
For the toposort solution (in Rust), we should do deduplication as early as possible to prevent OOM.

#### Complexity

- Time complexity:
$$O(E^2V + V^2log(V))$$ for DFS - groupBy will take O(E), DFS depth is O(E) and inside it we iterate over each sibling O(X), X is up to E where we do copy of all collected vertices O(V). The final step is sorting V collected vertexes - VlogV.

$$O(V + EVlog(V))$$, the Kahn algorithm for toposort takes O(V + E), in each step of edge taking we append V vertices, and sorting them Vlog(V)

- Space complexity:
$$O(V^2 + E)$$ result takes the biggest space

#### Code

```kotlin

    fun getAncestors(n: Int, edges: Array<IntArray>): List<List<Int>> {
        val g = edges.groupBy({ it[1] }, { it[0] })
        val res = mutableMapOf<Int, Set<Int>>()
        fun dfs(i: Int): Set<Int> = res.getOrPut(i) {
            g[i]?.map { dfs(it) + it }?.flatten()?.toSet() ?: setOf()
        }
        return (0..<n).map { dfs(it).sorted() }
    }

```
```rust

    pub fn get_ancestors(n: i32, edges: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        let n = n as usize; let (mut deg, mut g, mut res, mut queue) =
            (vec![0; n], vec![vec![]; n], vec![vec![]; n], VecDeque::new());
        for e in edges {
            g[e[0] as usize].push(e[1] as usize); deg[e[1] as usize] += 1
        }
        for i in 0..n { if deg[i] == 0 { queue.push_back(i); }}
        while let Some(top) = queue.pop_front() { for &j in &g[top] {
            deg[j] -= 1; if deg[j] == 0 { queue.push_back(j); }
            res[j].push(top as i32); let t = res[top].clone();
            res[j].extend(t); res[j].sort_unstable(); res[j].dedup()
        }}; res
    }

```


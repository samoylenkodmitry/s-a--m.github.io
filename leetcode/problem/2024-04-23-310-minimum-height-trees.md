---
layout: leetcode-entry
title: "310. Minimum Height Trees"
permalink: "/leetcode/problem/2024-04-23-310-minimum-height-trees/"
leetcode_ui: true
entry_slug: "2024-04-23-310-minimum-height-trees"
---

[310. Minimum Height Trees](https://leetcode.com/problems/minimum-height-trees/description/) medium
[blog post](https://leetcode.com/problems/minimum-height-trees/solutions/5061843/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23042024-310-minimum-height-trees?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/UvkjYhS74_o)
![2024-04-23_10-25.webp](/assets/leetcode_daily_images/6915790c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/581

#### Problem TLDR

Center of an acyclic graph #medium #graph #toposort

#### Intuition

Didn't solve it myself again.

The naive intuition that didn't work for me was to move from the edges in BFS manner until a single or just two nodes left. This however doesn't work for some cases:
![2024-04-23_09-07.webp](/assets/leetcode_daily_images/66bc3978.webp)

After I gave up, in the solution section I saw a Topological Sort: always go from nodes with `indegree == 1` and decrease it as you go.

There is also a `two-dfs` solution exists, it's very clever: do two dfs runs from leaf to leaf and choose two middles of thier paths.

#### Approach

* careful with order of decreasing indegree: first decrease, then check for == 1.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun findMinHeightTrees(n: Int, edges: Array<IntArray>): List<Int> {
        val graph = mutableMapOf<Int, MutableList<Int>>()
        val indegree = IntArray(n)
        for ((a, b) in edges) {
            indegree[a]++
            indegree[b]++
            graph.getOrPut(a) { mutableListOf() } += b
            graph.getOrPut(b) { mutableListOf() } += a
        }
        var layer = mutableListOf<Int>()
        for (x in 0..<n) if (indegree[x] < 2) {
            layer += x; indegree[x]--
        }
        while (layer.size > 1) {
            val next = mutableListOf<Int>()
            for (x in layer) for (y in graph[x]!!) {
                indegree[y]--
                if (indegree[y] == 1) next += y
            }
            if (next.size < 1) break
            layer = next
        }
        return layer
    }

```
```rust

    pub fn find_min_height_trees(n: i32, edges: Vec<Vec<i32>>) -> Vec<i32> {
        let mut graph = HashMap::new();
        let mut indegree = vec![0; n as usize];
        for e in edges {
            indegree[e[0] as usize] += 1;
            indegree[e[1] as usize] += 1;
            graph.entry(e[0]).or_insert(vec![]).push(e[1]);
            graph.entry(e[1]).or_insert(vec![]).push(e[0])
        }
        let mut layer = vec![];
        for x in 0..n as usize { if indegree[x] < 2 {
            layer.push(x as i32); indegree[x] -= 1
        }}
        while layer.len() > 1 {
            let mut next = vec![];
            for x in &layer { if let Some(nb) = graph.get(&x) {
                for &y in nb {
                    indegree[y as usize] -= 1;
                    if indegree[y as usize] == 1 { next.push(y) }
                }
            }}
            if next.len() < 1 { break }
            layer = next
        }
        layer
    }

```


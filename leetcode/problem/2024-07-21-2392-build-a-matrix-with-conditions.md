---
layout: leetcode-entry
title: "2392. Build a Matrix With Conditions"
permalink: "/leetcode/problem/2024-07-21-2392-build-a-matrix-with-conditions/"
leetcode_ui: true
entry_slug: "2024-07-21-2392-build-a-matrix-with-conditions"
---

[2392. Build a Matrix With Conditions](https://leetcode.com/problems/build-a-matrix-with-conditions/description/) hard
[blog post](https://leetcode.com/problems/build-a-matrix-with-conditions/solutions/5510359/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21072024-2392-build-a-matrix-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/EPeWoa0srZ8)
![2024-07-21_10-34_1.webp](/assets/leetcode_daily_images/e85427c6.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/677

#### Problem TLDR

Build a matrix from a graph conditions #hard #graph #toposort

#### Intuition

Failed to solve in 1 hour.

Let's observe how the conditions work:
```j

    // k = 3
    // r=1,2 3,2
    // c=2,1 3,2
    //
    // y: 2->1, 2->3 start=2, end=1,3
    // x: 1->2, 2->3 or 1->2->3 start=1, end=3

```

Some observations:
* rules are independent for columns and rows
* some rules form a long graph nodes, so we can use toposort

So, we can apply first rows positions for each value 1..k, then apply columns' positions.

To find the positions, let's take the graph and just increment some counter from the deepest nodes to the top. It is a topological sorted order.

When graph has cycles the toposort will not visit all the nodes.

(Why I failed with a simple DFS: because the nodes are not visited in the deepest to top order)

#### Approach

Reuse the sort functions for rows and columns.

#### Complexity

- Time complexity:
$$O(E + k^2)$$

- Space complexity:
$$O(E + k^2)$$

#### Code

```kotlin

    fun buildMatrix(k: Int, rowConditions: Array<IntArray>, colConditions: Array<IntArray>): Array<IntArray> {
        fun sort(cond: Array<IntArray>): IntArray = ArrayDeque<Int>().run {
            val inOrder = IntArray(k + 1); val g = mutableMapOf<Int, MutableList<Int>>()
            for ((a, b) in cond) { inOrder[b]++; g.getOrPut(a) { mutableListOf() } += b }
            for (v in 1..k) if (inOrder[v] == 0) add(v)
            val res = IntArray(k + 1); var i = 0
            while (size > 0) removeFirst().let { v ->
                res[v] = i++; g[v]?.forEach { if (--inOrder[it] == 0) add(it) }
            }
            if (i < k) intArrayOf() else res
        }
        val r = sort(rowConditions); val c = sort(colConditions)
        if (r.size < 1 || c.size < 1) return arrayOf()
        val res = Array(k) { IntArray(k) }; for (v in 1..k) res[r[v]][c[v]] = v
        return res
    }

```
```rust

    pub fn build_matrix(k: i32, row_conditions: Vec<Vec<i32>>, col_conditions: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        fn sort(k: usize, cond: &Vec<Vec<i32>>) -> Vec<usize> {
            let (mut i, mut ins, mut g, mut queue, mut res) =
                (0, vec![0; k + 1], HashMap::new(), VecDeque::new(), vec![0; k + 1]);
            for c in cond {
                ins[c[1] as usize] += 1; g.entry(c[0] as usize).or_insert_with(|| vec![]).push(c[1] as usize);
            }
            for v in 1..=k { if ins[v] == 0 { queue.push_back(v); } }
            while let Some(v) = queue.pop_front() {
                res[v] = i; i += 1;
                if let Some(sibl) = g.remove(&v) { for e in sibl {
                    ins[e] -= 1; if ins[e] == 0 { queue.push_back(e); }
                }}
            }
            if i < k { vec![] } else { res }
        }
        let k = k as usize; let r = sort(k, &row_conditions); let c = sort(k, &col_conditions);
        if r.len() < 1 || c.len() < 1 { return vec![] }
        let mut res = vec![vec![0; k]; k];
        for v in 1..=k { res[r[v]][c[v]] = v as i32 }
        res
    }

```


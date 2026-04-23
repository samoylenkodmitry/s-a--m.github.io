---
layout: leetcode-entry
title: "1514. Path with Maximum Probability"
permalink: "/leetcode/problem/2024-08-31-1514-path-with-maximum-probability/"
leetcode_ui: true
entry_slug: "2024-08-31-1514-path-with-maximum-probability"
---

[1514. Path with Maximum Probability](https://leetcode.com/problems/path-with-maximum-probability/description/) medium
[blog post](https://leetcode.com/problems/path-with-maximum-probability/solutions/5713978/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/31082024-1514-path-with-maximum-probability?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/yrc4aAnOp-g)
![1.webp](/assets/leetcode_daily_images/707c66d5.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/719

#### Problem TLDR

Max path in graph #medium #graph

#### Intuition

Several ways to solve it:

* Dijkstra, use array [0..n] of probabilities, from `start_node` to `i_node`, put in queue while the situation is improving
* A*, use `PriorityQueue` (or `BinaryHeap`) and consider the paths with the largest probabilities so far, stop on the first arrival
* Bellman-Ford, improve the situation `n` times or until it stops improving (the N boundary can be proved, path without loops visits at most N nodes)

#### Approach

* let's write the shortest code possible
* we should use `fold`, as `any`, `all` and `none` are stopping early

#### Complexity

- Time complexity:
$$O(VE)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun maxProbability(n: Int, edges: Array<IntArray>, succProb: DoubleArray, start_node: Int, end_node: Int): Double {
        val pb = DoubleArray(n); pb[start_node] = 1.0
        for (i in 0..n) if (!edges.withIndex().fold(false) { r, (i, e) ->
            val a = pb[e[0]]; val b = pb[e[1]]
            pb[e[0]] = max(a, succProb[i] * b); pb[e[1]] = max(b, succProb[i] * a)
            r || pb[e[0]] > a || pb[e[1]] > b
        }) break
        return pb[end_node]
    }

```
```rust

    pub fn max_probability(n: i32, edges: Vec<Vec<i32>>, succ_prob: Vec<f64>, start_node: i32, end_node: i32) -> f64 {
        let mut pb = vec![0f64; n as usize]; pb[start_node as usize] = 1f64;
        loop { if !edges.iter().zip(succ_prob.iter()).fold(false, |r, (e, p)| {
            let (e0, e1) = (e[0] as usize, e[1] as usize); let (a, b) = (pb[e0], pb[e1]);
            pb[e0] = pb[e0].max(pb[e1] * p); pb[e1] = pb[e1].max(pb[e0] * p);
            r || a < pb[e0] || b < pb[e1]
        }) { break }}
        pb[end_node as usize]
    }

```


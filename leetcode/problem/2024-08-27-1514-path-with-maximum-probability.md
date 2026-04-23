---
layout: leetcode-entry
title: "1514. Path with Maximum Probability"
permalink: "/leetcode/problem/2024-08-27-1514-path-with-maximum-probability/"
leetcode_ui: true
entry_slug: "2024-08-27-1514-path-with-maximum-probability"
---

[1514. Path with Maximum Probability](https://leetcode.com/problems/path-with-maximum-probability/description/) medium
[blog post](https://leetcode.com/problems/path-with-maximum-probability/solutions/5696750/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27082024-1514-path-with-maximum-probability?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/HmtraQZOI80)
![1.webp](/assets/leetcode_daily_images/e998cf6b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/715

#### Problem TLDR

Max path in a weighted graph #medium #graph

#### Intuition

There is a standard algorithm for finding all the shortest paths from one node to any other nodes - Bellman-Ford Algorithm. Only visit the nodes that are improving the situation.

#### Approach

* we can store each paths' probability in the queue, or just reuse what is in `pb` array

#### Complexity

- Time complexity:
$$O(EV)$$

- Space complexity:
$$O(EV)$$

#### Code

```kotlin

    fun maxProbability(n: Int, edges: Array<IntArray>, succProb: DoubleArray, start_node: Int, end_node: Int): Double {
        val pb = DoubleArray(n + 1); val g = mutableMapOf<Int, MutableList<Pair<Int, Double>>>()
        for ((i, e) in edges.withIndex()) {
            g.getOrPut(e[0]) { mutableListOf() } += e[1] to succProb[i]
            g.getOrPut(e[1]) { mutableListOf() } += e[0] to succProb[i]
        }
        val queue = ArrayDeque<Pair<Int, Double>>(); queue += start_node to 1.0
        while (queue.size > 0) {
            val (curr, p) = queue.removeFirst()
            if (p <= pb[curr]) continue
            pb[curr] = p
            g[curr]?.onEach { (sibl, prob) -> queue += sibl to p * prob }
        }
        return pb[end_node]
    }

```
```rust

    pub fn max_probability(n: i32, edges: Vec<Vec<i32>>, succ_prob: Vec<f64>, start_node: i32, end_node: i32) -> f64 {
        let (mut pb, mut g, mut queue) = (vec![0f64; 1 + n as usize], HashMap::new(), VecDeque::from([start_node]));
        for (i, e) in edges.into_iter().enumerate() {
            g.entry(e[0]).or_insert_with(|| vec![]).push((e[1], succ_prob[i]));
            g.entry(e[1]).or_insert_with(|| vec![]).push((e[0], succ_prob[i]));
        }
        pb[start_node as usize] = 1f64;
        while let Some(curr) = queue.pop_front() {
            for &(sibl, prob) in g.get(&curr).unwrap_or(&vec![]) {
                if pb[sibl as usize] < pb[curr as usize] * prob {
                    pb[sibl as usize] = pb[curr as usize] * prob; queue.push_back(sibl);
                }
            }
        }; pb[end_node as usize]
    }

```


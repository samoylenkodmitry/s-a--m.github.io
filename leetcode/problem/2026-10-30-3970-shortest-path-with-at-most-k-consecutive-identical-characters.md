---
layout: leetcode-entry
title: "3970. Shortest Path With At Most K Consecutive Identical Characters"
permalink: "/leetcode/problem/2026-10-30-3970-shortest-path-with-at-most-k-consecutive-identical-characters/"
leetcode_ui: true
entry_slug: "2026-10-30-3970-shortest-path-with-at-most-k-consecutive-identical-characters"
---

[3970. Shortest Path With At Most K Consecutive Identical Characters](https://leetcode.com/problems/shortest-path-with-at-most-k-consecutive-identical-characters/solutions/8594815/kotlin-rust-by-samoylenkodmitry-8pan/) medium
[substack](https://dmitriisamoilenko.substack.com/p/30102026-3970-shortest-path-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/OL-TKrZx1Xc)

https://dmitrysamoylenko.com/leetcode/

![30.10.2026.webp](/assets/leetcode_daily_images/30.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1527

#### Problem TLDR

Min path in weighted graph having at most K chars repeats

#### Intuition

Dijkstra: state is (node, count), storing the chars is not required. Put in PriorityQueue sorted by smallest weight.

#### Approach

* attention to fileter out steps out of k
* visited set is a HashSet of states

#### Complexity

- Time complexity:
$$O(nklog(nk))$$

- Space complexity:
$$O(nk)$$

#### Code

```kotlin
    fun shortestPath(n: Int, e: Array<IntArray>, l: String, k: Int): Int {
        val g = e.groupBy { it[0] }; val vis = HashSet<Int>()
        val q = PriorityQueue<IntArray>(compareBy { it[0] }); q += intArrayOf(0, 1, 0)
        while (q.isNotEmpty()) {
            val (w, c, i) = q.poll(); if (c > k) continue; if (i == n - 1) return w
            if (vis.add(i * (k + 1) + c))
                g[i]?.forEach { (_, j, wj) -> q += intArrayOf(w + wj, if (l[i] == l[j]) c + 1 else 1, j) }
        }
        return -1
    }
```
```rust
    pub fn shortest_path(n: i32, e: Vec<Vec<i32>>, l: String, k: i32) -> i32 {
        let (g, mut q, mut v, l) = (e.into_iter().into_group_map_by(|x| x[0]),
            BinaryHeap::from([(0, 1, 0)]), HashSet::new(), l.as_bytes());
        while let Some((w, c, i)) = q.pop() {
            if c <= k && v.insert((i, c)) { if i == n - 1 { return -w }
                for x in g.get(&i).into_iter().flatten() {
                    q.push((w - x[2], if l[i as usize] == l[x[1] as usize] { c + 1 } else { 1 }, x[1]))
                }
            }
        }
        -1
    }
```


---
layout: leetcode-entry
title: "3243. Shortest Distance After Road Addition Queries I"
permalink: "/leetcode/problem/2024-11-27-3243-shortest-distance-after-road-addition-queries-i/"
leetcode_ui: true
entry_slug: "2024-11-27-3243-shortest-distance-after-road-addition-queries-i"
---

[3243. Shortest Distance After Road Addition Queries I](https://leetcode.com/problems/shortest-distance-after-road-addition-queries-i/description/) medium
[blog post](https://leetcode.com/problems/shortest-distance-after-road-addition-queries-i/solutions/6088161/kotlin-rust/)
[substack](https://notebooklm.google.com/notebook/c40394e3-f653-4676-b10e-9c6843090ef1/audio)
[youtube](https://youtu.be/7dTgKkTZO_4)
[deep-dive](https://open.substack.com/pub/dmitriisamoilenko/p/27112024-3243-shortest-distance-after?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
![1.webp](/assets/leetcode_daily_images/e433296e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/813

#### Problem TLDR

Query shortest paths after adding new edges #medium #bfs

#### Intuition

Unidirectional - one way. (spent 10 minutes on this)

The problem size is small, the simple BFS for each query is accepted.

Some optimizations:

* we can preserve the `lengths` array and only observe improved edges
* we can start at the end of the added node

Another angle of thinking from Vlad (https://leetcode.com/problems/shortest-distance-after-road-addition-queries-i/solutions/5583452/dp/):
* for each new edge [a,b] improve all [b..n] nodes lengths and siblings of each

#### Approach

* pay attention to suspicous words

#### Complexity

- Time complexity:
$$O(qn)$$

- Space complexity:
$$O(q + n)$$

#### Code

```kotlin

    fun shortestDistanceAfterQueries(n: Int, queries: Array<IntArray>): IntArray {
        val g = Array(n) { mutableListOf(min(n - 1, it + 1)) }
        val len = IntArray(n) { it }; val q = ArrayDeque<Pair<Int, Int>>()
        return queries.map { (a, b) ->
            g[a] += b; q += b to len[a] + 1
            while (q.size > 0) {
                val (x, s) = q.removeFirst()
                if (len[x] <= s) continue
                len[x] = s
                for (sibl in g[x]) q += sibl to s + 1
            }
            len[n - 1]
        }.toIntArray()
    }

```
```rust

    pub fn shortest_distance_after_queries(n: i32, queries: Vec<Vec<i32>>) -> Vec<i32> {
        let n = n as usize;
        let (mut len, mut q) = ((0..n).collect::<Vec<_>>(), vec![]);
        let mut g: Vec<_> = (0..n).map(|i| vec![(n - 1).min(i + 1)]).collect();
        queries.iter().map(|e| {
            let a = e[0] as usize; let b = e[1] as usize;
            g[a].push(b); q.push((b, 1 + len[a]));
            while let Some((x, s)) = q.pop() {
                if len[x] <= s { continue }
                len[x] = s;
                q.extend(g[x].iter().map(|sibl| (*sibl, 1 + s)))
            }; len[n - 1] as i32
        }).collect()
    }

```
```c++

    vector<int> shortestDistanceAfterQueries(int n, vector<vector<int>>& queries) {
        vector<int> d(n); for (int i = n; i--;) d[i] = i;
        vector<int> r; vector<vector<int>> g(n);
        for (auto e: queries) {
            int a = e[0], b = e[1];
            g[b].push_back(a);
            for (int x = b; x < n; ++x) {
                d[x] = min(d[x], d[x - 1] + 1);
                for (int sibl: g[x]) d[x] = min(d[x], d[sibl] + 1);
            }
            r.push_back(d[n - 1]);
        } return r;
    }

```


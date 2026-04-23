---
layout: leetcode-entry
title: "2503. Maximum Number of Points From Grid Queries"
permalink: "/leetcode/problem/2025-03-28-2503-maximum-number-of-points-from-grid-queries/"
leetcode_ui: true
entry_slug: "2025-03-28-2503-maximum-number-of-points-from-grid-queries"
---

[2503. Maximum Number of Points From Grid Queries](https://leetcode.com/problems/maximum-number-of-points-from-grid-queries/description/) hard
[blog post](https://leetcode.com/problems/maximum-number-of-points-from-grid-queries/solutions/6588933/kotlin-rust-by-samoylenkodmitry-bx5m/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28032025-2503-maximum-number-of-points?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/LqjyXlL10Go)
![1.webp](/assets/leetcode_daily_images/46e56f0f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/941

#### Problem TLDR

Running queries of increasing BFS #hard #bfs #heap

#### Intuition

Sort queries, then do a BFS.

#### Approach

* we can make a separate non-sorted queue for the current run, extra code, small gains
* loop through queries is more elegant than a single bfs loop with adjusting queries index

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun maxPoints(g: Array<IntArray>, queries: IntArray): IntArray {
        val q = PriorityQueue<Pair<Int, Int>>(compareBy { (y, x) -> g[y][x] })
        q += 0 to 0; val res = IntArray(queries.size); val visited = HashSet<Pair<Int, Int>>()
        for (i in queries.indices.sortedBy { queries[it] }) {
            while (q.size > 0 && g[q.peek().first][q.peek().second] < queries[i]) {
                val (y, x) = q.poll()
                if (!visited.add(y to x)) continue
                for ((dx, dy) in listOf(0, -1, 0, 1, 0).windowed(2))
                    if (x + dx in 0..<g[0].size && y + dy in 0..<g.size) q += y + dy to x + dx
            }
            res[i] = visited.size
        }
        return res
    }

```
```rust

    pub fn max_points(mut g: Vec<Vec<i32>>, queries: Vec<i32>) -> Vec<i32> {
        let (mut q, mut res, mut cnt) = (BinaryHeap::from_iter([(-g[0][0], 0, 0)]), vec![0; queries.len()], 0);
        let mut idx: Vec<_> = (0..queries.len()).collect(); idx.sort_unstable_by_key(|&x| queries[x]);
        for i in idx {
            while let Some(&(v, y, x)) = q.peek() {
                if -v >= queries[i] { break }; q.pop(); if g[y][x] < 0 { continue }; g[y][x] = -1; cnt += 1;
                for (i, j) in [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)] {
                    if i < g.len() && j < g[0].len() { q.push((-g[i][j], i, j)) }}
            }
            res[i] = cnt;
        } res
    }

```
```c++

    vector<int> maxPoints(vector<vector<int>>& g, vector<int>& q) {
        priority_queue<tuple<int, int, int>, vector<tuple<int, int, int>>, greater<>> pq;
        vector<int> r(q.size()), ix(q.size()); int c = 0, i = 0;
        iota(ix.begin(), ix.end(), 0); sort(ix.begin(), ix.end(), [&](int a, int b) { return q[a] < q[b]; });
        for (pq.emplace(g[0][0], 0, 0); auto k : ix) {
            while (!pq.empty() && get<0>(pq.top()) < q[k]) {
                auto [v, y, x] = pq.top(); pq.pop();
                if (g[y][x] >= 0 && (g[y][x] = -++c))
                    for (auto [d, e] : {pair{1, 0}, {-1, 0}, {0, 1}, {0, -1}})
                        if ((y + d) < g.size() && (x + e) < g[0].size())
                            pq.emplace(g[y + d][x + e], y + d, x + e);
            }
            r[k] = c;
        }
        return r;
    }

```


---
layout: leetcode-entry
title: "3446. Sort Matrix by Diagonals"
permalink: "/leetcode/problem/2025-08-28-3446-sort-matrix-by-diagonals/"
leetcode_ui: true
entry_slug: "2025-08-28-3446-sort-matrix-by-diagonals"
---

[3446. Sort Matrix by Diagonals](https://leetcode.com/problems/sort-matrix-by-diagonals/description/) medium
[blog post](https://leetcode.com/problems/sort-matrix-by-diagonals/solutions/7130785/kotlin-rust-by-samoylenkodmitry-cqln/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28082025-3446-sort-matrix-by-diagonals?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/LTqpki6XuQU)
![1.webp](/assets/leetcode_daily_images/c860e553.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1095

#### Problem TLDR

Sort bottom left and top right diagonals #medium #matrix

#### Intuition

Problem is small, put them into lists, sort, then put back.

#### Approach

* iterate over `y` for bottom-left, `x` for top-right
* priority queue gives a compact code

#### Complexity

- Time complexity:
$$O(n^2logn)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 18ms
    fun sortMatrix(g: Array<IntArray>) = g.apply {
        val m = HashMap<Int, PriorityQueue<Int>>()
        for (y in indices) for (x in g[0].indices) m.getOrPut(y-x)
            { PriorityQueue { a,b -> if (y < x) a - b else b - a }} += g[y][x]
        for (y in indices) for (x in g[0].indices) g[y][x] = m[y-x]!!.poll()
    }

```
```kotlin

// 35ms
    fun sortMatrix(g: Array<IntArray>) = g.apply {
        for (y in indices) {
            val ns = (0..<g.size-y).map { g[y+it][it] }.sortedDescending()
            for (x in ns.indices) g[y+x][x] = ns[x]
        }
        for (x in 1..<g[0].size) {
            val ns = (0..<g[0].size-x).map { g[it][x+it] }.sorted()
            for (y in ns.indices) g[y][x+y] = ns[y]
        }
    }

```
```rust

// 0ms
    pub fn sort_matrix(mut g: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        for (x0, y0) in (0..g.len()).map(|y| (y, 0)).chain((1..g[0].len()).map(|x| (0, x))) {
            let (mut x, mut y, mut v) = (x0, y0, vec![]);
            while x < g[0].len() && y < g.len() { v.push(g[y][x]); y += 1; x += 1 }
            v.sort_unstable(); if y >= x { v.reverse() }; (x, y) = (x0, y0);
            for v in v { g[y][x] = v; y += 1; x += 1 }
        } g
    }

```
```c++

// 3ms
    vector<vector<int>> sortMatrix(vector<vector<int>>& g) {
        int n = size(g), m = size(g[0]), b = n - 1;
        vector<priority_queue<int>> q(n+m-1);
        for (int y = 0; y < n; ++y) for (int x = 0; x < m; ++x) q[x-y+b].push(y<x?-g[y][x]:g[y][x]);
        for (int y = 0; y < n; ++y) for (int x = 0; x < m; ++x)
        { int d = x - y + b, v = q[d].top(); q[d].pop(); g[y][x]=y<x?-v:v; } return g;
    }

```
```python

// 12ms
    def sortMatrix(_, g):
        d = {}
        [[heappush(d.setdefault(y-x, []), t*(1,-1)[y>=x]) for x,t in enumerate(r)] for y,r in enumerate(g)]
        g[:] = [[(heappop(d[y-x])*(1,-1)[y>=x]) for x in range(len(r))] for y,r in enumerate(g)]
        return g

```


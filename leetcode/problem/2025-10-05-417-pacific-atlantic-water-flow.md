---
layout: leetcode-entry
title: "417. Pacific Atlantic Water Flow"
permalink: "/leetcode/problem/2025-10-05-417-pacific-atlantic-water-flow/"
leetcode_ui: true
entry_slug: "2025-10-05-417-pacific-atlantic-water-flow"
---

[417. Pacific Atlantic Water Flow](https://leetcode.com/problems/pacific-atlantic-water-flow/description) medium
[blog post](https://leetcode.com/problems/pacific-atlantic-water-flow/solutions/7250742/kotlin-rust-by-samoylenkodmitry-ffx7/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05102025-417-pacific-atlantic-water?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/LUF__exWhd0)

![1 (4).webp](/assets/leetcode_daily_images/8c338f53.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1133

#### Problem TLDR

Cells travel to TL&BR in decrease order #medium #dfs

#### Intuition

Think in reverse: travel from oceans in increase order and mark with `1` and `2`. Collect both marks.

#### Approach

* you already can collect on a second DFS run
* the marking DFS can run in any order
* for BFS: put (y,x,mask) to queue, go from walls, same logic

#### Complexity

- Time complexity:
$$O()$$

- Space complexity:
$$O()$$

#### Code

```kotlin

// 31ms
    fun pacificAtlantic(h: Array<IntArray>) = buildList<List<Int>> {
        val v = Array(h.size) { IntArray(h[0].size) }
        fun dfs(y: Int, x: Int, m: Int) {
            if (v[y][x] and m > 0) return; v[y][x] = v[y][x] or m
            if (v[y][x] > 2) add(listOf(y, x))
            for ((r,u) in listOf(-1,0,1,0,-1).zipWithNext())
            if (x+r in h[0].indices && y+u in h.indices && h[y+u][x+r]>=h[y][x]) dfs(y+u,x+r,m)
        }
        for (y in h.indices) { dfs(y, 0, 1); dfs(y, h[0].size-1, 2) }
        for (x in h[0].indices) { dfs(0, x, 1); dfs(h.size-1, x, 2) }
    }

```
```rust

// 0ms
    pub fn pacific_atlantic(h: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        let (m,n) = (h.len(),h[0].len()); let mut q = Vec::with_capacity(m*n);
        let (mut v,mut res) = (vec![0u8;m*n],vec![]);
        for i in 0..m { q.push((i,0,1)); q.push((i,n-1,2)) }
        for i in 0..n { q.push((0,i,1)); q.push((m-1,i,2)) }
        while let Some((y,x,b)) = q.pop() {
            if v[y*n+x]&b<1 { v[y*n+x] |= b; if v[y*n+x]>2 { res.push(vec![y as i32,x as i32]) }
            for (u,r) in [(y-1,x),(y+1,x),(y,x-1),(y,x+1)] {
                if u<m && r<n && h[u][r] >= h[y][x] { q.push((u,r,b)) }
        }}} res
    }

```


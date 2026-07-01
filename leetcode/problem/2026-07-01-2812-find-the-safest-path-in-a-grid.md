---
layout: leetcode-entry
title: "2812. Find the Safest Path in a Grid"
permalink: "/leetcode/problem/2026-07-01-2812-find-the-safest-path-in-a-grid/"
leetcode_ui: true
entry_slug: "2026-07-01-2812-find-the-safest-path-in-a-grid"
---

[2812. Find the Safest Path in a Grid](https://leetcode.com/problems/find-the-safest-path-in-a-grid/solutions/8369344/kotlin-rust-by-samoylenkodmitry-eg3w/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/01072026-2812-find-the-safest-path?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/fsdF1ZmGKBE)

https://dmitrysamoylenko.com/leetcode/

![01.07.2026.webp](/assets/leetcode_daily_images/01.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1407

#### Problem TLDR

Safest path from ones

#### Intuition

BFS from all ones to mark cells safety. Second BFS to find safest path with PriorityQueue

#### Approach

* we can reuse PriorityQueue (but it costs)
* we can use the grid as a storage

#### Complexity

- Time complexity:
$$O(n^2logn)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin
    fun maximumSafenessFactor(G: List<List<Int>>): Int {
        val n = G.size; val q = PriorityQueue<IntArray> { a, b -> b[0] - a[0] }
        val g = Array(n) { IntArray(n) { -1 } }; var p = 0
        for (y in 0..<n) for (x in 0..<n) if (G[y][x] > 0) q += intArrayOf(0, x, y)
        while (p < 2) {
            if (q.isEmpty()) { p++; q += intArrayOf(g[0][0], 0, 0); g[0][0] = -1; continue }
            val (s, x, y) = q.poll()
            if (p == 0) {
                if (g[y][x] < 0) { g[y][x] = -s; for (k in 0..3) { val X=x+(k%2)*(k-2); val Y=y+(1-k%2)*(k-1)
                    if (X in 0..<n && Y in 0..<n) q += intArrayOf(s - 1, X, Y) } }
            } else if (x == n - 1 && y == n - 1) return s
            else for (k in 0..3) { val X=x+(k%2)*(k-2); val Y=y+(1-k%2)*(k-1)
                if (X in 0..<n && Y in 0..<n && g[Y][X]>=0) { q += intArrayOf(min(s,g[Y][X]),X,Y); g[Y][X]=-1}}
        }
        return 0
    }
```
```rust
    pub fn maximum_safeness_factor(mut g: Vec<Vec<i32>>) -> i32 {
        let (n, mut q, mut h) = (g.len(), VecDeque::new(), BinaryHeap::new());
        for y in 0..n { for x in 0..n { if g[y][x] > 0 { q.push_back((y, x)) } } }
        while let Some((y, x)) = q.pop_front() { for (a, b) in [(1,0),(!0,0),(0,1),(0,!0)] {
            let (y2, x2) = (y.wrapping_add(a), x.wrapping_add(b));
            if y2 < n && x2 < n && g[y2][x2] < 1 { g[y2][x2] = g[y][x] + 1; q.push_back((y2, x2)) }
        }}
        h.push((g[0][0], 0, 0)); g[0][0] *= -1;
        while let Some((f, y, x)) = h.pop() { if y == n - 1 && x == n - 1 { return f - 1 } for (a, b) in [(1,0),(!0,0),(0,1),(0,!0)] {
            let (y, x) = (y.wrapping_add(a), x.wrapping_add(b));
            if y < n && x < n && g[y][x] > 0 { h.push((f.min(g[y][x]), y, x)); g[y][x] *= -1 }
        }}
        -1
    }
```


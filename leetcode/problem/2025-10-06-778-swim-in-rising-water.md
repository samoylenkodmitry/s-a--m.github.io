---
layout: leetcode-entry
title: "778. Swim in Rising Water"
permalink: "/leetcode/problem/2025-10-06-778-swim-in-rising-water/"
leetcode_ui: true
entry_slug: "2025-10-06-778-swim-in-rising-water"
---

[778. Swim in Rising Water](https://leetcode.com/problems/swim-in-rising-water/description) hard
[blog post](https://leetcode.com/problems/swim-in-rising-water/solutions/7253132/kotlin-rust-by-samoylenkodmitry-gk6h/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06102025-778-swim-in-rising-water?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/LZ5U90g4VkE)

![2b53bb4d-e70e-4e33-9ffa-ab8b910dab88 (1).webp](/assets/leetcode_daily_images/a64f9e22.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1134

#### Problem TLDR

Min time to swim to end, flooding every second #hard #bfs

#### Intuition

Iterate over time (0..50^2) and do BFS step while less than time.
Or, just put time variable in a PriorityQueue.

#### Approach

* another way is the BinarySearch: check reachability in O(n^2), do log(n) search in time
* the simple 0-1 BFS didn't work here: all times should be sorted

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 42ms
    fun swimInWater(g: Array<IntArray>): Int {
        val q = PriorityQueue<IntArray>(compareBy { it[0] })
        var r = g[0][0]; q += intArrayOf(r,0,0); g[0][0] = -1
        while (q.size > 0) {
            val (t,y,x) = q.poll(); r = max(r,t); if (y==g.size-1&&x==g[0].size-1) break
            for ((u,r) in listOf(0,1,0,-1,0).zipWithNext())
                if (x+r in g[0].indices && y+u in g.indices && g[y+u][x+r]>=0)
                    { q += intArrayOf(g[y+u][x+r],y+u,x+r); g[y+u][x+r] = -1 }
        }; return r
    }

```
```rust

// 0ms
    pub fn swim_in_water(mut g: Vec<Vec<i32>>) -> i32 {
        let (n,m,mut h) = (g.len(),g[0].len(), BinaryHeap::from([(-g[0][0],0,0)]));
        g[0][0] = -1; let mut r = 0;
        while let Some((t,y,x)) = h.pop() {
            r = r.max(-t); if (y,x) == (n-1,m-1) { break }
            for (u,r) in [(y-1,x),(y+1,x),(y,x-1),(y,x+1)] {
                if u < n && r < m && g[u][r] >= 0 { h.push((-g[u][r],u,r)); g[u][r] = -1 }
        }} r
    }

```


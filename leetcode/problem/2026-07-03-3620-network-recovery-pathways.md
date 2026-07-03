---
layout: leetcode-entry
title: "3620. Network Recovery Pathways"
permalink: "/leetcode/problem/2026-07-03-3620-network-recovery-pathways/"
leetcode_ui: true
entry_slug: "2026-07-03-3620-network-recovery-pathways"
---

[3620. Network Recovery Pathways](https://leetcode.com/problems/network-recovery-pathways/solutions/8373254/kotlin-rust-by-samoylenkodmitry-fl0s/) hard
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/03072026-3620-network-recovery-pathways?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/FOD1ETEIiyQ)

https://dmitrysamoylenko.com/leetcode/

![03.07.2026.webp](/assets/leetcode_daily_images/03.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1409

#### Problem TLDR

Max(min(paths)) in DAG

#### Intuition

Didn't solve.

```j
    // got hint from the start: binary search + dijkstra
    // 16 minute MLE 630/637
    // 20 minute wrong answer 631/637
    // 23 minute TLE 632/637
    // 35 minute: no idea how to improve time - hint: prune graph on each bs step
    //            still TLE
    // 43 minute: give up, my dijkstra TLEs
```

* binary search an answer
* use Dijkstra

#### Approach

* take nodes with smallest paths sum first, skip node if it was imporved, keep track in a separate array

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(nlogn)$$

#### Code

```kotlin
    fun findMaxPathScore(e: Array<IntArray>, o: BooleanArray, k: Long): Int {
        var l = 0; var h = Int.MAX_VALUE; val g = Array(o.size) { ArrayList<Long>() }
        for ((a, b, c) in e) if (o[a] && o[b]) g[a] += (1L*c shl 32) + b
        s@ while (l <= h) {
            val m = l + (h - l) / 2; val d = LongArray(o.size) { Long.MAX_VALUE }; d[0] = 0
            val q = PriorityQueue<LongArray>(compareBy { it[0] }); q += longArrayOf(0, 0)
            while (q.size > 0) {
                val (S, X) = q.poll(); val x = X.toInt(); if (x == o.size - 1) { l = m + 1; continue@s }
                if (S <= d[x]) for (E in g[x]) {
                    val cst = E shr 32; val y = E.toInt(); val S2 = S + cst
                    if (cst >= m && S2 <= k && S2 < d[y]) { d[y]=S2; q += longArrayOf(S2, y.toLong())}
                }
            }
            h = m - 1
        }
        return h
    }
```
```rust
    pub fn find_max_path_score(e: Vec<Vec<i32>>, o: Vec<bool>, k: i64) -> i32 {
        let (mut l, mut h, n) = (0, i32::MAX, o.len()); let mut g = vec![vec![]; n];
        for v in e { if o[v[0] as usize] & o[v[1] as usize] { g[v[0] as usize].push(((v[2] as u64)<<32)|v[1] as u64) } }
        while l <= h {
            let (m, mut d, mut q, mut ok) = (l+(h-l)/2, vec![i64::MAX; n], BinaryHeap::from([(0,0)]), false); d[0]=0;
            while let Some((s, x)) = q.pop() {
                let (s, x) = (-s, x as usize); if x == n-1 { ok=true; break }
                if s <= d[x] { for &E in &g[x] {
                    let (c, y) = ((E>>32) as i64, E as u32 as usize); let s2 = s+c;
                    if c >= m as i64 && s2 <= k && s2 < d[y] { d[y]=s2; q.push((-s2, y as u64)) }
                }}
            }
            if ok { l=m+1 } else { h=m-1 }
        } h
    }
```


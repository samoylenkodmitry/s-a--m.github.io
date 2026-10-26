---
layout: leetcode-entry
title: "3619. Count Islands With Total Value Divisible by K"
permalink: "/leetcode/problem/2026-10-26-3619-count-islands-with-total-value-divisible-by-k/"
leetcode_ui: true
entry_slug: "2026-10-26-3619-count-islands-with-total-value-divisible-by-k"
---

[3619. Count Islands With Total Value Divisible by K](https://leetcode.com/problems/count-islands-with-total-value-divisible-by-k/solutions/8589540/kotlin-rust-by-samoylenkodmitry-posu/) medium
[substack](https://dmitriisamoilenko.substack.com/p/26102026-3619-count-islands-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/RktIByCe70A)

https://dmitrysamoylenko.com/leetcode/

![26.10.2026.webp](/assets/leetcode_daily_images/26.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1523

#### Problem TLDR

Count islands sum %k in grid

#### Intuition

Do dfs, modifiy in-place to zeros

#### Approach

* Rust pattern-match Some(v @ 1..) = g.get_mut(y)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun countIslands(g: Array<IntArray>, k: Int)=run{
        fun d(y: Int, x: Int): Long = g.getOrNull(y)?.getOrNull(x)?.takeIf{it>0}
            ?.let { g[y][x]=0; it+d(y-1,x)+d(y+1,x)+d(y,x-1)+d(y,x+1)}?:0
        g.indices.sumOf{y->g[0].indices.count{val d=d(y,it); d>0&&d%k<1}}
    }
```
```rust
    pub fn count_islands(mut g: Vec<Vec<i32>>, k: i32) -> i32 {
        fn d(g: &mut [Vec<i32>], y: usize, x: usize) -> i64 {
            if let Some(v @ 1..) = g.get_mut(y).and_then(|r| r.get_mut(x)) {
                take(v) as i64 + d(g, y - 1, x) + d(g, y + 1, x) + d(g, y, x - 1) + d(g, y, x + 1)
            } else { 0 }
        }
        (0..g.len()).fold(0, |a, y| (0..g[0].len()).fold(a, |a, x| {
            let s = d(&mut g, y, x); a + (s > 0 && s % k as i64 == 0) as i32
        }))
    }
```


---
layout: leetcode-entry
title: "2943. Maximize Area of Square Hole in Grid"
permalink: "/leetcode/problem/2026-01-15-2943-maximize-area-of-square-hole-in-grid/"
leetcode_ui: true
entry_slug: "2026-01-15-2943-maximize-area-of-square-hole-in-grid"
---

[2943. Maximize Area of Square Hole in Grid](https://leetcode.com/problems/maximize-area-of-square-hole-in-grid/description) medium
[blog post](https://leetcode.com/problems/maximize-area-of-square-hole-in-grid/solutions/7496492/kotlin-rust-by-samoylenkodmitry-vbr8/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15012026-2943-maximize-area-of-square?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/FdEqql8wE5o)

![fa08aa27-6ff7-4712-a51a-bf99478a31fe (1).webp](/assets/leetcode_daily_images/d4223d1e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1238

#### Problem TLDR

Max square area after removing bars in a grid #medium #intervals

#### Intuition

Count consequtive intervals separately for horizontal and vertical bars. Then min(max(h,v))^2.

One way: sort bars, count consequtive with counter
Second way: use a HashMap(x, length) and update x-l, x+r with 1+m[x-1]+m[x+1]

#### Approach

* do +1 or just init counter with 2
* for the HashMap we can skip writing to the middle m[x]

#### Complexity

- Time complexity:
$$O(nlog(n))$$, or O(n)

- Space complexity:
$$O(1)$$ or O(n)

#### Code

```kotlin
// 22ms
    fun maximizeSquareHoleArea(n: Int, m: Int, h: IntArray, v: IntArray) =
        listOf(h,v).minOf { val m = HashMap<Int, Int>()
            1 + it.maxOf { x -> val l = m[x-1]?:0; val r = m[x+1]?:0
                (l+r+1).also{m[x-l]=it;m[x+r]=it}}
        }.let{it*it}
```
```rust
// 0ms
    pub fn maximize_square_hole_area(_: i32, _: i32, h: Vec<i32>, v: Vec<i32>) -> i32 {
        [h, v].map(|s| s.into_iter().sorted().collect::<Vec<_>>()
            .chunk_by(|a, b| b - a < 2).map(|c| c.len() + 1).max().unwrap_or(2)
        ).into_iter().min().unwrap().pow(2) as _
    }
```


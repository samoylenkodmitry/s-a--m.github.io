---
layout: leetcode-entry
title: "1401. Circle and Rectangle Overlapping"
permalink: "/leetcode/problem/2026-09-19-1401-circle-and-rectangle-overlapping/"
leetcode_ui: true
entry_slug: "2026-09-19-1401-circle-and-rectangle-overlapping"
---

[1401. Circle and Rectangle Overlapping](https://leetcode.com/problems/circle-and-rectangle-overlapping/solutions/8529142/kotlin-rust-by-samoylenkodmitry-u4mv/) medium
[substack](https://dmitriisamoilenko.substack.com/p/19092026-1401-circle-and-rectangle?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/v2bpKVLACUE)

https://dmitrysamoylenko.com/leetcode/

![19.09.2026.webp](/assets/leetcode_daily_images/19.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1487

#### Problem TLDR

Do circle intersect the resct

#### Intuition

Brute-force: check every point on the rect sides.
Clever: closest point on rect is independent of axis and is a distance to the range: Xc - Xc.clamp(X1..X2)

#### Approach

* Kotlin: coerceIn, Rust: clamp
* max(0,x1-x,x-x2) is also the distance to the range

#### Complexity

- Time complexity:
$$O(1)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun checkOverlap(r: Int, x: Int, y: Int, a: Int, b: Int, c: Int, d: Int) =
        Math.hypot(.0+maxOf(0,a-x,x-c),.0+maxOf(0,b-y,y-d))<=r
```
```rust
    pub fn check_overlap(r: i32, x: i32, y: i32, a: i32, b: i32, c: i32, d: i32) -> bool {
        (x-x.clamp(a,c)).pow(2)+(y-y.clamp(b,d)).pow(2)<=r*r
    }
```

